# RELER

RELER is a compact training codebase for text embeddings. It supports supervised ranking objectives (InfoNCE, RankNet, and LambdaLoss), joint query/document GRPO, and optional query-only GRPO against an immutable document corpus.

The public workflow is deliberately small: choose a data schema, compose a model and training configuration, train either a joint encoder or a query encoder against fixed documents, then evaluate the saved checkpoint with MTEB.

## Setup

Python 3.10 or later and a CUDA-capable PyTorch environment are required for training.

```bash
uv sync --extra dev
source .venv/bin/activate
```

## Training data

Training input is JSONL. Each record must use one of the supported schemas below.

### E2Rank listwise schema

```json
{
  "query": "user query",
  "document": ["candidate A", "candidate B", "candidate C"],
  "ranking": [2, 1, 3],
  "source": "optional-task-name"
}
```

`ranking` is a 1-indexed permutation of positions in `document`, ordered from most to least relevant. This schema retains the full teacher order and supports graded relevance, RankNet, LambdaLoss, and ranking rewards. `configs/dataset/e2rank_listwise.yaml` is the corresponding dataset configuration.

### Binary positive/negative schema

```json
{
  "query": "user query",
  "pos": ["relevant passage"],
  "neg": ["non-relevant passage 1", "non-relevant passage 2"],
  "pos_scores": [1.0],
  "neg_scores": [0.8, 0.4]
}
```

`pos_scores` and `neg_scores` are optional. The loader selects one positive and fills the rest of the candidate slate from negatives; this schema is binary-only. Preprocessed `embedding_candidates_v1` and `embedding_candidates_v2` records are also accepted when candidate-level relevance has already been materialized.

All records in a batch must be compatible with the selected `relevance_scheme`. Set the data path, candidate `slate_size`, query/document lengths, source filters, and optional per-source cap in a dataset config.

## Configuration model

Configuration files are composable YAML fragments under `configs/`:

- `configs/train/`: optimizer, precision, batch size, epochs, checkpointing, and LoRA.
- `configs/dataset/`: input schema, paths, lengths, candidate slate, and source sampling.
- `configs/model/`: backbone plus the embedding protocol.
- `configs/baseline/`: supervised objective and its hyperparameters.
- `configs/grpo/` and `configs/reward/`: GRPO policy and retrieval reward.
- `configs/corpus/`: immutable document-index settings for query-only GRPO.

`configs/examples/supervised.yaml` and `configs/examples/grpo.yaml` are complete starting points. Copy one or override its `data_path` and output fields for a concrete run.

## Supervised training

Use the unified `reler-train` CLI directly for one process, or the thin
`scripts/run_supervised.sh` wrapper for `torchrun`:

```bash
NPROC_PER_NODE=1 bash scripts/run_supervised.sh configs/examples/supervised.yaml \
  --data_path /path/to/train.jsonl \
  --output_dir checkpoints/reler-infonce
```

For an explicit composition, use `--base-train`, `--base-dataset`,
`--base-model`, and `--base-baseline`. Switch the baseline fragment to select
another objective:

```text
configs/baseline/infonce_in_batch.yaml
configs/baseline/ranknet.yaml
configs/baseline/lambdaloss.yaml
```

InfoNCE uses in-batch negatives in the supplied example. RankNet learns ordered candidate pairs, while LambdaLoss weights pairwise updates by their nDCG change.

## GRPO training

The GRPO example composes the train, dataset, model, GRPO, and reward fragments:

```bash
NPROC_PER_NODE=1 bash scripts/run_grpo.sh configs/examples/grpo.yaml \
  --data_path /path/to/train.jsonl \
  --output_dir checkpoints/reler-grpo
```

For a custom composition, pass `--base-train`, `--base-dataset`,
`--base-model`, `--base-grpo`, and `--base-reward`. The default GRPO
configuration trains joint query and document actions; reward configurations
define the retrieval signal and its candidate-pool behavior.

`NPROC_PER_NODE` controls the local process count and defaults to `8`. `NNODES` defaults to `1`. Both launchers forward other arguments to their training entrypoint.

## Fixed-corpus GRPO

Fixed-corpus training encodes documents once, then trains only the query encoder. Build a lookup directory with the same model and document protocol used to initialize training:

```bash
reler-index \
  --input /path/to/train.jsonl \
  --input-format training_candidates \
  --output-dir artifacts/reler-corpus \
  --model-config configs/model/qwen3_embedding_0.6b.yaml \
  --max-length 1024 \
  --num-shards 16
```

Raw listwise and `pos`/`neg` RELER records are accepted. Explicit `document_ids` are preferred; otherwise the same normalized-text key used by the training collator is generated. The resulting directory contains the embedding protocol, a document-to-ordinal mapping, and vector shards.

Train with the dedicated mode so index concerns remain outside ordinary GRPO:

```bash
NPROC_PER_NODE=1 bash scripts/run_fixed_grpo.sh \
  configs/examples/fixed_grpo.yaml \
  --data_path /path/to/train.jsonl \
  --index_dir artifacts/reler-corpus \
  --output_dir checkpoints/reler-fixed-grpo
```

The mode requires a query-only action policy. It never tokenizes document text during training and keeps document vectors outside autograd. Opening the directory validates its mapping, shard layout, dtype, and embedding protocol.

## Embedding invariants

The model configuration defines pooling, padding side, terminal-token handling, query and document templates, and maximum embedding length. These choices are part of the model, not merely evaluation options. Every saved checkpoint includes `embedding_protocol.json`; evaluate and deploy the checkpoint with that protocol rather than substituting templates or pooling settings.

Similarity scoring and ranking-loss contractions run in FP32, including during mixed-precision training. Keep this behavior enabled: close candidate scores can otherwise collapse into ties.

See [the embedding protocol](docs/embedding_protocol.md) and
[the GRPO objective](docs/grpo.md) for the behavioral contracts preserved by the
test suite.

## MTEB evaluation

Install the optional evaluation dependencies, then evaluate a checkpoint:

```bash
uv sync --extra evaluation
```

```bash
bash scripts/run_mteb.sh \
  checkpoints/reler-infonce \
  'MTEB(eng, v2)' \
  configs/model/qwen3_embedding_0.6b.yaml
```

Results are written under `results/mteb/`. A checkpoint containing `embedding_protocol.json` supplies its own pooling and text-formatting settings; the model config remains useful for a base model without that sidecar.

For BRIGHT, a query-only checkpoint can be evaluated against documents encoded by its immutable E0 model. The first run builds a sequential cache per subset; later runs validate its protocol and reuse it:

```bash
reler-eval \
  --model checkpoints/reler-fixed-grpo \
  --tasks BrightRetrieval \
  --langs eng \
  --output_dir results/mteb \
  --fixed_corpus_model Qwen/Qwen3-Embedding-0.6B \
  --fixed_corpus_model_revision MODEL_REVISION \
  --fixed_corpus_cache_dir artifacts/bright-e0
```

The query checkpoint's saved embedding protocol is inherited by the E0 encoder unless explicitly overridden with `--fixed_corpus_model_kwargs`.

## Repository layout

- `src/reler/`: installable package containing data, objectives, trainers, fixed-corpus support, CLI, and evaluation adapters.
- `configs/`: composable training, data, model, objective, and reward settings.
- `scripts/`: launchers and model utilities; see `scripts/README.md`.
- `tests/`: focused CPU checks for data, objectives, and protocol behavior.
