# RELER

RELER is a compact training codebase for text embeddings. It supports supervised ranking objectives (InfoNCE, RankNet, and LambdaLoss), joint query/document GRPO, and optional query-only GRPO against an immutable document corpus.

The public workflow is deliberately small: provide prepared candidate data, compose a model and training configuration, train either a joint encoder or a query encoder against fixed documents, then evaluate the saved checkpoint with MTEB.

## Setup

Python 3.10 or later and a CUDA-capable PyTorch environment are required for training.

```bash
uv sync --extra dev
source .venv/bin/activate
```

## Training data

Training input is local JSONL using only `embedding_candidates_v2`. Each record
contains the complete prepared candidate list:

```json
{
  "schema": "embedding_candidates_v2",
  "id": "sample-1",
  "source": "task-a",
  "query": "user query",
  "document": ["candidate A", "candidate B", "candidate C"],
  "document_ids": ["a", "b", "c"],
  "document_keys": ["text-key-a", "text-key-b", "text-key-c"],
  "known_document_ids": ["a", "b", "c", "d"],
  "relevance": [1, 1, 0],
  "graded_relevance": [1, 2, 3],
  "rank_labels": [1, 2, 3]
}
```

All displayed fields are required. Candidate-aligned arrays have the same length
as `document`; at least two candidates and one binary negative are required.
The first candidate is the preselected positive representative. The loader keeps
all candidates, their order, and every positive label; it does not sample or
truncate the candidate list. Variable lengths are padded and masked within a batch.

- `relevance` contains binary labels, including any additional positives.
  It defines the positive mask for contrastive learning independently of teacher labels.
- `graded_relevance` contains teacher grades in 0–3. `relevance_scheme: graded`
  uses these for ranking rewards and graded losses; `binary` uses `relevance`.
- `rank_labels` is a permutation of 1 through the candidate count, aligned with
  `document`. Larger values mean higher teacher preference; these values are
  per-document scores, not a permutation of document positions to apply.
- `document_ids` are strings scoped by `source`. `known_document_ids` lists known
  documents to exclude from cross-query pools, including candidates absent from
  this record. It does not assign positive labels.
- `document_keys` are precomputed normalized-text deduplication keys shared across
  sources (typically SHA-256, abbreviated above). They filter duplicate candidates
  and identify documents in an index built from the training data.

Use `configs/dataset/prepared.yaml` and set `data_path` to a file or directory.
Directory inputs match `file_glob` recursively. `source` always comes from each
record and controls single-source batching, optional filtering, and the per-source
sample cap. Incomplete source batches are dropped once before epoch shuffling.

## Configuration model

Configuration files are composable YAML fragments under `configs/`:

- `configs/train/`: optimizer, precision, batch size, epochs, checkpointing, and LoRA.
- `configs/dataset/`: input paths, token lengths, and source sampling.
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
  --data_path /path/to/train.ready.jsonl \
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
  --data_path /path/to/train.ready.jsonl \
  --output_dir checkpoints/reler-grpo
```

For a custom composition, pass `--base-train`, `--base-dataset`,
`--base-model`, `--base-grpo`, and `--base-reward`. The default GRPO
configuration trains joint query and document actions; reward configurations
define the retrieval signal and its candidate-pool behavior.

`NPROC_PER_NODE` controls the local process count and defaults to `8`. `NNODES` defaults to `1`. Both launchers forward other arguments to their training entrypoint.

For the RELER own-candidate objective (graded nDCG@10 plus pairwise feedback),
use `configs/examples/reler.yaml`. It enables CMP with 64 actions per side and
alignment 0.70; `--gradient_estimator score_function` switches both reward
components to unprojected RLOO. See [training configuration](docs/grpo.md#training-configuration)
for the command and overrides.

## Fixed-corpus GRPO

Fixed-corpus training encodes documents once, then trains only the query encoder. Build a lookup directory with the same model and document protocol used to initialize training:

```bash
reler-index \
  --input /path/to/train.ready.jsonl \
  --input-format training_candidates \
  --output-dir artifacts/reler-corpus \
  --model-config configs/model/qwen3_embedding_0.6b.yaml \
  --max-length 1024 \
  --num-shards 16
```

The builder reads the same prepared training records and uses `document_keys`
for global document identity, since `document_ids` may repeat across sources.
The resulting directory contains the embedding protocol, a document-to-ordinal
mapping, and vector shards. A separately supplied corpus can instead use
`--input-format document_jsonl` with `{id, content}` records; its IDs must match
the training records' document keys or unambiguous document IDs.

Train with the dedicated mode so index concerns remain outside ordinary GRPO:

```bash
NPROC_PER_NODE=1 bash scripts/run_fixed_grpo.sh \
  configs/examples/fixed_grpo.yaml \
  --data_path /path/to/train.ready.jsonl \
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

For BRIGHT, select the official GPT-4 reasoning queries with one argument:

```bash
reler-eval \
  --model checkpoints/reler \
  --tasks BrightRetrieval \
  --langs eng \
  --bright_query_set gpt4-reasoning \
  --model_kwargs '{"max_length": 8192}' \
  --batch_size 16 \
  --output_dir results/bright
```

`--bright_query_set original` is the default. The GPT-4 mode reads the complete
`query` field from the official `gpt4_reason` configuration at the same pinned
dataset revision as the corpus. It matches query IDs exactly, keeps the original
corpus and relevance labels, and never appends human reasoning annotations.
Missing, duplicate, or mismatched queries fail the run.

Original results retain the usual output path. GPT-4 results and predictions go
under `<output_dir>/query-gpt4-reasoning/`, so both modes can use the same output
root without reusing each other's results. The switch also works with fixed-corpus
evaluation; document embeddings remain reusable because the corpus is unchanged.

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
