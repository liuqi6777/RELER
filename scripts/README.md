# RELER scripts

Run commands from the repository root. The launchers cover supervised, joint GRPO, and fixed-corpus GRPO training.

| Script | Purpose |
| --- | --- |
| `run_supervised.sh` | Launch supervised InfoNCE, RankNet, or LambdaLoss training. |
| `run_grpo.sh` | Launch GRPO training. |
| `run_fixed_grpo.sh` | Launch query-only GRPO against immutable document embeddings. |
| `run_mteb.sh` | Evaluate a checkpoint with the packaged MTEB runner. |
| `merge_lora.py` | Merge a LoRA adapter into its base model for standalone use. |
| `zero3.json` | Optional DeepSpeed ZeRO-3 configuration for large-model runs. |

## Launchers

All training launchers use `torchrun` and accept `NNODES` and `NPROC_PER_NODE`. `NPROC_PER_NODE` defaults to `8`; set it explicitly for the visible GPUs.

```bash
NPROC_PER_NODE=1 bash scripts/run_supervised.sh configs/examples/supervised.yaml \
  --base-baseline configs/baseline/ranknet.yaml \
  --data_path /path/to/train.ready.jsonl \
  --output_dir checkpoints/reler-ranknet
```

Use `configs/baseline/infonce_in_batch.yaml` for InfoNCE or
`configs/baseline/lambdaloss.yaml` for LambdaLoss. The supervised launcher
accepts the configuration slots `train`, `dataset`, `model`, and `baseline`.

```bash
NPROC_PER_NODE=1 bash scripts/run_grpo.sh configs/examples/grpo.yaml \
  --data_path /path/to/train.ready.jsonl \
  --output_dir checkpoints/reler-grpo
```

The GRPO example composes all required slots. Alternatively,
`run_grpo.sh` accepts `--base-train`, `--base-dataset`, `--base-model`,
`--base-grpo`, and `--base-reward` for an explicit composition.

Build an immutable lookup directory with `reler-index`, then launch its dedicated
training mode:

```bash
reler-index \
  --input /path/to/train.ready.jsonl \
  --output-dir artifacts/reler-corpus \
  --model-config configs/model/qwen3_embedding_0.6b.yaml \
  --max-length 1024

NPROC_PER_NODE=1 bash scripts/run_fixed_grpo.sh \
  configs/examples/fixed_grpo.yaml \
  --data_path /path/to/train.ready.jsonl \
  --index_dir artifacts/reler-corpus
```

`fixed-grpo` adds the `corpus` configuration slot and requires the query-only
policy in `configs/grpo/query_only.yaml`.

## Data and model conventions

The data loader accepts only prepared `embedding_candidates_v2` JSONL records. Use `configs/dataset/prepared.yaml`; see the root README for required fields and the independent binary, graded, and ranking labels.

Do not change pooling, padding, terminal-token handling, or query/document templates when evaluating a trained checkpoint. Training saves these values in `embedding_protocol.json`, and the MTEB runner reads that file automatically. Scoring remains FP32 by design.

## MTEB

```bash
bash scripts/run_mteb.sh \
  checkpoints/reler-ranknet \
  'MTEB(eng, v2)' \
  configs/model/qwen3_embedding_0.6b.yaml
```

The runner writes results to `results/mteb/`. To summarize an existing result tree:

```bash
reler-summarize results/mteb 'MTEB(eng, v2)'
```
