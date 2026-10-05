# RELER

Official code for **Learning to Retrieve via Reinforcement Learning in Embedding Space**.

Qi Liu, Fengming Liang, Yiqun Chen, Erhan Zhang, and Jiaxin Mao · Renmin University of China

[Training data](https://huggingface.co/datasets/liuqi6777/reler-reasonrank-train) · [Training and evaluation guide](docs/usage.md) · Paper: arXiv link coming soon

## Overview

**RELER** (**RE**inforcement **LE**arning for **R**etrieval) adapts pretrained embedding models directly to retrieval rewards. During training, it explores query and document embeddings with von Mises–Fisher policies and learns from scalar feedback using REINFORCE with a leave-one-out baseline (RLOO). At inference, it uses ordinary deterministic vector retrieval.

- **Learn from retrieval rewards.** Optimize graded nDCG@10 and pairwise feedback without differentiating through ranking.
- **Reduce exploration noise.** Conditional-mean projection (CMP) removes reward-invisible gradient noise while preserving the expected fixed-candidate gradient.
- **Reuse embedding samples.** Product rollouts combine query and document actions to obtain more reward feedback without additional encoder forward passes.

This release includes embedding training, supervised baselines, fixed-corpus query training, and retrieval evaluation. RAG code and data will be released later.

## Results

On BRIGHT, RELER achieves the highest average nDCG@10 among the pretrained encoder, InfoNCE, and LambdaLoss baselines across all three backbones, using both original and GPT-4 reasoning queries.

BRIGHT with original queries; nDCG@10 × 100, averaged over 12 subsets. Post-training results are mean ± SD over three seeds.

| Backbone | Pretrained | InfoNCE | LambdaLoss | RELER |
| --- | ---: | ---: | ---: | ---: |
| BGE-M3 | 10.66 | 12.63 ± 0.11 | 11.40 ± 0.08 | **14.49 ± 0.04** |
| Qwen3-Embedding-0.6B | 15.10 | 22.64 ± 0.20 | 19.36 ± 0.28 | **23.38 ± 0.17** |
| Qwen3-Embedding-4B | 18.70 | 30.08 ± 0.16 | 25.01 ± 0.37 | **30.46 ± 0.55** |

## Quick start

Requires Python 3.10+ and a CUDA-capable PyTorch environment.

```bash
git clone https://github.com/liuqi6777/RELER.git
cd RELER
uv sync --extra dev
source .venv/bin/activate
```

Download `train.ready.jsonl` from the [training dataset](https://huggingface.co/datasets/liuqi6777/reler-reasonrank-train) and place it in `data/`. It contains the prepared ReasonRank candidates and teacher labels used for retrieval post-training, in the `embedding_candidates_v2` format.

Train Qwen3-Embedding-0.6B with the RELER example configuration:

```bash
NPROC_PER_NODE=8 bash scripts/run_grpo.sh configs/examples/reler.yaml \
  --data_path data/train.ready.jsonl \
  --output_dir checkpoints/reler
```

The paper's retrieval experiments use eight NVIDIA H100 80GB GPUs. Adjust the configuration and process count for your hardware.

For supervised baselines, other backbones, fixed-corpus training, and BRIGHT/MTEB evaluation, see the [usage guide](docs/usage.md). Method and configuration details are in [docs/grpo.md](docs/grpo.md).

## Citation

If you use RELER, please cite our paper. The entry below is a placeholder; the arXiv identifier and URL will be added after publication.

```bibtex
@misc{liu2026reler,
  title  = {Learning to Retrieve via Reinforcement Learning in Embedding Space},
  author = {Qi Liu and Fengming Liang and Yiqun Chen and Erhan Zhang and Jiaxin Mao},
  year   = {2026},
  note   = {Preprint; arXiv identifier to be added}
}
```

## License

[Apache License 2.0](LICENSE).
