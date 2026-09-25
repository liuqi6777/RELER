#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model to produce a standalone checkpoint.

This supports a supervised-to-GRPO handoff: supervised training writes a PEFT
adapter, and GRPO starts a *fresh* adapter on top of the merged weights.

Merging rather than chaining adapters (`lora_path`) matters for two reasons:

1. The vMF KL anchor is computed by disabling the adapter. If GRPO continued
   training the supervised adapter, disabling it would anchor the policy to the raw
   base model instead of to the supervised checkpoint.
2. It keeps the supervised and GRPO adapter hyperparameters (rank, alpha, target
   modules) independent.

Usage:
    python scripts/merge_lora.py \
        --base Qwen/Qwen3-0.6B \
        --adapter checkpoints/reler-supervised \
        --out checkpoints/reler-supervised-merged
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Base model id or path the adapter was trained on")
    parser.add_argument("--revision", help="Immutable Hugging Face revision for --base")
    parser.add_argument("--adapter", required=True, help="Directory holding adapter_config.json")
    parser.add_argument("--out", required=True, help="Destination for the merged checkpoint")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Dtype to load and save in. Keep this equal to the training dtype.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace --out if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = pathlib.Path(args.adapter)
    out_dir = pathlib.Path(args.out)

    if not (adapter_dir / "adapter_config.json").is_file():
        # A full fine-tune writes a plain model, not an adapter; nothing to merge.
        raise SystemExit(
            f"No adapter_config.json in {adapter_dir}. If supervised training was a full "
            f"fine-tune, point GRPO at that directory directly instead of merging."
        )
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(out_dir)

    dtype = getattr(torch, args.dtype)
    print(f"Loading base model {args.base} ({args.dtype})")
    revision_kwargs = {"revision": args.revision} if args.revision else {}
    model = AutoModel.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True, **revision_kwargs
    )

    print(f"Applying adapter {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir), torch_dtype=dtype)

    print("Merging")
    model = model.merge_and_unload()

    print(f"Saving to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    # The tokenizer travels with the checkpoint: GRPO loads both from model_name_or_path.
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").is_file() else args.base
    protocol_path = adapter_dir / "embedding_protocol.json"
    padding_side = "left"
    if protocol_path.is_file():
        with protocol_path.open(encoding="utf-8") as handle:
            padding_side = json.load(handle).get("padding_side", padding_side)
    print(f"Saving tokenizer from {tokenizer_source}")
    AutoTokenizer.from_pretrained(
        str(tokenizer_source), padding_side=padding_side, trust_remote_code=True,
        **(revision_kwargs if str(tokenizer_source) == args.base else {}),
    ).save_pretrained(str(out_dir))
    if protocol_path.is_file():
        shutil.copy2(protocol_path, out_dir / protocol_path.name)

    print(f"Done. Point GRPO at: --model_name_or_path {out_dir}")


if __name__ == "__main__":
    main()
