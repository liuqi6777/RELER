"""Unified command-line entrypoint for RELER training."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from reler.training.common import shutdown_distributed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reler-train",
        description=(
            "Train a RELER embedding model with a supervised, joint GRPO, "
            "or fixed-corpus GRPO objective."
        ),
    )
    parser.add_argument("mode", choices=("supervised", "grpo", "fixed-grpo"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        parser.print_help()
        return
    args = parser.parse_args(raw_args[:1])
    training_args = raw_args[1:]

    try:
        if args.mode == "supervised":
            from reler.training.supervised import main as train
        elif args.mode == "fixed-grpo":
            from reler.fixed_corpus.training import main as train
        else:
            from reler.training.grpo import main as train
        train(training_args)
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
