from pathlib import Path

import pytest
from transformers import HfArgumentParser

from reler.config import (
    DataArguments,
    FrozenCorpusArguments,
    LoraArguments,
    ModelArguments,
    RLArguments,
    TrainingArguments,
)
from reler.config.loader import (
    BASE_CONFIG_SLOTS,
    FIXED_GRPO_CONFIG_SLOTS,
    parse_config_file,
)


def test_fixed_corpus_configuration_is_isolated_from_joint_training():
    assert "corpus" not in BASE_CONFIG_SLOTS
    assert FIXED_GRPO_CONFIG_SLOTS == (*BASE_CONFIG_SLOTS, "corpus")
    with pytest.raises(ValueError, match="non-empty path"):
        FrozenCorpusArguments(index_dir="  ")


def test_fixed_grpo_example_resolves_to_query_only_policy():
    root = Path(__file__).resolve().parents[1]
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            LoraArguments,
            RLArguments,
            FrozenCorpusArguments,
        )
    )
    *_, rl_args, corpus_args = parse_config_file(
        parser,
        str(root / "configs/examples/fixed_grpo.yaml"),
        cli_args=["--bf16", "false", "--tf32", "false", "--use_cpu", "true"],
    )

    assert rl_args.action_components == (("query",),)
    assert corpus_args.index_dir.endswith("frozen-corpus")
