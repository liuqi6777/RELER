"""Objective-independent helpers shared by RELER training entrypoints.

This module contains the common joint-encoder training setup used by every public
training entrypoint.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments as HFTrainingArguments

from reler.config import DataArguments, LoraArguments, ModelArguments
from reler.config.loader import (
    BASE_CONFIG_SLOTS,
    parse_config_file,
    parse_config_from_base_overrides,
    resolve_run_name_and_output_dir,
)
from reler.data.embedding import EmbeddingDataCollator, EmbeddingDataset
from reler.data.protocol import save_embedding_protocol

logger = logging.getLogger(__name__)

CONFIG_FILE_SUFFIXES = {".json", ".yaml", ".yml"}


def get_deepspeed_zero_stage(deepspeed_config) -> int | None:
    if deepspeed_config is None:
        return None
    if isinstance(deepspeed_config, dict):
        config = deepspeed_config
    elif isinstance(deepspeed_config, str):
        if os.path.isfile(deepspeed_config):
            with open(deepspeed_config, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        else:
            try:
                config = json.loads(deepspeed_config)
            except json.JSONDecodeError:
                return None
    else:
        return None
    zero_optimization = config.get("zero_optimization")
    if not isinstance(zero_optimization, dict):
        return None
    stage = zero_optimization.get("stage")
    return int(stage) if stage is not None else None


def resolve_gradient_checkpointing_kwargs(
    training_args: HFTrainingArguments,
    lora_args: LoraArguments,
) -> dict:
    kwargs = dict(training_args.gradient_checkpointing_kwargs or {})
    if (
        lora_args.lora_enabled
        and get_deepspeed_zero_stage(training_args.deepspeed) == 3
    ):
        if kwargs.get("use_reentrant") is not True:
            logger.warning(
                "Detected DeepSpeed ZeRO-3 with LoRA; using reentrant gradient "
                "checkpointing to avoid empty parameter shards."
            )
        kwargs["use_reentrant"] = True
    else:
        kwargs.setdefault("use_reentrant", False)
    return kwargs


def save_model_for_trainer(trainer: HFTrainer, output_dir: str) -> None:
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def split_launcher_args(
    cli_args: list[str],
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> tuple[dict[str, str], list[str]]:
    """Peel the ``--base-<slot> <path>`` launcher flags off the CLI argv."""
    base_flag_to_slot = {f"--base-{slot}": slot for slot in base_slots}

    base_overrides: dict[str, str] = {}
    passthrough_args: list[str] = []
    index = 0
    while index < len(cli_args):
        arg = cli_args[index]
        if arg in base_flag_to_slot:
            if index + 1 >= len(cli_args):
                raise ValueError(f"Expected a YAML path after {arg}")
            base_overrides[base_flag_to_slot[arg]] = cli_args[index + 1]
            index += 2
            continue

        passthrough_args.append(arg)
        index += 1

    return base_overrides, passthrough_args


def parse_arguments(
    parser: HfArgumentParser,
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
    argv: list[str] | None = None,
) -> tuple:
    """Resolve dataclasses from a top-level YAML, base flags, or plain CLI."""
    base_overrides, cli_args = split_launcher_args(
        sys.argv[1:] if argv is None else argv,
        base_slots=base_slots,
    )

    config_path = None
    if cli_args and pathlib.Path(cli_args[0]).suffix.lower() in CONFIG_FILE_SUFFIXES:
        config_path = cli_args[0]
        parsed = parse_config_file(
            parser=parser,
            config_path=config_path,
            cli_args=cli_args[1:],
            base_overrides=base_overrides,
        )
    elif base_overrides:
        parsed = parse_config_from_base_overrides(
            parser=parser,
            base_overrides=base_overrides,
            cli_args=cli_args,
            base_slots=base_slots,
        )
    else:
        parsed = parser.parse_args_into_dataclasses(cli_args)

    training_args = next(
        args for args in parsed if isinstance(args, HFTrainingArguments)
    )
    resolve_run_name_and_output_dir(
        config_path=config_path,
        base_overrides=base_overrides,
        training_args=training_args,
        base_slots=base_slots,
    )
    return parsed


def guard_output_dir(training_args: HFTrainingArguments) -> None:
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )


def setup_logging(
    training_args: HFTrainingArguments, extra_parameters: dict | None = None
) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    for label, parameters in (extra_parameters or {}).items():
        logger.info("%s parameters %s", label, parameters)


def load_backbone_and_tokenizer(
    model_args: ModelArguments,
    lora_args: LoraArguments,
    *,
    auto_config_cls=AutoConfig,
    auto_model_cls=AutoModel,
    auto_tokenizer_cls=AutoTokenizer,
):
    """Load the shared encoder, tokenizer, and optional trainable LoRA adapter."""
    revision_kwargs = (
        {"revision": model_args.model_revision} if model_args.model_revision else {}
    )
    config = auto_config_cls.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
        **revision_kwargs,
    )
    backbone = auto_model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        **revision_kwargs,
    )
    tokenizer = auto_tokenizer_cls.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        padding_side=model_args.padding_side,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        **revision_kwargs,
    )

    if lora_args.lora_enabled:
        if lora_args.lora_path:
            print(f"Loading LoRA from {lora_args.lora_path}")
            backbone = PeftModel.from_pretrained(
                backbone, lora_args.lora_path, is_trainable=True
            )
        else:
            print("Initializing LoRA")
            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="FEATURE_EXTRACTION",
            )
            backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()

    return backbone, tokenizer


def apply_gradient_checkpointing(
    model, training_args: HFTrainingArguments, lora_args: LoraArguments
) -> None:
    if not training_args.gradient_checkpointing:
        return
    training_args.gradient_checkpointing_kwargs = resolve_gradient_checkpointing_kwargs(
        training_args=training_args,
        lora_args=lora_args,
    )
    if training_args.gradient_checkpointing_kwargs.get("use_reentrant", True):
        model.enable_input_require_grads()
    logger.info(
        "Gradient checkpointing kwargs %s", training_args.gradient_checkpointing_kwargs
    )


def build_embedding_dataset(
    data_args: DataArguments,
    training_args: HFTrainingArguments,
    model_args: ModelArguments | None = None,
    *,
    preserve_document_metadata: bool = False,
):
    """Build the training dataset with the stable data seed."""
    data_seed = getattr(training_args, "data_seed", None)
    if data_seed is not None:
        set_seed(data_seed)
    return EmbeddingDataset(
        data_args=data_args,
        batch_size=training_args.per_device_train_batch_size,
        query_prompt_template=(
            model_args.query_prompt_template if model_args else None
        ),
        preserve_document_metadata=preserve_document_metadata,
    )


def build_joint_embedding_data(
    data_args: DataArguments,
    training_args: HFTrainingArguments,
    tokenizer,
    model_args: ModelArguments | None = None,
):
    """Build the datasets and collator for the standard joint encoder."""
    train_dataset = build_embedding_dataset(data_args, training_args, model_args)
    data_collator = EmbeddingDataCollator(
        tokenizer=tokenizer,
        query_max_length=min(
            data_args.q_max_len,
            model_args.embedding_max_length if model_args else data_args.q_max_len,
        ),
        doc_max_length=min(
            data_args.d_max_len,
            model_args.embedding_max_length if model_args else data_args.d_max_len,
        ),
        relevance_scheme=data_args.relevance_scheme,
        document_prompt_template=(
            model_args.document_prompt_template if model_args else "{document}"
        ),
        append_token=model_args.append_token if model_args else "pad",
    )
    return train_dataset, data_collator


def save_run_artifacts(
    trainer, training_args: HFTrainingArguments, tokenizer, **argument_objects
) -> None:
    """Persist the model plus the pickled argument dataclasses under ``output_dir``."""
    save_model_for_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        if "model_args" in argument_objects:
            save_embedding_protocol(
                argument_objects["model_args"], training_args.output_dir, tokenizer
            )
        torch.save(
            training_args, os.path.join(training_args.output_dir, "training_args.bin")
        )
        for name, argument_object in argument_objects.items():
            torch.save(
                argument_object, os.path.join(training_args.output_dir, f"{name}.bin")
            )
    print("Training done.")


def shutdown_distributed() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    print("Success.")
