"""GRPO training entrypoint for RELER embedding training."""

import logging
import os

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from reler.config import (
    DataArguments,
    LoraArguments,
    ModelArguments,
    RLArguments,
    TrainingArguments,
)
from reler.config.loader import BASE_CONFIG_SLOTS
from reler.objectives.rewards import warn_on_inert_cutoffs
from reler.training import common as training_common
from reler.training.common import (
    apply_gradient_checkpointing,
    build_joint_embedding_data,
    guard_output_dir,
    parse_arguments,
    save_run_artifacts,
    setup_logging,
    shutdown_distributed,
)
from reler.training.grpo_model import GRPOModel
from reler.training.trainer import GRPOTrainer, restore_grpo_state

logger = logging.getLogger(__name__)


def load_backbone_and_tokenizer(model_args: ModelArguments, lora_args: LoraArguments):
    """Load the encoder through patchable classes for lightweight smoke tests."""
    return training_common.load_backbone_and_tokenizer(
        model_args,
        lora_args,
        auto_config_cls=AutoConfig,
        auto_model_cls=AutoModel,
        auto_tokenizer_cls=AutoTokenizer,
    )


def main(argv: list[str] | None = None) -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, RLArguments)
    )
    model_args, data_args, training_args, lora_args, rl_args = parse_arguments(
        parser=parser,
        base_slots=BASE_CONFIG_SLOTS,
        argv=argv,
    )

    guard_output_dir(training_args)
    setup_logging(
        training_args,
        {"Model": model_args, "RL": rl_args},
    )

    set_seed(training_args.seed)

    backbone, tokenizer = load_backbone_and_tokenizer(model_args, lora_args)
    model = GRPOModel(
        model=backbone,
        rl_args=rl_args,
        pooling_method=model_args.pooling_method,
    )
    model.train()

    # Resolved before the trainer is built so the exploration scale can be restored while
    # the parameter is still whole (DeepSpeed partitions it during trainer construction).
    resume_checkpoint = None
    if not training_args.overwrite_output_dir and os.path.isdir(
        training_args.output_dir
    ):
        resume_checkpoint = get_last_checkpoint(training_args.output_dir)
    restore_grpo_state(model, resume_checkpoint)

    apply_gradient_checkpointing(model, training_args, lora_args)

    train_dataset, data_collator = build_joint_embedding_data(
        data_args,
        training_args,
        tokenizer,
        model_args,
    )
    data_collator.include_cross_batch_metadata = (
        rl_args.reward_cross_device_negatives
        or rl_args.reward_shortlist_count > 0
        or rl_args.cross_query_document_gradients
        or (rl_args.aux_infonce_coef > 0 and rl_args.aux_infonce_strong_negatives)
    )

    # Both halves of this check live in different config slots -- the cutoff in reward/, the
    # slate in dataset/ -- so nothing else notices when a change to one invalidates the other.
    for warning in warn_on_inert_cutoffs(
        model.grpo.reward_terms,
        slate_size=data_args.slate_size,
        batch_size=training_args.per_device_train_batch_size,
    ):
        logger.warning(warning)

    trainer = GRPOTrainer(
        model_args=model_args,
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=True if resume_checkpoint else None)

    save_run_artifacts(
        trainer, training_args, tokenizer, model_args=model_args, rl_args=rl_args
    )


if __name__ == "__main__":
    main()
    shutdown_distributed()
