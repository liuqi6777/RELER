"""Thin query-only GRPO adapter for immutable document embeddings."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F
from transformers import HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint

from reler.config.arguments import (
    DataArguments,
    FrozenCorpusArguments,
    LoraArguments,
    ModelArguments,
    RLArguments,
    TrainingArguments,
)
from reler.config.loader import FIXED_GRPO_CONFIG_SLOTS
from reler.fixed_corpus.data import FixedCorpusDataCollator
from reler.fixed_corpus.index import FrozenCorpusIndex
from reler.objectives.rewards import warn_on_inert_cutoffs
from reler.training.common import (
    apply_gradient_checkpointing,
    build_embedding_dataset,
    guard_output_dir,
    load_backbone_and_tokenizer,
    parse_arguments,
    save_run_artifacts,
    setup_logging,
    shutdown_distributed,
)
from reler.training.grpo_model import GRPOModel, GRPOModelOutput
from reler.training.trainer import GRPOTrainer, restore_grpo_state

logger = logging.getLogger(__name__)


def validate_fixed_corpus_grpo_arguments(rl_args: RLArguments) -> None:
    """Reject configurations that require trainable or cross-rank documents."""
    if tuple(rl_args.action_components) != (("query",),):
        raise ValueError(
            "Fixed-corpus GRPO requires action_components='query'; document actions "
            "would require re-encoding immutable documents."
        )
    if rl_args.gradient_estimator != "score_function":
        raise ValueError(
            "Fixed-corpus GRPO only supports gradient_estimator='score_function'; "
            "conditional projection requires a sampled document policy."
        )
    if rl_args.reward_cross_device_negatives:
        raise ValueError(
            "Fixed-corpus GRPO does not support reward_cross_device_negatives; "
            "candidate ordinals are local to the per-rank batch."
        )
    if rl_args.cross_query_document_gradients:
        raise ValueError(
            "Fixed-corpus GRPO does not support cross_query_document_gradients; "
            "frozen document vectors never participate in autograd."
        )


class FixedCorpusGRPOModel(GRPOModel):
    """Existing GRPO math with a trainable query encoder and fixed document lookup."""

    def __init__(
        self, model, index, rl_args: RLArguments, pooling_method: str = "last"
    ):
        validate_fixed_corpus_grpo_arguments(rl_args)
        super().__init__(model=model, rl_args=rl_args, pooling_method=pooling_method)
        self.frozen_corpus_index = index
        self.frozen_corpus_dimension = int(index.dimension)

    def _lookup_documents(
        self, candidate_ordinals: torch.Tensor, candidate_mask: torch.Tensor, device
    ) -> torch.Tensor:
        if candidate_ordinals.shape != candidate_mask.shape:
            raise ValueError(
                "candidate_ordinals and candidate_mask must have the same shape"
            )
        if candidate_ordinals.dim() != 2:
            raise ValueError("candidate_ordinals must have shape [batch, candidates]")
        if not candidate_mask.any(dim=-1).all():
            raise ValueError("Every query must retain at least one candidate")
        if (candidate_ordinals[candidate_mask] < 0).any():
            raise ValueError(
                "Valid fixed-corpus candidates require non-negative ordinals"
            )
        safe_ordinals = candidate_ordinals.masked_fill(~candidate_mask, 0)
        documents = self.frozen_corpus_index.lookup_embeddings(
            safe_ordinals, device=device
        ).detach()
        if documents.shape != (*candidate_ordinals.shape, self.frozen_corpus_dimension):
            raise ValueError(
                "Frozen corpus lookup returned an unexpected embedding shape: "
                f"{tuple(documents.shape)}"
            )
        if documents.size(-1) == 0:
            raise ValueError("Frozen corpus embeddings must have a positive dimension")
        return F.normalize(documents.float(), dim=-1).masked_fill(
            ~candidate_mask.unsqueeze(-1), 0
        )

    def forward(
        self,
        query: dict[str, torch.Tensor] | None = None,
        candidate_ordinals: torch.Tensor | None = None,
        relevance_labels: torch.Tensor | None = None,
        rank_labels: torch.Tensor | None = None,
        positive_mask: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        in_batch_positive_mask: torch.Tensor | None = None,
        in_batch_candidate_mask: torch.Tensor | None = None,
        cross_batch_metadata: list[dict] | None = None,
        **_: Any,
    ) -> GRPOModelOutput:
        if query is None:
            raise ValueError("query inputs are required for fixed-corpus GRPO training")
        if candidate_ordinals is None:
            raise ValueError(
                "candidate_ordinals are required for fixed-corpus GRPO training"
            )
        if relevance_labels is None:
            raise ValueError(
                "relevance_labels are required for fixed-corpus GRPO training"
            )
        if relevance_labels.shape != candidate_ordinals.shape:
            raise ValueError(
                "relevance_labels and candidate_ordinals must have the same shape"
            )
        mask = (
            candidate_ordinals >= 0
            if candidate_mask is None
            else candidate_mask.to(dtype=torch.bool)
        )

        policy_query_embeddings = self.encode(query)
        rollout_query_embeddings = policy_query_embeddings.detach()
        documents = self._lookup_documents(
            candidate_ordinals, mask, policy_query_embeddings.device
        )
        if documents.size(-1) != policy_query_embeddings.size(-1):
            raise ValueError(
                "Frozen corpus embedding dimension does not match query encoder: "
                f"{documents.size(-1)} != {policy_query_embeddings.size(-1)}"
            )

        reference_query_embeddings = None
        if self.grpo.kl_coef > 0:
            if not hasattr(self.model, "disable_adapter"):
                raise RuntimeError(
                    "kl_coef > 0 requires a PEFT/LoRA model exposing .disable_adapter(); "
                    "either enable LoRA or set kl_coef=0."
                )
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad(), self.model.disable_adapter():
                    reference_query_embeddings = self.encode(query)
            finally:
                self.model.train(was_training)

        loss, reward_stats, advantage_stats, sigma, kl = self.grpo(
            rollout_query_embeddings=rollout_query_embeddings,
            rollout_positive_document_embeddings=documents[:, :1],
            rollout_negative_document_embeddings=documents[:, 1:],
            relevance_labels=relevance_labels,
            rank_labels=rank_labels,
            candidate_mask=mask,
            in_batch_positive_mask=in_batch_positive_mask,
            in_batch_candidate_mask=in_batch_candidate_mask,
            policy_query_embeddings=policy_query_embeddings,
            reference_query_embeddings=reference_query_embeddings,
            positive_mask=positive_mask,
            cross_batch_metadata=cross_batch_metadata,
        )
        term_metrics = {key: value for key, value in reward_stats.items() if "/" in key}
        aggregate_stats = {
            key: value for key, value in reward_stats.items() if "/" not in key
        }
        return GRPOModelOutput(
            loss=loss,
            reward=reward_stats["reward_mean"],
            **aggregate_stats,
            **advantage_stats,
            sigma=sigma,
            kl=kl,
            reward_terms=term_metrics or None,
        )


def _require_index_protocol(
    index: FrozenCorpusIndex,
    model_args: ModelArguments,
    data_args: DataArguments,
    model,
) -> None:
    """Validate that frozen documents share the configured embedding protocol."""
    resolved_revision = (
        getattr(model.config, "_commit_hash", None) or model_args.model_revision
    )
    index.validate_protocol(
        model_name_or_path=model_args.model_name_or_path,
        resolved_model_revision=resolved_revision,
        pooling_method=model_args.pooling_method,
        padding_side=model_args.padding_side,
        append_token=model_args.append_token,
        document_prompt_template=model_args.document_prompt_template,
        document_max_length=min(data_args.d_max_len, model_args.embedding_max_length),
    )


def main(argv: list[str] | None = None) -> None:
    """Train only the query encoder while looking up immutable document vectors."""
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
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        rl_args,
        corpus_args,
    ) = parse_arguments(
        parser=parser,
        base_slots=FIXED_GRPO_CONFIG_SLOTS,
        argv=argv,
    )
    validate_fixed_corpus_grpo_arguments(rl_args)
    guard_output_dir(training_args)
    setup_logging(
        training_args,
        {"Model": model_args, "RL": rl_args, "Frozen corpus": corpus_args},
    )
    set_seed(training_args.seed)

    backbone, tokenizer = load_backbone_and_tokenizer(model_args, lora_args)
    with FrozenCorpusIndex(
        corpus_args.index_dir,
        device=training_args.device,
    ) as index:
        _require_index_protocol(index, model_args, data_args, backbone)
        model = FixedCorpusGRPOModel(
            model=backbone,
            index=index,
            rl_args=rl_args,
            pooling_method=model_args.pooling_method,
        )
        model.train()

        resume_checkpoint = None
        if not training_args.overwrite_output_dir and os.path.isdir(
            training_args.output_dir
        ):
            resume_checkpoint = get_last_checkpoint(training_args.output_dir)
        restore_grpo_state(model, resume_checkpoint)
        apply_gradient_checkpointing(model, training_args, lora_args)

        train_dataset = build_embedding_dataset(
            data_args,
            training_args,
            model_args,
            preserve_document_metadata=True,
        )
        data_collator = FixedCorpusDataCollator(
            tokenizer=tokenizer,
            index=index,
            query_max_length=min(data_args.q_max_len, model_args.embedding_max_length),
            relevance_scheme=data_args.relevance_scheme,
            append_token=model_args.append_token,
            include_cross_batch_metadata=(
                rl_args.reward_shortlist_count > 0
                or (
                    rl_args.aux_infonce_coef > 0
                    and rl_args.aux_infonce_strong_negatives
                )
            ),
        )
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
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        save_run_artifacts(
            trainer,
            training_args,
            tokenizer,
            model_args=model_args,
            rl_args=rl_args,
            corpus_args=corpus_args,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_distributed()
