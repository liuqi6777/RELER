from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import HfArgumentParser, PreTrainedModel, set_seed
from transformers import Trainer as HFTrainer
from transformers.file_utils import ModelOutput

from reler.config import (
    BaselineArguments,
    DataArguments,
    LoraArguments,
    ModelArguments,
    TrainingArguments,
)
from reler.config.loader import BASELINE_CONFIG_SLOTS
from reler.data.embedding import build_slate_inputs
from reler.data.protocol import encode_valid_candidates, pool_embeddings
from reler.objectives.contrastive import (
    compute_in_batch_positive_scores,
    compute_infonce_loss,
    cross_query_scores,
)
from reler.objectives.precision import fp32_scores
from reler.training.common import (
    apply_gradient_checkpointing,
    build_joint_embedding_data,
    guard_output_dir,
    load_backbone_and_tokenizer,
    parse_arguments,
    save_run_artifacts,
    setup_logging,
    shutdown_distributed,
)
from reler.training.trainer import EmbeddingTrainerMixin

logger = logging.getLogger(__name__)


@dataclass
class BaselineModelOutput(ModelOutput):
    loss: Optional[Tensor] = None


class BaselineTrainer(EmbeddingTrainerMixin, HFTrainer):
    train_metric_names = ()


def compute_ranknet_loss(
    scores: Tensor,
    rank_labels: Tensor,
    temperature: float = 0.03,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    """Return the mean RankNet loss over all strictly ordered pairs per sample.

    ``rank_labels`` normally preserve the record's full teacher permutation. Ties
    are excluded as a general fallback when graded relevance labels are supplied.
    """
    if temperature <= 0:
        raise ValueError(f"ranknet temperature must be positive, got {temperature}")
    if scores.dtype in {torch.float16, torch.bfloat16}:
        scores = scores.float()
    if scores.shape != rank_labels.shape:
        raise ValueError(
            "scores and rank_labels must have the same shape, "
            f"got {tuple(scores.shape)} and {tuple(rank_labels.shape)}"
        )

    if candidate_mask is not None:
        scores = scores.masked_fill(~candidate_mask, 0)
    scaled_scores = scores / float(temperature)
    score_differences = scaled_scores.unsqueeze(2) - scaled_scores.unsqueeze(1)
    label_differences = rank_labels.unsqueeze(2) - rank_labels.unsqueeze(1)
    ordered_pairs = label_differences > 0
    if candidate_mask is not None:
        ordered_pairs = (
            ordered_pairs & candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1)
        )
    pair_losses = F.softplus(-score_differences)
    pair_counts = ordered_pairs.sum(dim=(1, 2))
    loss_sums = (pair_losses * ordered_pairs.to(pair_losses.dtype)).sum(dim=(1, 2))
    return torch.where(
        pair_counts > 0,
        loss_sums / pair_counts.clamp_min(1).to(loss_sums.dtype),
        torch.zeros_like(loss_sums),
    )


def compute_lambdaloss_loss(
    scores: Tensor,
    relevance_labels: Tensor,
    k: int = 10,
    sigma: float = 1.0,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    """LambdaRank's LambdaLoss: pairwise logistic loss weighted by |delta nDCG@k|.

    Positions are recomputed from the current scores on every forward pass. The
    position-dependent weights are treated as constants by ``argsort``, matching
    the classic LambdaRank update. Relevance ties, padding, and pairs whose swap
    cannot affect nDCG@k contribute no loss.
    """
    if scores.dim() != 2 or relevance_labels.dim() != 2:
        raise ValueError(
            "scores and relevance_labels must both be 2D, "
            f"got {tuple(scores.shape)} and {tuple(relevance_labels.shape)}"
        )
    if scores.shape != relevance_labels.shape:
        raise ValueError(
            "scores and relevance_labels must have the same shape, "
            f"got {tuple(scores.shape)} and {tuple(relevance_labels.shape)}"
        )
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"lambdaloss k must be a positive integer, got {k}")
    if sigma <= 0:
        raise ValueError(f"lambdaloss sigma must be positive, got {sigma}")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(relevance_labels, dtype=torch.bool)
    elif candidate_mask.shape != scores.shape or candidate_mask.dtype != torch.bool:
        raise ValueError(
            "candidate_mask must be a boolean tensor matching scores, "
            f"got shape={tuple(candidate_mask.shape)} dtype={candidate_mask.dtype}"
        )

    if scores.dtype in {torch.float16, torch.bfloat16}:
        scores = scores.float()
    # Remove padding before differences: inf * zero would otherwise become NaN.
    scores = scores.masked_fill(~candidate_mask, 0)
    labels = relevance_labels.masked_fill(~candidate_mask, 0)
    ranking = scores.masked_fill(~candidate_mask, float("-inf")).argsort(
        dim=-1,
        descending=True,
        stable=True,
    )
    positions = torch.empty_like(ranking)
    positions.scatter_(
        dim=1,
        index=ranking,
        src=torch.arange(scores.size(1), device=scores.device).expand_as(ranking),
    )

    cutoff = min(k, scores.size(1))
    metric_dtype = (
        torch.float32
        if scores.dtype in {torch.float16, torch.bfloat16}
        else scores.dtype
    )
    position_discounts = torch.zeros(
        scores.size(1),
        device=scores.device,
        dtype=metric_dtype,
    )
    position_discounts[:cutoff] = 1.0 / torch.log2(
        torch.arange(2, cutoff + 2, device=scores.device, dtype=metric_dtype)
    )
    item_discounts = position_discounts[positions].masked_fill(~candidate_mask, 0)
    metric_labels = labels.to(metric_dtype)
    gains = torch.pow(2.0, metric_labels) - 1.0

    ideal_labels = (
        metric_labels.masked_fill(~candidate_mask, float("-inf"))
        .topk(
            k=cutoff,
            dim=-1,
        )
        .values
    )
    ideal_labels = ideal_labels.masked_fill(~torch.isfinite(ideal_labels), 0)
    idcg = (
        (torch.pow(2.0, ideal_labels) - 1.0) * position_discounts[:cutoff].unsqueeze(0)
    ).sum(dim=-1)

    label_differences = labels.unsqueeze(2) - labels.unsqueeze(1)
    preferred_pairs = label_differences > 0
    preferred_pairs = (
        preferred_pairs & candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1)
    )
    delta_ndcg = torch.abs(
        (gains.unsqueeze(2) - gains.unsqueeze(1))
        * (item_discounts.unsqueeze(2) - item_discounts.unsqueeze(1))
    )
    delta_ndcg = torch.where(
        idcg[:, None, None] > 0,
        delta_ndcg / idcg.clamp_min(torch.finfo(metric_dtype).eps)[:, None, None],
        torch.zeros_like(delta_ndcg),
    )
    weights = delta_ndcg * preferred_pairs.to(delta_ndcg.dtype)

    score_differences = scores.unsqueeze(2) - scores.unsqueeze(1)
    pair_losses = F.softplus(-float(sigma) * score_differences)
    weighted_loss = (pair_losses * weights).sum(dim=(1, 2))
    weight_sums = weights.sum(dim=(1, 2))
    return torch.where(
        weight_sums > 0,
        weighted_loss / weight_sums.clamp_min(torch.finfo(metric_dtype).eps),
        torch.zeros_like(weighted_loss),
    )


class BaselineModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        baseline_args: BaselineArguments,
        pooling_method: str = "last",
    ):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.baseline_args = baseline_args
        self.pooling_method = pooling_method

    def encode(self, model_inputs: Dict[str, Tensor]) -> Tensor:
        return pool_embeddings(
            self.model(**model_inputs).last_hidden_state,
            model_inputs["attention_mask"],
            pooling_method=self.pooling_method,
            normalize=True,
        )

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        positive_document: Dict[str, Tensor] = None,
        negative_document: Dict[str, Tensor] = None,
        relevance_labels: Tensor = None,
        rank_labels: Tensor = None,
        positive_mask: Tensor = None,
        candidate_mask: Tensor = None,
        in_batch_positive_mask: Tensor = None,
        in_batch_candidate_mask: Tensor = None,
        cross_batch_metadata: list[dict] = None,
    ) -> BaselineModelOutput:
        batch_size, slate_length = relevance_labels.shape
        if candidate_mask is None:
            candidate_mask = torch.ones_like(relevance_labels, dtype=torch.bool)
        query_embeddings = self.encode(query)
        document_inputs = build_slate_inputs(
            positive_document=positive_document,
            negative_document=negative_document,
            batch_size=batch_size,
            slate_length=slate_length,
        )
        document_embeddings = encode_valid_candidates(
            self.encode, document_inputs, candidate_mask
        ).reshape(batch_size, slate_length, -1)
        with fp32_scores(query_embeddings.device):
            scores = torch.matmul(
                document_embeddings.float(), query_embeddings.float().unsqueeze(-1)
            ).squeeze(-1)

        loss_weight = 1.0
        if self.baseline_args.extended_negative_pool:
            in_batch_scores, extra_mask, loss_weight = cross_query_scores(
                query_embeddings,
                document_embeddings,
                cross_batch_metadata,
                include_negatives=self.baseline_args.baseline_in_batch_include_negatives,
                cross_device=self.baseline_args.baseline_cross_device_negatives,
                detach_documents=self.baseline_args.baseline_detach_in_batch_documents,
            )
        elif self.baseline_args.baseline_use_in_batch_negatives:
            in_batch_scores = compute_in_batch_positive_scores(
                query_embeddings=query_embeddings,
                positive_embeddings=document_embeddings[:, 0],
            )
            if in_batch_positive_mask is None:
                extra_mask = torch.ones_like(in_batch_scores, dtype=torch.bool)
            else:
                off_diagonal = ~torch.eye(
                    batch_size, device=scores.device, dtype=torch.bool
                )
                extra_mask = in_batch_positive_mask[off_diagonal].reshape(
                    batch_size, -1
                )
        if self.baseline_args.baseline_use_in_batch_negatives:
            candidate_mask = torch.cat((candidate_mask, extra_mask), dim=-1)
            scores = torch.cat((scores, in_batch_scores), dim=-1)
            relevance_labels = torch.cat(
                (relevance_labels, relevance_labels.new_zeros(in_batch_scores.shape)),
                dim=-1,
            )
            if rank_labels is not None:
                # The teacher permutation orders the complete own-query slate.
                # Cross-query positives are unrelated candidates below that slate;
                # their shared zero label also excludes pairs among themselves.
                rank_labels = torch.cat(
                    (rank_labels, rank_labels.new_zeros(in_batch_scores.shape)),
                    dim=-1,
                )

        if self.baseline_args.baseline_loss == "infonce":
            # Binary identities remain independent of teacher grades and positions.
            positive_labels = torch.zeros_like(relevance_labels)
            if positive_mask is None:
                positive_labels[:, 0] = 1
            else:
                positive_labels[:, : positive_mask.size(1)] = positive_mask
            per_sample_loss = compute_infonce_loss(
                scores=scores,
                relevance_labels=positive_labels,
                temperature=self.baseline_args.baseline_temperature,
                candidate_mask=candidate_mask,
            )
        elif self.baseline_args.baseline_loss == "ranknet":
            per_sample_loss = compute_ranknet_loss(
                scores=scores,
                rank_labels=rank_labels
                if rank_labels is not None
                else relevance_labels,
                temperature=self.baseline_args.baseline_temperature,
                candidate_mask=candidate_mask,
            )
        elif self.baseline_args.baseline_loss == "lambdaloss":
            per_sample_loss = compute_lambdaloss_loss(
                scores=scores,
                relevance_labels=relevance_labels,
                k=self.baseline_args.baseline_ndcg_k,
                sigma=self.baseline_args.lambdaloss_sigma,
                candidate_mask=candidate_mask,
            )
        else:  # BaselineArguments validates this; keep the model failure explicit.
            raise ValueError(
                f"Unsupported baseline loss: {self.baseline_args.baseline_loss}"
            )
        return BaselineModelOutput(loss=per_sample_loss.mean() * loss_weight)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()


def main(argv: list[str] | None = None) -> None:
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            LoraArguments,
            BaselineArguments,
        )
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        baseline_args,
    ) = parse_arguments(
        parser=parser,
        base_slots=BASELINE_CONFIG_SLOTS,
        argv=argv,
    )

    guard_output_dir(training_args)
    setup_logging(
        training_args,
        {
            "Model": model_args,
            "Baseline": baseline_args,
        },
    )

    set_seed(training_args.seed)

    backbone, tokenizer = load_backbone_and_tokenizer(model_args, lora_args)
    model = BaselineModel(
        model=backbone,
        baseline_args=baseline_args,
        pooling_method=model_args.pooling_method,
    )
    model.train()

    apply_gradient_checkpointing(model, training_args, lora_args)

    train_dataset, data_collator = build_joint_embedding_data(
        data_args,
        training_args,
        tokenizer,
        model_args,
    )
    data_collator.include_cross_batch_metadata = baseline_args.extended_negative_pool

    trainer = BaselineTrainer(
        model_args=model_args,
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    resume = bool(
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.overwrite_output_dir
    )
    trainer.train(resume_from_checkpoint=True if resume else None)

    save_run_artifacts(
        trainer,
        training_args,
        tokenizer,
        model_args=model_args,
        baseline_args=baseline_args,
    )


if __name__ == "__main__":
    main()
    shutdown_distributed()
