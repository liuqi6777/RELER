"""Action-independent, stratified shortlists for fixed cross-query RL pools."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .contrastive import cross_query_document_pool
from .policy_math import (
    PolicyObjectiveResult,
    factorized_leave_one_out,
    tensor_statistics,
)
from .precision import fp32_scores
from .projection import conditional_projection_loss, pairwise_shortlist_loss
from .rewards import RewardEvaluator, RewardInputs, shortlist_positive_mask
from .rollout_rng import RolloutRNG

SHORTLIST_VERSION = 1


@dataclass(frozen=True)
class ShortlistSettings:
    """Runtime settings owned by the shortlist objective."""

    count: int
    size: int
    hard_count: int
    hard_pool_size: int
    pool_source: str
    cross_device: bool
    binary_weight: float
    pairwise_coef: float
    gradient_estimator: str

    @classmethod
    def from_source(cls, source) -> "ShortlistSettings":
        return cls(
            count=source.reward_shortlist_count,
            size=source.reward_shortlist_size,
            hard_count=source.reward_shortlist_hard_count,
            hard_pool_size=source.reward_shortlist_hard_pool_size,
            pool_source=source.reward_shortlist_pool_source,
            cross_device=source.reward_cross_device_negatives,
            binary_weight=source.reward_shortlist_binary_weight,
            pairwise_coef=source.reward_shortlist_pairwise_coef,
            gradient_estimator=source.gradient_estimator,
        )


@dataclass(frozen=True)
class _PreparedShortlists:
    pool: torch.Tensor
    indices: torch.Tensor
    selected_mask: torch.Tensor
    statistics: dict[str, torch.Tensor]
    loss_weight: float
    query_actions: torch.Tensor
    document_actions: torch.Tensor
    own_scores: torch.Tensor


@dataclass(frozen=True)
class _ShortlistEvaluation:
    loss: torch.Tensor
    statistics: dict[str, torch.Tensor]
    advantages: torch.Tensor
    degenerate_fraction: torch.Tensor
    reward_mean_by_query: torch.Tensor


def validate_shortlist_sampling(count, size, hard_count, hard_pool_size):
    for name, value, minimum in (
        ("count", count, 0),
        ("size", size, 1),
        ("hard_count", hard_count, 0),
        ("hard_pool_size", hard_pool_size, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"reward_shortlist_{name} must be an integer >= {minimum}")
    if hard_count > size or hard_pool_size < hard_count:
        raise ValueError("Shortlist hard_count must not exceed size or hard_pool_size")


def validate_shortlist_objectives(
    count,
    reward_terms,
    binary_weight,
    pairwise_coef=0.0,
    gradient_estimator="conditional_projection",
):
    if not math.isfinite(binary_weight) or not 0 <= binary_weight <= 1:
        raise ValueError("reward_shortlist_binary_weight must be finite and in [0, 1]")
    if binary_weight and (
        not count
        or len(reward_terms) != 1
        or reward_terms[0].type != "ndcg_in_batch"
        or reward_terms[0].weight != 1
        or reward_terms[0].name == "binary_ndcg"
    ):
        raise ValueError(
            "Binary shortlist mixing requires shortlists and one unit-weight nDCG term; "
            "binary_ndcg is reserved for the original-positive reward"
        )
    if not math.isfinite(pairwise_coef) or pairwise_coef < 0:
        raise ValueError(
            "reward_shortlist_pairwise_coef must be finite and non-negative"
        )
    if pairwise_coef and (not count or gradient_estimator != "conditional_projection"):
        raise ValueError(
            "Pairwise shortlist rewards require shortlists and conditional_projection"
        )


@torch.no_grad()
def shortlist_rewards(
    evaluator: RewardEvaluator,
    *,
    scores,
    labels,
    valid,
    rank_labels,
    cross_scores,
    binary_weight=0.0,
    positive_mask=None,
):
    """Return the mixed reward and unweighted term diagnostics on identical actions/pools."""
    inputs = RewardInputs(
        scores=scores,
        relevance_labels=labels,
        candidate_mask=valid,
        rank_labels=rank_labels,
        fixed_cross_scores=cross_scores,
    )
    evaluation = evaluator.evaluate(inputs)
    reward = evaluation.combined
    terms = dict(evaluation.term_rewards)
    if binary_weight:
        positives = shortlist_positive_mask(positive_mask, valid)
        binary = evaluator.evaluate(
            RewardInputs(
                scores=scores,
                relevance_labels=positives.float(),
                candidate_mask=valid,
                fixed_cross_scores=cross_scores,
            )
        ).combined
        terms["binary_ndcg"] = binary
        reward = (1 - binary_weight) * reward + binary_weight * binary
    return reward, terms


def shortlist_contract(head):
    if not getattr(head, "reward_shortlist_count", 0):
        return None
    contract = dict(
        version=SHORTLIST_VERSION,
        **{
            key: getattr(head, f"reward_shortlist_{key}")
            for key in ("count", "size", "hard_count", "hard_pool_size")
        },
    )
    # Preserve the exact v1 contract for existing cross-device/all-candidate runs.
    source = getattr(head, "reward_shortlist_pool_source", "cross_device_all")
    if source != "cross_device_all":
        contract["pool_source"] = source
    if getattr(head, "reward_shortlist_binary_weight", 0):
        contract["binary_reward"] = dict(
            weight=head.reward_shortlist_binary_weight,
            labels="positive_mask",
            version=1,
        )
    if getattr(head, "reward_shortlist_pairwise_coef", 0):
        contract["pairwise_reward"] = dict(
            coefficient=head.reward_shortlist_pairwise_coef,
            labels="positive_mask",
            version=1,
            pairs="all_own_positives_x_own_and_selected_negatives",
            reduction="per_query_mean",
            estimator="per_pair_conditional_projection",
            ties=0.5,
        )
    return contract


@torch.no_grad()
def sample_shortlists(scores, valid, *, count, size, hard_count, hard_pool_size):
    """Return indices/mask [B,T,K] and hard membership, using only mean scores.

    Partition each query's allowed pool into its highest-scoring H documents and
    the remainder. Randomly permute each stratum ONCE and interleave chunks at the
    requested ratio. When a stratum runs out, fill from the remaining UNUSED
    candidates. Wrap only after the ENTIRE pool is exhausted: coverage is exactly
    min(pool_size, T*K), with no within-list duplicates. Later lists can have a
    different hard fraction; coverage takes priority over a fixed hard quota.
    The first list and RNG consumption do not depend on T.

    On scarcity, transfer unavailable quota to the other stratum, then pad if
    the entire pool has fewer than K items. hard_count=0 is uniform over the
    entire allowed pool, not just the low-scoring remainder.
    """
    validate_shortlist_sampling(count, size, hard_count, hard_pool_size)
    if (
        count == 0
        or scores.ndim != 2
        or valid.shape != scores.shape
        or valid.dtype != torch.bool
    ):
        raise ValueError(
            "Sampling needs count > 0 and matching [batch,pool] scores/bool mask"
        )
    if not torch.isfinite(scores[valid]).all():
        raise ValueError("Valid shortlist mean scores must be finite")
    batch = scores.size(0)
    indices = torch.zeros((batch, count, size), dtype=torch.long, device=scores.device)
    mask = torch.zeros_like(indices, dtype=torch.bool)
    hard_mask = torch.zeros_like(mask)
    for b in range(batch):
        allowed = valid[b].nonzero(as_tuple=True)[0]
        if not allowed.numel():
            continue
        if hard_count:
            order = scores[b, allowed].argsort(descending=True, stable=True)
            allowed = allowed[order]
        nh = min(hard_pool_size, allowed.numel()) if hard_count else 0
        n = min(size, allowed.numel())
        take_hard = min(hard_count, nh, n)
        take_rest = min(n - take_hard, allowed.numel() - nh)
        take_hard = n - take_rest
        hard = allowed[:nh][torch.randperm(nh, device=scores.device)]
        rest = allowed[nh:][torch.randperm(allowed.numel() - nh, device=scores.device)]
        items = torch.cat((hard, rest))
        membership = torch.arange(items.numel(), device=scores.device) < nh
        if take_hard and take_rest:
            ih = torch.arange(nh, device=scores.device)
            ir = torch.arange(rest.numel(), device=scores.device)
            priority = torch.cat(
                (
                    ih // take_hard * n + ih % take_hard,
                    ir // take_rest * n + take_hard + ir % take_rest,
                )
            )
            order = priority.argsort(stable=True)
            items, membership = items[order], membership[order]
        positions = (
            torch.arange(count * n, device=scores.device).reshape(count, n)
            % items.numel()
        )
        indices[b, :, :n] = items[positions]
        mask[b, :, :n] = True
        hard_mask[b, :, :n] = membership[positions]
    return indices, mask, hard_mask


@torch.no_grad()
def shortlist_statistics(scores, allowed, indices, mask, hard_mask):
    counts = mask.sum(-1).float()
    pool_counts = allowed.sum(-1).float()
    unique = torch.stack(
        [
            indices[b][mask[b]].unique().numel() * scores.new_ones(())
            for b in range(scores.size(0))
        ]
    )
    total = counts.sum(-1)
    selected_scores = (
        scores.gather(1, indices.flatten(1))
        if scores.size(1)
        else counts.new_zeros(scores.size(0), indices.size(1) * indices.size(2))
    )
    selected_scores = selected_scores.masked_fill(~mask.flatten(1), 0)
    return {
        "shortlist/count": counts.new_tensor(indices.size(1)),
        "shortlist/pool_candidates_mean": pool_counts.mean(),
        "shortlist/pool_candidates_max": pool_counts.max(),
        "shortlist/unique_candidates_mean": unique.mean(),
        "shortlist/coverage_mean": (unique / pool_counts.clamp_min(1)).mean(),
        "shortlist/repeat_fraction": ((total - unique) / total.clamp_min(1)).mean(),
        "shortlist/hard_candidates_mean": hard_mask.sum(-1).float().mean(),
        "shortlist/selected_score_mean": selected_scores.sum()
        / total.sum().clamp_min(1),
        "reward_pool/cross_candidates_mean": counts.mean(),
        "reward_pool/cross_candidates_max": counts.max(),
    }


def _prepare_shortlists(
    *,
    settings: ShortlistSettings,
    rollout_rng: RolloutRNG,
    step: int,
    training: bool,
    query,
    document,
    labels: torch.Tensor,
    metadata,
    valid: torch.Tensor,
) -> _PreparedShortlists:
    pool, allowed, loss_weight = cross_query_document_pool(
        document.rollout_embeddings,
        metadata,
        include_negatives=settings.pool_source != "cross_device_representatives",
        cross_device=settings.cross_device,
        detach_documents=True,
    )
    if not settings.cross_device and torch.distributed.is_initialized():
        # Match the cross-device path's global query mean on uneven DDP tails.
        count = torch.tensor(labels.size(0), device=labels.device, dtype=torch.long)
        torch.distributed.all_reduce(count)
        loss_weight = torch.distributed.get_world_size() * labels.size(0) / count.item()

    with torch.no_grad(), fp32_scores(query.rollout_embeddings.device):
        query_actions = query.sampled_embeddings.detach().float()
        document_actions = document.sampled_embeddings.detach().float()
        pool = F.normalize(pool.float(), dim=-1)
        mean_scores = query.rollout_embeddings.detach().float() @ pool.T
        with rollout_rng.draw(
            query_actions.device,
            step,
            training,
            stream="shortlist",
        ):
            indices, selected_mask, hard_mask = sample_shortlists(
                mean_scores,
                allowed,
                count=settings.count,
                size=settings.size,
                hard_count=settings.hard_count,
                hard_pool_size=settings.hard_pool_size,
            )
        statistics = shortlist_statistics(
            mean_scores, allowed, indices, selected_mask, hard_mask
        )
        own_scores = torch.einsum("bid,bjmd->bijm", query_actions, document_actions)

    return _PreparedShortlists(
        pool=pool,
        indices=indices,
        selected_mask=selected_mask,
        statistics=statistics,
        loss_weight=loss_weight,
        query_actions=query_actions,
        document_actions=document_actions,
        own_scores=own_scores,
    )


def _evaluate_one_shortlist(
    *,
    settings: ShortlistSettings,
    evaluator: RewardEvaluator,
    query,
    document,
    prepared: _PreparedShortlists,
    shortlist_index: int,
    labels: torch.Tensor,
    rank_labels: torch.Tensor | None,
    positive_mask: torch.Tensor | None,
    valid: torch.Tensor,
    frozen_document_scale: torch.Tensor | None,
) -> _ShortlistEvaluation:
    with torch.no_grad(), fp32_scores(prepared.query_actions.device):
        fixed = (
            prepared.pool[prepared.indices[:, shortlist_index]]
            if prepared.pool.size(0)
            else prepared.pool.new_zeros(
                prepared.query_actions.size(0),
                settings.size,
                prepared.query_actions.size(-1),
            )
        )
        mask = prepared.selected_mask[:, shortlist_index]
        cross_scores = torch.einsum("bid,bkd->bik", prepared.query_actions, fixed)
        if frozen_document_scale is not None:
            cross_scores = cross_scores * frozen_document_scale
        cross_scores = cross_scores.masked_fill(~mask[:, None], -torch.inf)
        rewards, term_rewards = shortlist_rewards(
            evaluator,
            scores=prepared.own_scores,
            labels=labels,
            valid=valid,
            rank_labels=rank_labels,
            cross_scores=cross_scores,
            binary_weight=settings.binary_weight,
            positive_mask=positive_mask,
        )
        credit = factorized_leave_one_out(rewards, term_rewards)
        statistics = tensor_statistics(rewards, prefix="reward")
        statistics.update(credit.statistics)
        # Aggregate moments over cells/lists, not the mean of per-list stds.
        statistics["reward_second_moment"] = rewards.square().mean()
        statistics["reward_pool/zero_reward_frac"] = (rewards == 0).float().mean()
        for term_name, values in term_rewards.items():
            statistics[f"reward/{term_name}/n_distinct"] = evaluator.distinct_levels(
                values
            )
            if settings.binary_weight:
                statistics.update(
                    tensor_statistics(
                        values,
                        prefix=f"reward/{term_name}",
                        separator="/",
                    )
                )

    if settings.gradient_estimator == "conditional_projection":
        loss, ranks = conditional_projection_loss(
            query.policy_embeddings,
            document.policy_embeddings,
            prepared.query_actions,
            prepared.document_actions,
            rewards,
            query.kappa,
            valid,
            frozen_documents=fixed,
            frozen_mask=mask,
        )
        statistics["projection/query_span_rank_mean"] = ranks.float().mean()
        statistics["projection/query_span_rank_max"] = ranks.max().float()
    else:
        with fp32_scores(prepared.query_actions.device):
            query_log_prob = query.kappa.detach() * (
                query.policy_embeddings[:, None].float() * prepared.query_actions
            ).sum(-1)
            document_log_prob = document.kappa.detach() * (
                document.policy_embeddings[:, None].float() * prepared.document_actions
            ).masked_fill(~valid[:, None, :, None], 0).sum((-1, -2))
            loss = (
                -(credit.query * query_log_prob).mean()
                - (credit.documents * document_log_prob).mean()
            )

    if settings.pairwise_coef:
        pairwise_loss, pairwise_stats = pairwise_shortlist_loss(
            query.policy_embeddings,
            document.policy_embeddings,
            prepared.query_actions,
            prepared.document_actions,
            positive_mask,
            valid,
            fixed,
            mask,
            query.kappa,
            frozen_scale=(
                1.0 if frozen_document_scale is None else frozen_document_scale
            ),
            own_scores=prepared.own_scores,
            cross_scores=cross_scores,
        )
        statistics.update(pairwise_stats)
        statistics["train/loss_listwise"] = loss.detach()
        statistics["train/loss_pairwise"] = pairwise_loss.detach()
        statistics["train/loss_pairwise_weighted"] = (
            settings.pairwise_coef * pairwise_loss.detach()
        )
        statistics["reward/combined_mean"] = (
            rewards.mean()
            + settings.pairwise_coef * pairwise_stats["reward/pairwise/mean"]
        )
        loss = loss + settings.pairwise_coef * pairwise_loss

    return _ShortlistEvaluation(
        loss=loss,
        statistics=statistics,
        advantages=credit.flattened,
        degenerate_fraction=credit.degenerate_fraction,
        reward_mean_by_query=rewards.mean((1, 2)),
    )


def _reduce_shortlist_evaluations(
    *,
    base_statistics: dict[str, torch.Tensor],
    evaluations: list[_ShortlistEvaluation],
    loss_weight: float,
) -> PolicyObjectiveResult:
    aggregates = {}
    for evaluation in evaluations:
        for key, value in evaluation.statistics.items():
            if key not in aggregates:
                aggregates[key] = value
            elif key.endswith(("_min", "/min")):
                aggregates[key] = torch.minimum(aggregates[key], value)
            elif key.endswith(("_max", "/max")):
                aggregates[key] = torch.maximum(aggregates[key], value)
            else:
                aggregates[key] = aggregates[key] + value

    count = len(evaluations)
    statistics = dict(base_statistics)
    statistics.update(
        {
            key: value
            if key.endswith(("_min", "/min", "_max", "/max"))
            else value / count
            for key, value in aggregates.items()
        }
    )
    statistics["reward_std"] = (
        (statistics.pop("reward_second_moment") - statistics["reward_mean"].square())
        .clamp_min(0)
        .sqrt()
    )
    statistics["shortlist/reward_std_across_lists"] = (
        torch.stack(
            [evaluation.reward_mean_by_query for evaluation in evaluations], dim=1
        )
        .std(dim=1, unbiased=False)
        .mean()
    )
    return PolicyObjectiveResult(
        loss=torch.stack([evaluation.loss for evaluation in evaluations]).mean()
        * loss_weight,
        statistics=statistics,
        advantages=torch.cat(
            [evaluation.advantages for evaluation in evaluations], dim=1
        ),
        degenerate_fraction=torch.stack(
            [evaluation.degenerate_fraction for evaluation in evaluations]
        ).mean(),
    )


def shortlist_policy_objective(
    *,
    settings: ShortlistSettings,
    evaluator: RewardEvaluator,
    rollout_rng: RolloutRNG,
    step: int,
    training: bool,
    query,
    documents,
    labels: torch.Tensor,
    rank_labels: torch.Tensor | None,
    metadata,
    positive_mask: torch.Tensor | None,
    frozen_document_scale: torch.Tensor | None,
) -> PolicyObjectiveResult:
    """Evaluate the complete fixed-pool shortlist policy objective."""
    if len(documents) != 1 or not query.is_active or not documents[0].is_active:
        raise ValueError("Reward shortlists require joint query/document actions")
    document = documents[0]
    for component in (query, document):
        if not torch.allclose(
            component.policy_embeddings.detach(),
            component.rollout_embeddings,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("Reward shortlists require on-policy mean directions")
    if not torch.allclose(query.kappa, document.kappa):
        raise ValueError("Reward shortlists require the same fixed kappa")

    valid = document.document_mask
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    prepared = _prepare_shortlists(
        settings=settings,
        rollout_rng=rollout_rng,
        step=step,
        training=training,
        query=query,
        document=document,
        labels=labels,
        metadata=metadata,
        valid=valid,
    )
    evaluations = [
        _evaluate_one_shortlist(
            settings=settings,
            evaluator=evaluator,
            query=query,
            document=document,
            prepared=prepared,
            shortlist_index=index,
            labels=labels,
            rank_labels=rank_labels,
            positive_mask=positive_mask,
            valid=valid,
            frozen_document_scale=frozen_document_scale,
        )
        for index in range(settings.count)
    ]
    return _reduce_shortlist_evaluations(
        base_statistics=prepared.statistics,
        evaluations=evaluations,
        loss_weight=prepared.loss_weight,
    )
