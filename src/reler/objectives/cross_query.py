"""Joint document actions shared across queries, with complete policy gradients.

Each document draw is one independent bundle over the whole candidate pool.
Query draws remain independent. Reward rows are local to each query; document
score coefficients collect contributions from every row that uses that action.
"""

import torch
import torch.nn.functional as F

from .contrastive import cross_document_mask, cross_query_document_pool
from .policy_math import (
    PolicyObjectiveResult,
    factorized_leave_one_out,
    tensor_statistics,
)
from .precision import fp32_scores
from .projection import project_span
from .rewards import RewardEvaluator, RewardInputs


def sampled_document_pool(
    means,
    actions,
    valid,
    *,
    cross_device,
    metadata=None,
    positive_mask=None,
    candidate_mask=None,
):
    """Return live pool means, detached actions, two foreign masks and DDP weight.

    Keep document *slots* as policy variables. Identity filtering chooses which
    slot represents a duplicate for each query, exactly as in the fixed pool.
    Only the small means tensor uses differentiable all_gather. Samples carry no
    autograd graph; backward communicates summed document score coefficients.
    """
    batch, width, dim = means.shape
    group = actions.size(1)
    packed = actions.detach().transpose(1, 2).reshape(batch, width, group * dim)
    if cross_device:
        pool, all_mask, weight = cross_query_document_pool(
            means,
            metadata,
            include_negatives=True,
            cross_device=True,
            detach_documents=False,
        )
        samples, _, _ = cross_query_document_pool(
            packed,
            metadata,
            include_negatives=True,
            cross_device=True,
            detach_documents=True,
        )
        # Cross-device mode requires all-document reward terms. Keeping the
        # second mask empty avoids silently defining a new representative pool.
        representatives = torch.zeros_like(all_mask)
    else:
        pool, samples, weight = (
            means.reshape(-1, dim),
            packed.reshape(-1, group * dim),
            1.0,
        )
        allowed = valid[None].expand(batch, -1, -1).clone()
        allowed[torch.arange(batch), torch.arange(batch)] = False
        all_mask = allowed.clone()
        representatives = torch.zeros_like(allowed)
        representatives[:, :, 0] = allowed[:, :, 0]
        if metadata is not None:
            all_mask &= cross_document_mask(
                metadata, metadata, width, 0, means.device
            ).reshape_as(all_mask)
            representatives[:, :, 0] &= cross_document_mask(
                metadata, metadata, 1, 0, means.device
            )
        if candidate_mask is not None:
            all_mask &= candidate_mask
        if positive_mask is not None:
            representatives[:, :, 0] &= positive_mask
        all_mask = all_mask.reshape(batch, -1)
        representatives = representatives.reshape(batch, -1)
    return (
        pool,
        samples.reshape(-1, group, dim).transpose(0, 1),
        all_mask,
        representatives,
        weight,
    )


@torch.no_grad()
def sampled_pool_rewards(
    evaluator: RewardEvaluator,
    query_actions,
    document_actions,
    pool_actions,
    labels,
    valid,
    all_mask,
    representative_mask,
    *,
    document_chunk=8,
):
    """Exact ranking rewards; never allocate [B,Gq,Gd,pool] in full.

    Candidate order is own slate followed by the original foreign pool, including
    its masked positions. This preserves the reward function's existing tie rule.
    No frozen-document rescaling: all documents here are sampled actions.
    """
    batch, gq, _ = query_actions.shape
    gd = document_actions.size(1)
    width = document_actions.size(2)
    with fp32_scores(query_actions.device):
        q = F.normalize(query_actions.float(), dim=-1)
        docs = F.normalize(document_actions.float(), dim=-1)
        pool = F.normalize(pool_actions.float(), dim=-1)
        own = torch.einsum("bid,bjmd->bijm", q, docs)
        result = {term.name: own.new_empty(batch, gq, gd) for term in evaluator.terms}
        requirements = evaluator.requirements
        for i in range(gq):
            for j in range(0, gd, document_chunk):
                stop = min(j + document_chunk, gd)
                cross = torch.einsum("bid,jpd->bijp", q[:, i : i + 1], pool[j:stop])
                evaluated = evaluator.evaluate(
                    RewardInputs(
                        scores=own[:, i : i + 1, j:stop],
                        relevance_labels=labels,
                        candidate_mask=valid,
                        in_batch_candidate_scores=(
                            cross.masked_fill(~all_mask[:, None, None], -torch.inf)
                            if requirements.all_candidates
                            else None
                        ),
                        in_batch_positive_scores=(
                            cross[..., ::width].masked_fill(
                                ~representative_mask[:, None, None, ::width],
                                -torch.inf,
                            )
                            if requirements.representative_candidates
                            else None
                        ),
                    )
                )
                for name, value in evaluated.term_rewards.items():
                    result[name][:, i : i + 1, j:stop] = value
    return evaluator.from_term_rewards(result), own


def shared_document_policy_objective(
    *,
    evaluator: RewardEvaluator,
    query,
    documents,
    labels: torch.Tensor,
    cross_device: bool,
    gradient_estimator: str,
    in_batch_positive_mask: torch.Tensor | None,
    in_batch_candidate_mask: torch.Tensor | None,
    metadata,
) -> PolicyObjectiveResult:
    """Evaluate the complete policy objective for shared document actions."""
    if len(documents) != 1 or not query.is_active or not documents[0].is_active:
        raise ValueError(
            "Cross-query gradients require active query and joint document components"
        )
    document = documents[0]
    for component in (query, document):
        if not torch.allclose(
            component.policy_embeddings.detach(),
            component.rollout_embeddings,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                "Cross-query document gradients require on-policy mean directions"
            )
    if not torch.allclose(query.kappa, document.kappa):
        raise ValueError("Cross-query document gradients require the same fixed kappa")

    valid = document.document_mask
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    pool, actions, all_mask, representative_mask, loss_weight = sampled_document_pool(
        document.policy_embeddings,
        document.sampled_embeddings,
        valid,
        cross_device=cross_device,
        metadata=metadata,
        positive_mask=in_batch_positive_mask,
        candidate_mask=in_batch_candidate_mask,
    )
    reward_evaluation, _ = sampled_pool_rewards(
        evaluator,
        query.sampled_embeddings,
        document.sampled_embeddings,
        actions,
        labels,
        valid,
        all_mask,
        representative_mask,
    )
    rewards = reward_evaluation.combined
    cross_mask = evaluator.cross_candidate_mask(all_mask, representative_mask)
    loss, ranks = sampled_pool_loss(
        query.policy_embeddings,
        document.policy_embeddings,
        pool,
        query.sampled_embeddings,
        document.sampled_embeddings,
        actions,
        rewards,
        query.kappa,
        valid,
        cross_mask,
        estimator=gradient_estimator,
    )

    statistics = tensor_statistics(rewards, prefix="reward")
    statistics["reward_pool/cross_candidates_mean"] = cross_mask.sum(-1).float().mean()
    statistics["reward_pool/cross_candidates_max"] = cross_mask.sum(-1).max().float()
    statistics["reward_pool/zero_reward_frac"] = (rewards == 0).float().mean()
    if gradient_estimator == "conditional_projection":
        statistics["projection/query_span_rank_mean"] = ranks.float().mean()
        statistics["projection/query_span_rank_max"] = ranks.max().float()

    credit = factorized_leave_one_out(rewards, reward_evaluation.term_rewards)
    statistics.update(credit.statistics)
    for term_name, values in reward_evaluation.term_rewards.items():
        statistics[f"reward/{term_name}/n_distinct"] = evaluator.distinct_levels(values)
        if len(reward_evaluation.term_rewards) > 1:
            statistics.update(
                tensor_statistics(
                    values,
                    prefix=f"reward/{term_name}",
                    separator="/",
                )
            )
            statistics[f"reward/{term_name}/group_std"] = (
                values.flatten(1).std(-1, unbiased=False).mean()
            )

    return PolicyObjectiveResult(
        loss=loss * loss_weight,
        statistics=statistics,
        advantages=credit.flattened,
        degenerate_fraction=credit.degenerate_fraction,
    )


@torch.no_grad()
def _shared_full_rank_draws(
    query_means, document_actions, pool_actions, valid, cross_mask
):
    """Certify full query spans once per document draw using common candidates.

    A subset's smallest singular value lower-bounds that of every enlarged
    span. The Frobenius norm upper-bounds its largest singular value, so this
    shortcut is conservative under project_span's FP32 numerical-rank rule.
    Otherwise use the complete per-query SVD; never drop reward-visible vectors.
    """
    gd, _, dim = pool_actions.shape
    common = cross_mask.all(0)
    full = torch.zeros(gd, device=pool_actions.device, dtype=torch.bool)
    if int(common.sum()) < dim:
        return full
    max_width = int((valid.sum(-1) + cross_mask.sum(-1) + 1).max())
    own_norm = (
        document_actions.masked_fill(~valid[:, None, :, None], 0)
        .square()
        .sum((-1, -2))
        .max(0)
        .values
    )
    bound = (
        pool_actions.square().sum((-1, -2))
        + own_norm
        + query_means.square().sum(-1).max()
    ).sqrt()
    threshold = max(dim, max_width) * torch.finfo(torch.float32).eps * bound
    for j in range(gd):
        singular = torch.linalg.svdvals(pool_actions[j, common].T)
        full[j] = singular[-1] > threshold[j]
    return full


@torch.no_grad()
def sampled_pool_coefficients(
    query_means,
    document_means,
    pool_means,
    query_actions,
    document_actions,
    pool_actions,
    rewards,
    valid,
    cross_mask,
    *,
    estimator,
    document_chunk=128,
):
    """Unscaled detached score coefficients; CP conditions separately per reward.

    LOO along i removes the current query action; LOO along j removes the whole
    current document bundle, including remote documents. Both are independent
    of the action they baseline. Summing document terms over queries therefore
    estimates the gradient of the mean reward over the shared joint policy.
    """
    with fp32_scores(query_actions.device):
        q = query_actions.detach().float()
        docs = document_actions.detach().float()
        pool = pool_actions.detach().float()
        hq = F.normalize(query_means.detach().float(), dim=-1)
        hd = F.normalize(document_means.detach().float(), dim=-1)
        hp = F.normalize(pool_means.detach().float(), dim=-1)
        reward = rewards.detach().float()
        batch, gq, gd = reward.shape
        dim = q.size(-1)
        aq = (reward - reward.mean(1, keepdim=True)) * (gq / (gq - 1))
        ad = (reward - reward.mean(2, keepdim=True)) * (gd / (gd - 1))
        qc, dc, pc = torch.zeros_like(hq), torch.zeros_like(hd), torch.zeros_like(hp)
        ranks = torch.zeros(batch, gd, device=q.device, dtype=torch.long)
        projected = estimator == "conditional_projection"
        full = (
            _shared_full_rank_draws(hq, docs, pool, valid, cross_mask)
            if projected
            else torch.zeros(gd, device=q.device, dtype=torch.bool)
        )
        full = full.tolist()  # One synchronization, not one per query/draw.
        for b in range(batch):
            local_ids = valid[b].nonzero(as_tuple=True)[0]
            pool_ids = cross_mask[b].nonzero(as_tuple=True)[0]
            local_count = len(local_ids)
            means = torch.cat((hd[b, local_ids], hp[pool_ids]), dim=0)
            weighted_q = aq[b].T @ q[b]
            if projected and all(full):
                qc[b] = weighted_q.sum(0)
                ranks[b] = dim
            elif projected:
                for j in range(gd):
                    if full[j]:
                        qc[b] += weighted_q[j]
                        ranks[b, j] = dim
                    else:
                        columns = torch.cat(
                            (hq[b : b + 1], docs[b, j, local_ids], pool[j, pool_ids]),
                            dim=0,
                        ).T
                        value, rank = project_span(weighted_q[j], columns)
                        qc[b] += value
                        ranks[b, j] = rank
            else:
                qc[b] = weighted_q.sum(0)
            coefficients = torch.empty_like(means)
            for start in range(0, len(means), document_chunk):
                stop = start + document_chunk
                pieces = []
                if start < local_count:
                    pieces.append(docs[b, :, local_ids[start : min(stop, local_count)]])
                if stop > local_count:
                    pieces.append(
                        pool[
                            :,
                            pool_ids[max(0, start - local_count) : stop - local_count],
                        ]
                    )
                samples = torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
                mean = means[start:stop]
                if not projected:
                    coefficients[start:stop] = torch.einsum(
                        "j,jmd->md", ad[b].sum(0), samples
                    )
                    continue
                # Keep only [Gq,chunk,D], not [B,Gq,Gd,pool,D]. Each reward row
                # gets its own span(h_m, q_i) before coefficients are summed.
                weighted = torch.einsum("ij,jmd->imd", ad[b], samples)
                mean64 = F.normalize(mean.double(), dim=-1)
                tangent = (
                    q[b].double()[:, None]
                    - (q[b].double()[:, None] * mean64[None]).sum(-1, keepdim=True)
                    * mean64[None]
                )
                norm = tangent.norm(dim=-1, keepdim=True)
                tangent = (
                    (tangent / norm.clamp_min(1e-12))
                    .float()
                    .masked_fill(norm <= 1e-12, 0)
                )
                value = ((weighted * tangent).sum(-1, keepdim=True) * tangent).sum(0)
                value += (weighted * mean[None]).sum((0, 2))[:, None] * mean
                coefficients[start:stop] = value
            dc[b, local_ids] = coefficients[:local_count]
            pc.index_add_(0, pool_ids, coefficients[local_count:])
    return qc, dc, pc, ranks


def sampled_pool_loss(
    query_means,
    document_means,
    pool_means,
    query_actions,
    document_actions,
    pool_actions,
    rewards,
    kappa,
    valid,
    cross_mask,
    *,
    estimator,
):
    qc, dc, pc, ranks = sampled_pool_coefficients(
        query_means,
        document_means,
        pool_means,
        query_actions,
        document_actions,
        pool_actions,
        rewards,
        valid,
        cross_mask,
        estimator=estimator,
    )
    with fp32_scores(query_means.device):
        scale = (
            torch.as_tensor(kappa, device=query_means.device).detach().float()
            / rewards.numel()
        )
        # Always include the pool term, even when every local mask is empty:
        # differentiable all_gather's backward must run on every rank.
        loss = -scale * (
            (query_means.float() * qc).sum()
            + (document_means.float() * dc).sum()
            + (pool_means.float() * pc).sum()
        )
    return loss, ranks
