"""Conditional projection and pairwise RLOO for joint vMF product rollouts."""

import torch
import torch.nn.functional as F

from .precision import fp32_scores
from .rewards import shortlist_positive_mask


def fixed_projection_candidates(
    document_embeddings,
    *,
    in_batch_positive_scores=None,
    in_batch_candidate_scores=None,
    fixed_cross_pool=None,
    fixed_cross_mask=None,
):
    """Build the exact fixed candidate union visible to a projection reward."""
    batch_size = document_embeddings.size(0)
    candidates = []
    masks = []
    if in_batch_positive_scores is not None:
        candidates.append(document_embeddings[:, 0][None].expand(batch_size, -1, -1))
        masks.append(torch.isfinite(in_batch_positive_scores).any(dim=(1, 2)))
    if fixed_cross_pool is not None:
        candidates.append(fixed_cross_pool[None].expand(batch_size, -1, -1))
        masks.append(fixed_cross_mask)
    elif in_batch_candidate_scores is not None:
        candidates.append(
            document_embeddings.reshape(1, -1, document_embeddings.size(-1)).expand(
                batch_size, -1, -1
            )
        )
        masks.append(torch.isfinite(in_batch_candidate_scores).any(dim=(1, 2)))

    if not candidates:
        return None, None
    return (
        candidates[0] if len(candidates) == 1 else torch.cat(candidates, dim=1),
        masks[0] if len(masks) == 1 else torch.cat(masks, dim=1),
    )


def project_span(vectors, columns, *, tolerance_width=None):
    """Project [..., D] onto the numerical span of [..., D, K].

    Reduced SVD handles duplicate vectors, zero padding and K > D. A plain QR
    would add arbitrary directions for dependent columns. Rank uses the standard
    dimension-scaled FP32 tolerance; this numerical cutoff is not a tuning knob.
    """
    basis, singular, _ = torch.linalg.svd(columns, full_matrices=False)
    width = columns.shape[-1] if tolerance_width is None else tolerance_width
    tolerance = (
        max(columns.shape[-2], width)
        * torch.finfo(columns.dtype).eps
        * singular[..., :1]
    )
    keep = singular > tolerance
    coordinates = torch.einsum("...dk,...d->...k", basis, vectors) * keep
    return torch.einsum("...dk,...k->...d", basis, coordinates), keep.sum(-1)


def project_with_fixed_documents(vectors, moving_columns, fixed_documents, fixed_mask):
    """Same numerical projector without replicating a large fixed pool over Gd.

    Factor F = U S V^T once per query; replacing F with U S preserves A A^T,
    its singular values and its left singular vectors. Keep the ORIGINAL width
    in the rank tolerance. A conservative singular-value bound allows an exact
    identity projection when F alone guarantees full numerical rank.
    """
    batch, group, dim = vectors.shape
    projected = torch.empty_like(vectors)
    ranks = torch.empty((batch, group), dtype=torch.long, device=vectors.device)
    width = moving_columns.size(-1) + fixed_documents.size(1)
    for b in range(batch):
        fixed = (
            fixed_documents[b]
            .detach()
            .float()
            .masked_fill(~fixed_mask[b, :, None], 0)
            .T
        )
        basis, singular, _ = torch.linalg.svd(fixed, full_matrices=False)
        moving = moving_columns[b]
        upper = (singular[0].square() + moving.square().sum((-2, -1)).max()).sqrt()
        threshold = max(dim, width) * torch.finfo(vectors.dtype).eps * upper
        if singular.numel() == dim and singular[-1] > threshold:
            projected[b] = vectors[b]
            ranks[b] = dim
            continue
        compressed = basis * singular[None, :]
        for draw in range(group):
            columns = torch.cat((moving[draw], compressed), dim=-1)
            projected[b, draw], ranks[b, draw] = project_span(
                vectors[b, draw],
                columns,
                tolerance_width=width,
            )
    return projected, ranks


def conditional_projection_loss(
    query_mean,
    document_means,
    query_actions,
    document_actions,
    rewards,
    kappa,
    candidate_mask,
    *,
    frozen_documents=None,
    frozen_mask=None,
    stream_frozen=False,
):
    """Return a surrogate and ranks for [B, Gq, Gd] cellwise LOO rewards.

    query_mean/document_means are live, on-policy unit directions. Actions and
    every quantity used to construct projectors/advantages are detached. Cross
    candidates must include ALL fixed directions used by any reward term.
    No [B, Gq, Gd, M, D] tensor is materialized.
    """
    batch, gq, dim = query_actions.shape
    gd, count = document_actions.shape[1:3]
    if (
        rewards.shape != (batch, gq, gd)
        or min(gq, gd) < 2
        or document_actions.shape != (batch, gd, count, dim)
        or query_mean.shape != (batch, dim)
        or document_means.shape != (batch, count, dim)
        or candidate_mask.shape != (batch, count)
    ):
        raise ValueError(
            "Conditional projection requires joint query/document product rollouts"
        )
    with fp32_scores(query_mean.device):
        with torch.no_grad():
            q, docs = query_actions.detach().float(), document_actions.detach().float()
            hq = F.normalize(query_mean.detach().float(), dim=-1)
            hd = F.normalize(document_means.detach().float(), dim=-1)
            reward = rewards.detach().float()
            aq = (reward - reward.mean(1, keepdim=True)) * (gq / (gq - 1))
            ad = (reward - reward.mean(2, keepdim=True)) * (gd / (gd - 1))
            weighted_q = torch.einsum("bij,bid->bjd", aq, q)
            columns = [
                hq[:, None, None].expand(-1, gd, -1, -1),
                docs.masked_fill(~candidate_mask[:, None, :, None], 0),
            ]
            if frozen_documents is not None:
                if (
                    frozen_mask is None
                    or frozen_mask.shape != frozen_documents.shape[:2]
                ):
                    raise ValueError(
                        "Frozen projection directions require a matching mask"
                    )
                if not stream_frozen:
                    fixed = (
                        frozen_documents.detach()
                        .float()
                        .masked_fill(~frozen_mask[..., None], 0)
                    )
                    columns.append(fixed[:, None].expand(-1, gd, -1, -1))
            moving = torch.cat(columns, dim=2).transpose(-2, -1)
            if stream_frozen and frozen_documents is not None:
                projected_q, ranks = project_with_fixed_documents(
                    weighted_q, moving, frozen_documents, frozen_mask
                )
            else:
                projected_q, ranks = project_span(weighted_q, moving)
            q_coefficient = projected_q.sum(1)

            # For document m at query draw i, span(h_m, q_i) has at most two
            # dimensions. Form the tangent in FP64 to avoid cancellation for
            # nearly parallel directions; the contractions remain FP32.
            hd64 = F.normalize(hd.double(), dim=-1)
            tangent = (
                q.double()[:, :, None]
                - (q.double()[:, :, None] * hd64[:, None]).sum(-1, keepdim=True)
                * hd64[:, None]
            )
            tangent_norm = tangent.norm(dim=-1, keepdim=True)
            tangent = (tangent / tangent_norm.clamp_min(1e-12)).float()
            tangent = tangent.masked_fill(tangent_norm <= 1e-12, 0)
            # First sum over document draws, then project for each query draw.
            weighted_docs = torch.einsum("bij,bjmd->bimd", ad, docs)
            tangent_coefficient = (weighted_docs * tangent).sum(-1, keepdim=True)
            projected_d = (tangent_coefficient * tangent).sum(1)
            # Retain the radial part too: this is the full conditional score
            # vector. Normalization of live means removes it in backpropagation.
            projected_d += (weighted_docs * hd[:, None]).sum((1, 3))[..., None] * hd
            projected_d = projected_d.masked_fill(~candidate_mask[..., None], 0)
            scale = torch.as_tensor(kappa, device=q.device).detach().float() / (gq * gd)
        surrogate = (
            -scale
            * (
                (query_mean.float() * q_coefficient).sum(-1)
                + (document_means.float() * projected_d).sum((1, 2))
            ).mean()
        )
    return surrogate, ranks


def _tangents(vectors, means):
    """Return stable unit tangents and their lengths for broadcastable inputs."""
    vectors, means = vectors.double(), means.double()
    radial = (vectors * means).sum(-1)
    tangent = vectors - radial[..., None] * means
    length = tangent.norm(dim=-1)
    unit = (tangent / length.clamp_min(1e-12)[..., None]).masked_fill(
        (length <= 1e-12)[..., None], 0
    )
    return unit.float(), radial.float(), length.float()


def _document_coefficients(mean, actions, queries, scores, advantage):
    """Return per-pair endpoint coefficients without a four-dimensional action grid."""
    unit, radial_q, length = _tangents(queries[:, None], mean[None])
    mean = mean.float()
    radial_d = (actions * mean[None]).sum(-1)
    projected_scores = (scores - radial_q[:, None] * radial_d[None]) / length[
        :, None
    ].clamp_min(1e-12)
    projected_scores = projected_scores.masked_fill((length <= 1e-12)[:, None], 0)
    tangent = torch.einsum("ip,ipd->pd", (advantage * projected_scores).sum(1), unit)
    radial = (advantage * radial_d[None]).sum((0, 1))[:, None] * mean
    return tangent + radial


def pairwise_shortlist_loss(
    query_mean,
    document_means,
    query_actions,
    document_actions,
    positive_mask,
    candidate_mask,
    frozen_documents,
    frozen_mask,
    kappa,
    *,
    gradient_estimator="conditional_projection",
    frozen_scale=1.0,
    own_scores=None,
    cross_scores=None,
    chunk_size=32,
):
    """Return the endpoint-local RLOO surrogate, optionally projected with CMP.

    Ties earn one half. Every valid annotated positive competes against every
    valid own negative and selected fixed negative. A query with no pair
    contributes zero. Pair membership never uses sampled scores, and only live
    means receive gradients. Both estimators share rewards, per-axis LOO and
    per-query pair normalization; only the action coefficients differ.
    """
    if gradient_estimator not in {"conditional_projection", "score_function"}:
        raise ValueError(
            f"Unsupported pairwise gradient estimator: {gradient_estimator!r}"
        )
    project = gradient_estimator == "conditional_projection"
    batch, gq, dim = query_actions.shape
    gd, count = document_actions.shape[1:3]
    if (
        min(gq, gd) < 2
        or query_mean.shape != (batch, dim)
        or document_means.shape != (batch, count, dim)
        or document_actions.shape != (batch, gd, count, dim)
        or candidate_mask.shape != (batch, count)
        or candidate_mask.dtype != torch.bool
        or frozen_documents.ndim != 3
        or frozen_documents.shape[0] != batch
        or frozen_documents.shape[-1] != dim
        or frozen_mask.shape != frozen_documents.shape[:2]
        or frozen_mask.dtype != torch.bool
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError(
            "Pairwise RLOO requires joint product actions, valid candidate masks "
            "and a positive chunk size"
        )
    positives = shortlist_positive_mask(positive_mask, candidate_mask)
    with fp32_scores(query_mean.device):
        with torch.no_grad():
            q, docs = query_actions.detach().float(), document_actions.detach().float()
            hq = F.normalize(query_mean.detach().double(), dim=-1)
            hd = F.normalize(document_means.detach().double(), dim=-1)
            fixed = frozen_documents.detach().float() * frozen_scale
            scores = (
                torch.einsum("bid,bjmd->bijm", q, docs)
                if own_scores is None
                else own_scores.detach().float()
            )
            cross = (
                torch.einsum("bid,bkd->bik", q, fixed)
                if cross_scores is None
                else cross_scores.detach().float()
            )
            if scores.shape != (batch, gq, gd, count) or cross.shape != (
                batch,
                gq,
                fixed.size(1),
            ):
                raise ValueError(
                    "Pairwise scores must match the shared shortlist action grid"
                )
            qc = torch.zeros_like(query_mean, dtype=torch.float32)
            dc = torch.zeros_like(document_means, dtype=torch.float32)
            means = q.new_zeros(batch)
            counts = q.new_zeros(batch)
            active = q.new_zeros(batch)
            rank_max = q.new_zeros(())
            for b in range(batch):
                pos = positives[b].nonzero(as_tuple=True)[0]
                neg = (candidate_mask[b] & ~positives[b]).nonzero(as_tuple=True)[0]
                cross_neg = frozen_mask[b].nonzero(as_tuple=True)[0] + count
                neg = torch.cat((neg, cross_neg))
                pairs = pos.numel() * neg.numel()
                counts[b] = pairs
                if not pairs:
                    continue
                a = pos.repeat_interleave(neg.numel())
                n = neg.repeat(pos.numel())
                if project:
                    pool = torch.cat(
                        (docs[b], fixed[b][None].expand(gd, -1, -1)), dim=1
                    )
                    radial_q = q[b] @ hq[b].float()
                pool_scores = torch.cat(
                    (scores[b], cross[b, :, None].expand(-1, gd, -1)), dim=-1
                )
                for start in range(0, pairs, chunk_size):
                    pa = a[start : start + chunk_size]
                    pn = n[start : start + chunk_size]
                    difference = pool_scores[..., pa] - pool_scores[..., pn]
                    reward = (difference > 0).float() + 0.5 * (difference == 0).float()
                    aq = (reward - reward.mean(0, keepdim=True)) * (gq / (gq - 1))
                    ad = (reward - reward.mean(1, keepdim=True)) * (gd / (gd - 1))
                    means[b] += reward.mean((0, 1)).sum() / pairs
                    flat = reward.flatten(0, 1)
                    active[b] += (
                        flat.max(0).values != flat.min(0).values
                    ).float().sum() / pairs

                    if not project:
                        qc[b] += torch.einsum("ijp,id->d", aq, q[b]) / pairs
                        # A pair credits only its trainable document endpoints.
                        weights = docs.new_zeros(gd, count)
                        weights.index_add_(1, pa, ad.sum(0))
                        own = pn < count
                        weights.index_add_(1, pn[own], ad[..., own].sum(0))
                        dc[b] += torch.einsum("jm,jmd->md", weights, docs[b]) / pairs
                        continue

                    delta = pool[:, pa] - pool[:, pn]
                    unit, radial_delta, length = _tangents(delta, hq[b])
                    q_projection = (
                        difference - radial_q[:, None, None] * radial_delta[None]
                    ) / length[None].clamp_min(1e-12)
                    q_projection = q_projection.masked_fill((length <= 1e-12)[None], 0)
                    qc[b] += (
                        torch.einsum("jp,jpd->d", (aq * q_projection).sum(0), unit)
                        / pairs
                    )
                    qc[b] += (
                        hq[b].float() * (aq * radial_q[:, None, None]).sum() / pairs
                    )
                    rank_max = torch.maximum(
                        rank_max, 1 + (length > 1e-12).float().max()
                    )

                    positive = _document_coefficients(
                        hd[b, pa], docs[b, :, pa], q[b], scores[b, ..., pa], ad
                    )
                    dc[b].index_add_(0, pa, positive / pairs)
                    own = pn < count
                    negative = _document_coefficients(
                        hd[b, pn[own]],
                        docs[b, :, pn[own]],
                        q[b],
                        scores[b, ..., pn[own]],
                        ad[..., own],
                    )
                    dc[b].index_add_(0, pn[own], negative / pairs)
            scale = torch.as_tensor(kappa, device=q.device).detach().float() / (gq * gd)
            stats = {
                "reward/pairwise/mean": means.mean(),
                "reward/pairwise/pairs_mean": counts.mean(),
                "reward/pairwise/active_pair_fraction": active.mean(),
                "reward/pairwise/no_pairs_frac": (counts == 0).float().mean(),
            }
            if project:
                stats["projection/pairwise_query_span_rank_max"] = rank_max
        loss = (
            -scale
            * (
                (query_mean.float() * qc).sum(-1)
                + (document_means.float() * dc).sum((1, 2))
            ).mean()
        )
    return loss, stats
