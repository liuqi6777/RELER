"""Shared supervised InfoNCE and its direct auxiliary loss for embedding RL."""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.nn.functional import all_gather

from .precision import fp32_scores


def compute_in_batch_positive_scores(
    query_embeddings: Tensor,
    positive_embeddings: Tensor,
) -> Tensor:
    """Score other samples' detached representatives, excluding the diagonal."""
    if query_embeddings.dim() != 2 or positive_embeddings.dim() != 2:
        raise ValueError("query_embeddings and positive_embeddings must both be 2D")
    if query_embeddings.shape != positive_embeddings.shape:
        raise ValueError(
            "query_embeddings and positive_embeddings must have matching shapes"
        )
    query_embeddings = query_embeddings.float()
    positive_embeddings = positive_embeddings.float()
    batch_size = query_embeddings.size(0)
    if batch_size <= 1:
        return query_embeddings.new_empty((batch_size, 0))
    with fp32_scores(query_embeddings.device):
        cross_scores = torch.matmul(query_embeddings, positive_embeddings.detach().T)
    off_diagonal = ~torch.eye(batch_size, device=cross_scores.device, dtype=torch.bool)
    return cross_scores.masked_select(off_diagonal).reshape(batch_size, batch_size - 1)


def compute_infonce_loss(
    scores: Tensor,
    relevance_labels: Tensor,
    temperature: float = 0.03,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    """Mean over positives, each competing only against valid true negatives.

    This is the joint CL objective, kept unchanged when extracted from train_baseline.
    Returns one loss per query; a query with no negatives contributes zero.
    """
    if temperature <= 0:
        raise ValueError(f"infonce temperature must be positive, got {temperature}")
    if scores.dtype in {torch.float16, torch.bfloat16}:
        scores = scores.float()
    valid = (
        torch.ones_like(scores, dtype=torch.bool)
        if candidate_mask is None
        else candidate_mask.bool()
    )
    positives = (relevance_labels > 0) & valid
    negatives = valid & ~positives
    scaled = scores.masked_fill(~valid, 0) / float(temperature)
    negative_scores = scaled.masked_fill(~negatives, float("-inf"))
    has_negative = negatives.any(-1)
    negative_scores = torch.where(
        has_negative[:, None], negative_scores, torch.zeros_like(negative_scores)
    )
    log_negatives = torch.logsumexp(negative_scores, dim=-1, keepdim=True)
    losses = F.softplus(log_negatives - scaled)
    losses = losses.masked_fill(~positives | ~has_negative[:, None], 0)
    return losses.sum(-1) / positives.sum(-1).clamp_min(1)


def validate_aux_infonce(coefficient: float, temperature: float) -> None:
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("aux_infonce_coef must be finite and non-negative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("aux_infonce_temperature must be finite and positive")


def auxiliary_infonce_loss(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    positive_mask: Tensor | None,
    candidate_mask: Tensor | None = None,
    *,
    temperature: float = 0.03,
    use_in_batch_negatives: bool = False,
    in_batch_positive_mask: Tensor | None = None,
    index_route_ids: Tensor | None = None,
    strong_negatives: bool = False,
    cross_batch_metadata: list[dict] | None = None,
) -> Tensor:
    """Direct InfoNCE on unperturbed means, with caller-controlled gradient scope.

    Inactive action branches must be detached by the caller. Legacy cross-query
    documents are detached; strong negatives preserve the caller's gradient scope.
    Binary positive identities are mandatory and never inferred from teacher grades.
    """
    if document_embeddings.dim() != 3 or query_embeddings.shape != (
        document_embeddings.size(0),
        document_embeddings.size(2),
    ):
        raise ValueError(
            "Auxiliary InfoNCE expects queries [batch, dim] and documents [batch, candidates, dim]"
        )
    shape = document_embeddings.shape[:2]
    if (
        positive_mask is None
        or positive_mask.shape != shape
        or positive_mask.dtype != torch.bool
    ):
        raise ValueError(
            "Auxiliary InfoNCE requires binary positive_mask [batch, candidates]"
        )
    if candidate_mask is None:
        candidate_mask = torch.ones_like(positive_mask)
    if candidate_mask.shape != shape or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean and match positive_mask")
    if not (positive_mask & candidate_mask).any(-1).all():
        raise ValueError("Auxiliary InfoNCE requires a valid positive for every query")

    # Avoid bf16 matmul rounding before division by a small temperature. This is
    # a differentiable cast, not a detach, and uses the same forward embeddings.
    with fp32_scores(query_embeddings.device):
        queries = query_embeddings.float()
        documents = document_embeddings.float()
        scores = torch.einsum("bd,bmd->bm", queries, documents)
        positives = positive_mask
        valid = candidate_mask
        batch = queries.size(0)
        loss_weight = 1.0
        if strong_negatives:
            metadata = cross_batch_metadata
            if index_route_ids is not None and metadata is not None:
                if index_route_ids.shape != (batch,):
                    raise ValueError("index_route_ids must have shape [batch]")
                metadata = [
                    dict(row, index_route_id=route)
                    for row, route in zip(metadata, index_route_ids.tolist())
                ]
            cross_scores, cross_mask, loss_weight = cross_query_scores(
                queries,
                documents,
                metadata,
                include_negatives=True,
                cross_device=True,
                detach_documents=False,
            )
            scores = torch.cat((scores, cross_scores), dim=-1)
            valid = torch.cat((valid, cross_mask), dim=-1)
            positives = torch.cat((positives, torch.zeros_like(cross_mask)), dim=-1)
        elif use_in_batch_negatives and batch > 1:
            if not (positive_mask[:, 0] & candidate_mask[:, 0]).all():
                raise ValueError(
                    "In-batch representatives at candidate 0 must be valid positives"
                )
            off_diagonal = ~torch.eye(batch, device=scores.device, dtype=torch.bool)
            cross_allowed = in_batch_positive_mask
            if cross_allowed is not None and (
                cross_allowed.shape != (batch, batch)
                or cross_allowed.dtype != torch.bool
            ):
                raise ValueError(
                    "in_batch_positive_mask must be boolean [batch, batch]"
                )
            if index_route_ids is not None:
                if index_route_ids.shape != (batch,):
                    raise ValueError("index_route_ids must have shape [batch]")
                same_route = index_route_ids[:, None] == index_route_ids[None, :]
                cross_allowed = (
                    same_route if cross_allowed is None else cross_allowed & same_route
                )
            if cross_allowed is None:
                cross_mask = torch.ones(
                    (batch, batch - 1), device=scores.device, dtype=torch.bool
                )
            else:
                cross_mask = cross_allowed[off_diagonal].reshape(batch, batch - 1)
            cross_scores = compute_in_batch_positive_scores(queries, documents[:, 0])
            scores = torch.cat((scores, cross_scores), dim=-1)
            valid = torch.cat((valid, cross_mask), dim=-1)
            positives = torch.cat((positives, torch.zeros_like(cross_mask)), dim=-1)
        return (
            compute_infonce_loss(scores, positives, temperature, valid).mean()
            * loss_weight
        )


def aux_infonce_contract(head) -> dict | None:
    """Resume contract; disabled runs remain compatible with existing checkpoints."""
    coefficient = getattr(head, "aux_infonce_coef", 0.0)
    if coefficient == 0:
        return None
    contract = {
        "objective": "per_positive_infonce_v1",
        "coefficient": coefficient,
        "temperature": head.aux_infonce_temperature,
        "use_in_batch_negatives": head.aux_infonce_use_in_batch_negatives,
    }
    if getattr(head, "aux_infonce_strong_negatives", False):
        contract["strong_negatives"] = True
    return contract


def cross_document_mask(local_metadata, pool_metadata, width, own_offset, device):
    """Exclude own slate, known positives/IDs and duplicate cross-query documents."""
    mask = torch.zeros(
        (len(local_metadata), len(pool_metadata), width), dtype=torch.bool
    )
    for i, query in enumerate(local_metadata):
        seen = {key for key in query["keys"] if key is not None}
        seen.update(query["known_positive_keys"])
        known_ids = set(query["known_ids"])
        for j, other in enumerate(pool_metadata):
            if j == own_offset + i or (
                "index_route_id" in query
                and query["index_route_id"] != other.get("index_route_id")
            ):
                continue
            for k, key in enumerate(other["keys"][:width]):
                if key is None:
                    continue
                candidate_id = other["ids"][k]
                known = (
                    candidate_id not in (None, "")
                    and query["source"] == other["source"]
                    and candidate_id in known_ids
                )
                if key not in seen and not known:
                    mask[i, j, k] = True
                    seen.add(key)
    return mask.reshape(len(local_metadata), -1).to(device)


def cross_query_document_pool(
    documents: Tensor,
    metadata: list[dict],
    *,
    include_negatives: bool,
    cross_device: bool,
    detach_documents: bool,
) -> tuple[Tensor, Tensor, float]:
    """Gather documents and the shared CL/RL identity mask.

    Differentiable all_gather sums document gradients from all query ranks.
    Normal DDP/ZeRO gradient averaging then gives the global mean query loss;
    no extra world-size multiplier is needed. Both batch and slate tails are padded
    for collectives and excluded from scoring via identity metadata.
    """
    batch, slate_width, _ = documents.shape
    if (
        metadata is None
        or len(metadata) != batch
        or any(
            len(row["keys"]) != slate_width or len(row["ids"]) != slate_width
            for row in metadata
        )
    ):
        raise ValueError(
            "Cross-query document pools require collated cross_batch_metadata"
        )
    width = slate_width if include_negatives else 1
    documents = documents[:, :width]
    if detach_documents:
        documents = documents.detach()
    pool_metadata = metadata
    own_offset = 0
    global_query_count = batch
    if cross_device and dist.is_initialized() and dist.get_world_size() > 1:
        rank = dist.get_rank()
        payloads = [None] * dist.get_world_size()
        dist.all_gather_object(payloads, dict(metadata=metadata, width=width))
        sizes = [len(payload["metadata"]) for payload in payloads]
        global_query_count = sum(sizes)
        max_batch = max(sizes)
        width = max(payload["width"] for payload in payloads)
        padded = torch.nn.functional.pad(
            documents, (0, 0, 0, width - documents.size(1), 0, max_batch - batch)
        ).contiguous()
        if detach_documents:
            parts = [torch.empty_like(padded) for _ in payloads]
            dist.all_gather(parts, padded)
        else:
            parts = all_gather(padded)
        documents = torch.cat([part[:size] for part, size in zip(parts, sizes)], dim=0)
        pool_metadata = [row for payload in payloads for row in payload["metadata"]]
        own_offset = sum(sizes[:rank])
    valid = cross_document_mask(
        metadata, pool_metadata, width, own_offset, documents.device
    )
    # Rank-local mean losses need this correction if retained tails have unequal
    # query counts. Subsequent DDP averaging becomes a global query mean.
    loss_weight = (
        dist.get_world_size() * batch / global_query_count
        if cross_device and dist.is_initialized()
        else 1.0
    )
    return documents.reshape(-1, documents.size(-1)), valid, loss_weight


def cross_query_scores(
    queries: Tensor,
    documents: Tensor,
    metadata: list[dict],
    *,
    include_negatives: bool,
    cross_device: bool,
    detach_documents: bool,
) -> tuple[Tensor, Tensor, float]:
    """Score the shared document pool, preserving caller-selected gradients."""
    pool, valid, loss_weight = cross_query_document_pool(
        documents,
        metadata,
        include_negatives=include_negatives,
        cross_device=cross_device,
        detach_documents=detach_documents,
    )
    with fp32_scores(queries.device):
        scores = queries.float() @ pool.float().T
    return scores, valid, loss_weight
