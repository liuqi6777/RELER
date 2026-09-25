"""Ranking reward definitions and validation for RELER."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

SUPPORTED_REWARD_TYPES = {
    "ndcg",
    "ndcg_in_batch",
    "rbo",
    "top_weighted_pairwise",
    "contrastive",
    "infonce",
    "mrr",
    "mrr_in_batch",
}

# Rewards split into two families with incomparable scales: ranking metrics are bounded in
# [0, 1] while the contrastive margins live on a temperature-scaled log-score axis whose
# within-group spread is typically an order of magnitude larger. Mixing families under a raw
# weighted sum therefore does NOT mix them in the ratio of their weights (see
# SUPPORTED_REWARD_COMBINE_MODES), which is what these sets are used to warn about.
BOUNDED_REWARD_TYPES = frozenset(
    {"ndcg", "ndcg_in_batch", "mrr", "mrr_in_batch", "rbo", "top_weighted_pairwise"}
)
UNBOUNDED_REWARD_TYPES = frozenset({"contrastive", "infonce"})

# 'sum'            -- R = sum_i w_i R_i, then one advantage over the combined reward. Keeps the
#                     terms' relative effect sizes, so w_i is a weight on the *raw* reward.
# 'normalized_sum' -- A = sum_i w_i A_i with each A_i group-standardized on its own. The loss is
#                     linear in the advantage, so this is a scale-free combination in which w_i
#                     really is the mixing ratio. Preferred whenever families are mixed.
SUPPORTED_REWARD_COMBINE_MODES = ("sum", "normalized_sum")

_TERM_FIELD_ALIASES = {
    "k": "k",
    "ndcg_k": "k",
    "reward_ndcg_k": "k",
    "cutoff": "k",
    "temperature": "temperature",
    "contrastive_temperature": "temperature",
    "tau": "temperature",
    "p": "rbo_p",
    "rbo_p": "rbo_p",
    "reward_rbo_p": "rbo_p",
    "weight": "weight",
    "w": "weight",
    "name": "name",
    "type": "type",
    "reward_type": "type",
    "include_negatives": "ndcg_in_batch_include_negatives",
    "ndcg_in_batch_include_negatives": "ndcg_in_batch_include_negatives",
    "in_batch_negatives": "contrastive_use_in_batch_negatives",
    "contrastive_use_in_batch_negatives": "contrastive_use_in_batch_negatives",
}

_TRUE_STRINGS = {"true", "1", "yes", "y", "on"}
_FALSE_STRINGS = {"false", "0", "no", "n", "off"}


@dataclass(frozen=True)
class RewardTerm:
    """One additive term of the reward.

    ``k``/``temperature``/``rbo_p``/the two in-batch flags may be left unset (``None``), in
    which case :func:`normalize_reward_terms` fills them from the run-level defaults. That
    keeps a single-term config byte-identical to the pre-combination behaviour.
    """

    type: str
    weight: float = 1.0
    k: int | None = None
    temperature: float | None = None
    ndcg_in_batch_include_negatives: bool | None = None
    contrastive_use_in_batch_negatives: bool | None = None
    name: str = ""
    rbo_p: float | None = None


@dataclass(frozen=True)
class RewardInputs:
    """Score tables and supervision consumed by a reward implementation.

    The policy owns action sampling and score construction.  Reward code owns
    only the interpretation of those scores, which keeps new reward families
    independent from the GRPO rollout implementation.
    """

    scores: torch.Tensor
    relevance_labels: torch.Tensor
    candidate_mask: torch.Tensor | None = None
    rank_labels: torch.Tensor | None = None
    in_batch_positive_scores: torch.Tensor | None = None
    in_batch_candidate_scores: torch.Tensor | None = None
    fixed_cross_scores: torch.Tensor | None = None


@dataclass(frozen=True)
class RewardSignal:
    """One generic signal that the policy converts into advantages."""

    name: str
    weight: float
    values: torch.Tensor


@dataclass(frozen=True)
class RewardEvaluation:
    """Raw term values, their weighted sum, and generic weighted components."""

    term_rewards: dict[str, torch.Tensor]
    combined: torch.Tensor
    signals: tuple[RewardSignal, ...]


@dataclass(frozen=True)
class RewardRequirements:
    """Candidate pools required to evaluate a configured reward."""

    representative_candidates: bool
    all_candidates: bool


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}")


def _parse_term_string(spec: str) -> dict:
    """Parse the compact CLI form ``type[:weight][,key=value]*``."""
    head, _, tail = spec.partition(",")
    reward_type, _, weight = head.partition(":")
    fields: dict = {"type": reward_type.strip()}
    if weight.strip():
        fields["weight"] = weight.strip()

    for raw_field in tail.split(","):
        raw_field = raw_field.strip()
        if not raw_field:
            continue
        key, separator, value = raw_field.partition("=")
        if not separator:
            raise ValueError(
                f"Malformed reward term field {raw_field!r} in {spec!r}; expected 'key=value'"
            )
        fields[key.strip()] = value.strip()
    return fields


def _build_term(fields: Mapping) -> RewardTerm:
    normalized_fields: dict = {}
    for raw_key, value in fields.items():
        key = _TERM_FIELD_ALIASES.get(str(raw_key).strip().lower())
        if key is None:
            raise ValueError(
                f"Unsupported reward term field: {raw_key!r}. "
                f"Supported fields: {sorted(set(_TERM_FIELD_ALIASES))}"
            )
        if value is None:
            continue
        normalized_fields[key] = value

    if "type" not in normalized_fields:
        raise ValueError(
            f"Reward term is missing the required 'type' field: {dict(fields)!r}"
        )

    reward_type = str(normalized_fields["type"]).strip().lower()
    if reward_type not in SUPPORTED_REWARD_TYPES:
        raise ValueError(
            f"Unsupported reward type: {reward_type}. Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
        )

    weight = float(normalized_fields.get("weight", 1.0))
    if not (weight == weight) or weight in {float("inf"), float("-inf")}:
        raise ValueError(f"Reward term weight must be finite, got {weight}")

    k = normalized_fields.get("k")
    if k is not None:
        k = None if str(k).strip().lower() in {"none", "null", ""} else int(k)

    temperature = normalized_fields.get("temperature")
    if temperature is not None:
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError(
                f"Reward term temperature must be positive, got {temperature}"
            )

    rbo_p = normalized_fields.get("rbo_p")
    if rbo_p is not None:
        rbo_p = float(rbo_p)
        if not 0.0 <= rbo_p < 1.0:
            raise ValueError(f"RBO persistence p must lie in [0, 1), got {rbo_p}")

    include_negatives = normalized_fields.get("ndcg_in_batch_include_negatives")
    use_in_batch_negatives = normalized_fields.get("contrastive_use_in_batch_negatives")

    return RewardTerm(
        type=reward_type,
        weight=weight,
        k=k,
        temperature=temperature,
        rbo_p=rbo_p,
        ndcg_in_batch_include_negatives=(
            None if include_negatives is None else _coerce_bool(include_negatives)
        ),
        contrastive_use_in_batch_negatives=(
            None
            if use_in_batch_negatives is None
            else _coerce_bool(use_in_batch_negatives)
        ),
        name=str(normalized_fields.get("name", "")).strip(),
    )


def normalize_reward_terms(
    reward_terms,
    default_k: int | None = 10,
    default_temperature: float = 0.03,
    default_ndcg_in_batch_include_negatives: bool = False,
    default_contrastive_use_in_batch_negatives: bool = False,
    default_rbo_p: float = 0.9,
) -> tuple[RewardTerm, ...]:
    """Normalize a reward-term spec into fully resolved :class:`RewardTerm` objects.

    Accepts, in order of increasing verbosity:

    * a single type string -- ``"ndcg"``
    * the compact CLI form -- ``"ndcg:1.0,k=16;contrastive:0.5,in_batch_negatives=true"``
    * a YAML list of mappings -- ``[{type: ndcg, weight: 1.0, k: 16}, ...]``

    Unset per-term fields inherit the run-level defaults, so the single-term spec reproduces
    the legacy ``reward_type`` behaviour exactly.
    """
    if reward_terms is None:
        raise ValueError("reward_terms must not be None")
    if not 0.0 <= default_rbo_p < 1.0:
        raise ValueError(
            f"Default RBO persistence p must lie in [0, 1), got {default_rbo_p}"
        )

    if isinstance(reward_terms, RewardTerm):
        raw_terms: list = [reward_terms]
    elif isinstance(reward_terms, Mapping):
        raw_terms = [reward_terms]
    elif isinstance(reward_terms, str):
        raw_terms = [
            chunk.strip() for chunk in reward_terms.split(";") if chunk.strip()
        ]
    elif isinstance(reward_terms, Sequence):
        raw_terms = list(reward_terms)
    else:
        raise ValueError(
            "reward_terms must be a type string, a 'type:weight,key=value;...' string, "
            "or a list of mappings"
        )

    if not raw_terms:
        raise ValueError("reward_terms must contain at least one term")

    terms: list[RewardTerm] = []
    for raw_term in raw_terms:
        if isinstance(raw_term, RewardTerm):
            term = raw_term
        elif isinstance(raw_term, Mapping):
            term = _build_term(raw_term)
        elif isinstance(raw_term, str):
            term = _build_term(_parse_term_string(raw_term))
        else:
            raise ValueError(f"Unsupported reward term entry: {raw_term!r}")

        term = replace(
            term,
            k=default_k if term.k is None else term.k,
            temperature=default_temperature
            if term.temperature is None
            else term.temperature,
            rbo_p=default_rbo_p if term.rbo_p is None else term.rbo_p,
            ndcg_in_batch_include_negatives=(
                default_ndcg_in_batch_include_negatives
                if term.ndcg_in_batch_include_negatives is None
                else term.ndcg_in_batch_include_negatives
            ),
            contrastive_use_in_batch_negatives=(
                default_contrastive_use_in_batch_negatives
                if term.contrastive_use_in_batch_negatives is None
                else term.contrastive_use_in_batch_negatives
            ),
        )
        terms.append(term)

    if all(term.weight == 0.0 for term in terms):
        raise ValueError("At least one reward term must carry a non-zero weight")

    # Names index the per-term metrics, so they have to be unique and stable across ranks.
    named_terms: list[RewardTerm] = []
    used_names: dict[str, int] = {}
    for term in terms:
        base_name = term.name or term.type
        occurrence = used_names.get(base_name, 0)
        used_names[base_name] = occurrence + 1
        if occurrence and term.name:
            raise ValueError(f"Duplicate reward term name: {term.name!r}")
        name = base_name if not occurrence else f"{base_name}_{occurrence + 1}"
        named_terms.append(replace(term, name=name))
    return tuple(named_terms)


def normalize_reward_combine_mode(mode) -> str:
    if (
        not isinstance(mode, str)
        or mode.strip().lower() not in SUPPORTED_REWARD_COMBINE_MODES
    ):
        raise ValueError(
            f"Unsupported reward_combine mode: {mode!r}. "
            f"Expected one of {SUPPORTED_REWARD_COMBINE_MODES}."
        )
    return mode.strip().lower()


def reward_terms_mix_scales(reward_terms: Sequence[RewardTerm]) -> bool:
    """True when bounded ranking rewards are mixed with unbounded contrastive margins."""
    types = {term.type for term in reward_terms if term.weight != 0.0}
    return bool(types & BOUNDED_REWARD_TYPES) and bool(types & UNBOUNDED_REWARD_TYPES)


def ranking_reward_pool_size(
    term: RewardTerm, slate_size: int, batch_size: int
) -> int | None:
    """How many candidates a rank-based term ranks over. None for the score-based families."""
    if term.type in {"mrr", "ndcg", "rbo", "top_weighted_pairwise"}:
        return slate_size
    if term.type in {"ndcg_in_batch", "mrr_in_batch"}:
        if term.ndcg_in_batch_include_negatives:
            return batch_size * slate_size
        return slate_size + max(batch_size - 1, 0)
    return None


def warn_on_inert_cutoffs(
    reward_terms: Sequence[RewardTerm],
    slate_size: int,
    batch_size: int,
) -> list[str]:
    """Report rank-based terms whose cutoff can never bind, and how coarse each one is.

    Three cases, distinguished because only one of them is a mistake:

    * ``k > pool`` is a config error. The cutoff can never apply, so ``@k`` names a truncation
      that does not exist, and every caption quoting it is wrong. Easy to reintroduce by
      editing ``slate_size`` alone, since the cutoff lives in a different config slot.
    * ``k == pool`` is the full metric by construction. That is the intended setting for the
      own-slate rows, so it is stated rather than flagged.
    * whatever ``k`` is, a small pool caps the reward's *resolution*: a rank-based reward takes
      at most one value per candidate, so a group of G rollouts over an n-candidate pool cannot
      resolve more than min(G, n) levels. The rest tie, and tied groups contribute no gradient.

    Returns the lines rather than logging them, so callers decide where they go and tests can
    assert on them.
    """
    warnings: list[str] = []
    for term in reward_terms:
        pool = ranking_reward_pool_size(term, slate_size, batch_size)
        if pool is None:
            continue
        if term.k is not None and term.k > pool:
            warnings.append(
                f"reward term '{term.name}': cutoff k={term.k} EXCEEDS its {pool}-candidate "
                f"pool, so no truncation ever happens and '@{term.k}' is a mislabel. Set "
                f"k <= {pool}."
            )
        cutoff = (
            "no cutoff, i.e. the full metric"
            if term.k is None or term.k == pool
            else f"cutoff @{min(term.k, pool)}"
        )
        if term.type in {"rbo", "top_weighted_pairwise"}:
            warnings.append(
                f"reward term '{term.name}': pool={pool} ({cutoff}); this permutation-native "
                "reward can realize more distinct values than the candidate count."
            )
        else:
            warnings.append(
                f"reward term '{term.name}': pool={pool} ({cutoff}), so the reward has at most "
                f"{pool} distinct values; watch reward/{term.name}/n_distinct against the group size."
            )
    return warnings


def reward_terms_need_in_batch_positives(reward_terms: Sequence[RewardTerm]) -> bool:
    return any(
        (
            term.type in {"ndcg_in_batch", "mrr_in_batch"}
            and not term.ndcg_in_batch_include_negatives
        )
        or (
            term.type in {"contrastive", "infonce"}
            and term.contrastive_use_in_batch_negatives
        )
        for term in reward_terms
    )


def reward_terms_need_in_batch_candidates(reward_terms: Sequence[RewardTerm]) -> bool:
    return any(
        term.type in {"ndcg_in_batch", "mrr_in_batch"}
        and term.ndcg_in_batch_include_negatives
        for term in reward_terms
    )


def shortlist_positive_mask(
    positive_mask: torch.Tensor | None,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Validate annotated positive identities for shortlist objectives."""
    if (
        positive_mask is None
        or positive_mask.shape != valid.shape
        or positive_mask.dtype != torch.bool
    ):
        raise ValueError(
            "Shortlist binary objectives require boolean positive_mask matching candidates"
        )
    positives = positive_mask & valid
    if not positives.any(-1).all():
        raise ValueError(
            "Shortlist binary objectives require a valid annotated positive per query"
        )
    return positives


def realized_reward_levels(rewards: torch.Tensor) -> torch.Tensor:
    """Mean distinct reward values realized inside each sample's rollout group."""
    flattened = rewards.detach().float().reshape(rewards.size(0), -1)
    if flattened.size(-1) < 2:
        return torch.ones((), device=rewards.device, dtype=torch.float32)
    ordered = flattened.sort(dim=-1).values
    tolerance = 1e-4 * flattened.abs().mean(dim=-1, keepdim=True).clamp_min(1.0)
    return ((ordered.diff(dim=-1).abs() > tolerance).sum(dim=-1) + 1).float().mean()


def compute_reward_terms(
    reward_terms: Sequence[RewardTerm],
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    relevance_scheme: str | None = None,
    in_batch_positive_scores: torch.Tensor | None = None,
    in_batch_candidate_scores: torch.Tensor | None = None,
    rank_labels: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate every reward term against one shared score table.

    The caller builds the (expensive) in-batch score tables once; each term picks the ones it
    needs via its own flags, so terms never pay for tables they ignore.
    """
    return {
        term.name: compute_reward_from_scores(
            scores=scores,
            candidate_mask=candidate_mask,
            relevance_labels=relevance_labels,
            rank_labels=rank_labels,
            reward_type=term.type,
            k=term.k,
            ndcg_in_batch_include_negatives=bool(term.ndcg_in_batch_include_negatives),
            contrastive_use_in_batch_negatives=bool(
                term.contrastive_use_in_batch_negatives
            ),
            contrastive_temperature=float(term.temperature),
            rbo_p=float(term.rbo_p),
            relevance_scheme=relevance_scheme,
            in_batch_positive_scores=in_batch_positive_scores,
            in_batch_candidate_scores=in_batch_candidate_scores,
        )
        for term in reward_terms
    }


def resolve_relevant_mask(
    ranked_relevance: torch.Tensor,
    relevance_labels: torch.Tensor,
    relevance_scheme: str | None = None,
) -> torch.Tensor:
    """Return the binary relevance mask used by MRR.

    With no explicit scheme the threshold is inferred from the labels: graded labels
    (which reach 3) count only grade >= 2 as relevant, binary labels count anything > 0.
    """
    if relevance_scheme is not None and relevance_scheme not in {"graded", "binary"}:
        raise ValueError(f"Unsupported relevance scheme: {relevance_scheme}")

    use_graded_threshold = (
        relevance_scheme == "graded"
        if relevance_scheme is not None
        else bool((relevance_labels > 1).any().item())
    )
    return ranked_relevance >= 2.0 if use_graded_threshold else ranked_relevance > 0.0


def _temperature_scaled_logsumexp(
    values: torch.Tensor,
    temperature: float,
    empty_value: float = 0.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    temperature_tensor = torch.as_tensor(
        float(temperature),
        device=values.device,
        dtype=values.dtype,
    )
    finite_mask = torch.isfinite(values)
    scaled_values = torch.where(
        finite_mask,
        values / temperature_tensor,
        torch.full_like(values, float("-inf")),
    )
    aggregated_values = temperature_tensor * torch.logsumexp(scaled_values, dim=-1)
    default_values = torch.full_like(aggregated_values, fill_value=empty_value)
    return torch.where(finite_mask.any(dim=-1), aggregated_values, default_values)


def compute_reward_terms_over_fixed_pool(
    reward_terms,
    *,
    scores,
    relevance_labels,
    cross_scores,
    candidate_mask=None,
    rank_labels=None,
):
    """Evaluate [B,Gq,Gd,M] scores over ALL fixed distractors in bounded chunks.

    Cross scores are [B,Gq,P] and independent of document actions. Slicing the
    query-action axis avoids materializing [B,Gq,Gd,P]; original candidate order
    and topk tie behavior are preserved, including ties with labeled documents.
    """
    if (
        scores.ndim != 4
        or cross_scores.ndim != 3
        or scores.shape[:2] != cross_scores.shape[:2]
    ):
        raise ValueError(
            "Fixed-pool reward requires product scores and query-action cross scores"
        )
    parts = {term.name: [] for term in reward_terms}
    for draw in range(scores.size(1)):
        extra = cross_scores[:, draw : draw + 1, None, :].expand(
            -1, -1, scores.size(2), -1
        )
        evaluated = compute_reward_terms(
            reward_terms,
            scores=scores[:, draw : draw + 1],
            relevance_labels=relevance_labels,
            candidate_mask=candidate_mask,
            rank_labels=rank_labels,
            in_batch_candidate_scores=extra,
        )
        for name, reward in evaluated.items():
            parts[name].append(reward)
    return {name: torch.cat(values, dim=1) for name, values in parts.items()}


def _extend_ranking_pool(
    scores: torch.Tensor,
    labels: torch.Tensor,
    extra_scores: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append unlabeled cross-query candidates to a ranking pool."""
    if extra_scores is None:
        return scores, labels
    return (
        torch.cat((scores, extra_scores), dim=-1),
        torch.cat((labels, torch.zeros_like(extra_scores)), dim=-1),
    )


def _rbo_reward(
    scores: torch.Tensor,
    teacher_order: torch.Tensor,
    cutoff: int,
    persistence: float,
) -> torch.Tensor:
    """Return finite-depth rank-biased overlap for flattened rollouts."""
    batch_size, rollout_count, slate_length = scores.shape
    metric_dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    positions = torch.arange(
        1, slate_length + 1, device=scores.device, dtype=torch.long
    )
    predicted_order = scores.argsort(dim=-1, descending=True, stable=True)
    teacher_positions = torch.empty_like(teacher_order)
    teacher_positions.scatter_(
        dim=-1,
        index=teacher_order,
        src=positions.view(1, -1).expand_as(teacher_order),
    )
    predicted_positions = torch.empty_like(predicted_order)
    predicted_positions.scatter_(
        dim=-1,
        index=predicted_order,
        src=positions.view(1, 1, -1).expand_as(predicted_order),
    )

    # A document enters the prefix intersection at the deeper of its teacher and
    # predicted positions. Histogram entry depths instead of materializing two
    # [batch, rollout, slate, slate] prefix-membership tensors.
    entry_depths = torch.maximum(predicted_positions, teacher_positions.unsqueeze(1))
    enters_by_cutoff = entry_depths <= cutoff
    overlap_histogram = torch.zeros(
        batch_size,
        rollout_count,
        cutoff,
        device=scores.device,
        dtype=metric_dtype,
    )
    overlap_histogram.scatter_add_(
        dim=-1,
        index=entry_depths.clamp_max(cutoff) - 1,
        src=enters_by_cutoff.to(metric_dtype),
    )
    depths = torch.arange(1, cutoff + 1, device=scores.device, dtype=metric_dtype)
    agreement = overlap_histogram.cumsum(dim=-1) / depths
    depth_weights = torch.as_tensor(
        persistence, device=scores.device, dtype=metric_dtype
    ).pow(torch.arange(cutoff, device=scores.device, dtype=metric_dtype))
    return (agreement * depth_weights).sum(dim=-1) / depth_weights.sum()


def _top_weighted_pairwise_reward(
    scores: torch.Tensor,
    rank_labels: torch.Tensor,
    teacher_order: torch.Tensor,
    cutoff: int,
) -> torch.Tensor:
    """Return discounted agreement with all teacher-ordered pairs."""
    batch_size, rollout_count, slate_length = scores.shape
    metric_dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    better_positions, worse_positions = torch.triu_indices(
        slate_length,
        slate_length,
        offset=1,
        device=scores.device,
    )
    inside_cutoff = better_positions < cutoff
    better_positions = better_positions[inside_cutoff]
    worse_positions = worse_positions[inside_cutoff]
    better_indices = teacher_order[:, better_positions]
    worse_indices = teacher_order[:, worse_positions]
    pair_count = better_indices.size(-1)
    better_scores = scores.gather(
        dim=-1,
        index=better_indices.unsqueeze(1).expand(batch_size, rollout_count, pair_count),
    )
    worse_scores = scores.gather(
        dim=-1,
        index=worse_indices.unsqueeze(1).expand(batch_size, rollout_count, pair_count),
    )
    strict_preference = rank_labels.gather(
        dim=-1, index=better_indices
    ) > rank_labels.gather(dim=-1, index=worse_indices)
    pair_weights = 1.0 / torch.log2(better_positions.to(metric_dtype) + 2.0)
    pair_weights = pair_weights.unsqueeze(0) * strict_preference

    score_differences = better_scores - worse_scores
    concordance = (score_differences > 0).to(metric_dtype)
    concordance += 0.5 * (score_differences == 0).to(metric_dtype)
    normalizer = pair_weights.sum(dim=-1, keepdim=True)
    weighted = (pair_weights.unsqueeze(1) * concordance).sum(dim=-1)
    return torch.where(
        normalizer > 0,
        weighted / normalizer,
        torch.zeros_like(weighted),
    )


def _ndcg_reward(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    cutoff: int,
) -> torch.Tensor:
    topk_indices = scores.topk(k=cutoff, dim=-1).indices
    topk_relevance = relevance.gather(dim=-1, index=topk_indices)
    discounts = 1.0 / torch.log2(
        torch.arange(2, cutoff + 2, device=scores.device, dtype=scores.dtype)
    )
    dcg = (((2.0**topk_relevance) - 1.0) * discounts).sum(dim=-1)
    ideal_relevance = relevance.topk(k=cutoff, dim=-1).values
    idcg = (((2.0**ideal_relevance) - 1.0) * discounts).sum(dim=-1)
    return torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))


def _mrr_reward(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    original_relevance: torch.Tensor,
    cutoff: int,
    relevance_scheme: str | None,
) -> torch.Tensor:
    ranked_indices = scores.topk(k=cutoff, dim=-1).indices
    ranked_relevance = relevance.gather(dim=-1, index=ranked_indices)
    relevant_mask = resolve_relevant_mask(
        ranked_relevance=ranked_relevance,
        relevance_labels=original_relevance,
        relevance_scheme=relevance_scheme,
    )
    reciprocal_ranks = relevant_mask.to(scores.dtype) / torch.arange(
        1,
        cutoff + 1,
        device=scores.device,
        dtype=scores.dtype,
    )
    return reciprocal_ranks.max(dim=-1).values


def _contrastive_reward(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    reward_type: str,
    in_batch_positive_scores: torch.Tensor | None,
    use_in_batch_negatives: bool,
    temperature: float,
) -> torch.Tensor:
    batch_size, _, slate_length = scores.shape
    positive_indices = relevance_labels.argmax(dim=-1)
    positive_scores = scores[
        torch.arange(batch_size, device=scores.device), :, positive_indices
    ]
    negative_scores = scores.masked_fill(
        F.one_hot(positive_indices, num_classes=slate_length).bool().unsqueeze(1),
        float("-inf"),
    )
    if use_in_batch_negatives and in_batch_positive_scores is not None:
        negative_scores = torch.cat((negative_scores, in_batch_positive_scores), dim=-1)

    # Contrastive excludes the positive from the partition; InfoNCE includes it.
    partition_scores = (
        negative_scores
        if reward_type == "contrastive"
        else torch.cat((positive_scores.unsqueeze(-1), negative_scores), dim=-1)
    )
    return positive_scores - _temperature_scaled_logsumexp(
        partition_scores, temperature=temperature
    )


def compute_reward_from_scores(
    scores: torch.Tensor,
    relevance_labels: torch.Tensor,
    reward_type: str = "ndcg",
    k: int | None = 10,
    ndcg_in_batch_include_negatives: bool = False,
    contrastive_use_in_batch_negatives: bool = False,
    contrastive_temperature: float = 0.03,
    relevance_scheme: str | None = None,
    in_batch_positive_scores: torch.Tensor | None = None,
    in_batch_candidate_scores: torch.Tensor | None = None,
    rank_labels: torch.Tensor | None = None,
    rbo_p: float = 0.9,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if candidate_mask is not None:
        if (
            candidate_mask.shape != relevance_labels.shape
            or candidate_mask.dtype != torch.bool
        ):
            raise ValueError("candidate_mask must be bool [batch, candidates]")
        if not candidate_mask.any(dim=-1).all():
            raise ValueError("Each query needs valid candidates")
        if not candidate_mask.all():
            # Exact compaction also handles order-based rewards (RBO/pairwise) and k=None.
            # Preserve the batch's relevance convention when evaluating individual rows.
            scheme = relevance_scheme or (
                "graded" if (relevance_labels > 1).any() else "binary"
            )
            return torch.cat(
                [
                    compute_reward_from_scores(
                        scores=scores[i : i + 1][..., mask],
                        relevance_labels=relevance_labels[i : i + 1, mask],
                        reward_type=reward_type,
                        k=k,
                        ndcg_in_batch_include_negatives=ndcg_in_batch_include_negatives,
                        contrastive_use_in_batch_negatives=contrastive_use_in_batch_negatives,
                        contrastive_temperature=contrastive_temperature,
                        relevance_scheme=scheme,
                        in_batch_positive_scores=None
                        if in_batch_positive_scores is None
                        else in_batch_positive_scores[i : i + 1],
                        in_batch_candidate_scores=None
                        if in_batch_candidate_scores is None
                        else in_batch_candidate_scores[i : i + 1],
                        rank_labels=None
                        if rank_labels is None
                        else rank_labels[i : i + 1, mask],
                        rbo_p=rbo_p,
                    )
                    for i, mask in enumerate(candidate_mask)
                ],
                dim=0,
            )
    squeeze_rollout_dim = False
    if scores.dim() == 2:
        scores = scores.unsqueeze(1)
        squeeze_rollout_dim = True
        if in_batch_positive_scores is not None:
            in_batch_positive_scores = in_batch_positive_scores.unsqueeze(1)
        if in_batch_candidate_scores is not None:
            in_batch_candidate_scores = in_batch_candidate_scores.unsqueeze(1)

    reward_type = reward_type.lower()
    if reward_type not in SUPPORTED_REWARD_TYPES:
        raise ValueError(
            f"Unsupported reward type: {reward_type}. Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
        )
    if scores.dim() < 3:
        raise ValueError(
            "scores must be [batch, slate] or [batch, *rollout, slate], "
            f"got shape {tuple(scores.shape)}"
        )
    if relevance_labels.dim() != 2:
        raise ValueError(
            f"relevance_labels must be [batch, slate], got shape {tuple(relevance_labels.shape)}"
        )
    if (
        scores.shape[0] != relevance_labels.shape[0]
        or scores.shape[-1] != relevance_labels.shape[1]
    ):
        raise ValueError(
            "relevance_labels shape must match scores [batch, slate], "
            f"got scores={tuple(scores.shape)} labels={tuple(relevance_labels.shape)}"
        )
    if rank_labels is not None and rank_labels.shape != relevance_labels.shape:
        raise ValueError(
            "rank_labels shape must match relevance_labels [batch, slate], "
            f"got ranks={tuple(rank_labels.shape)} labels={tuple(relevance_labels.shape)}"
        )
    if not 0.0 <= rbo_p < 1.0:
        raise ValueError(f"RBO persistence p must lie in [0, 1), got {rbo_p}")

    batch_size = scores.size(0)
    rollout_shape = scores.shape[1:-1]
    slate_length = scores.size(-1)

    def finish(reward: torch.Tensor) -> torch.Tensor:
        """Fold the flattened rollout axis back into the caller's rollout shape."""
        reward = reward.reshape(batch_size, *rollout_shape)
        return reward.squeeze(1) if squeeze_rollout_dim else reward

    rollout_count = 1
    for rollout_dim in rollout_shape:
        rollout_count *= rollout_dim

    scores = scores.reshape(batch_size, rollout_count, slate_length)
    if in_batch_positive_scores is not None:
        in_batch_positive_scores = in_batch_positive_scores.reshape(
            batch_size, rollout_count, -1
        )
    if in_batch_candidate_scores is not None:
        in_batch_candidate_scores = in_batch_candidate_scores.reshape(
            batch_size, rollout_count, -1
        )

    expanded_labels = relevance_labels.unsqueeze(1).expand(
        batch_size, rollout_count, slate_length
    )
    if reward_type in {"top_weighted_pairwise", "rbo"}:
        if rank_labels is None:
            raise ValueError(f"rank_labels are required for the {reward_type} reward")
        if not torch.isfinite(rank_labels).all():
            raise ValueError("rank_labels must be finite")

        cutoff = slate_length if k is None else min(k, slate_length)
        if cutoff <= 0:
            return finish(scores.new_zeros(batch_size, rollout_count))

        teacher_order = rank_labels.argsort(dim=-1, descending=True, stable=True)
        if reward_type == "rbo":
            return finish(_rbo_reward(scores, teacher_order, cutoff, rbo_p))
        return finish(
            _top_weighted_pairwise_reward(scores, rank_labels, teacher_order, cutoff)
        )

    if reward_type in {"ndcg", "ndcg_in_batch", "mrr", "mrr_in_batch"}:
        extra_scores = None
        if reward_type.endswith("_in_batch"):
            extra_scores = (
                in_batch_candidate_scores
                if ndcg_in_batch_include_negatives
                else in_batch_positive_scores
            )
        ranking_scores, ranking_labels = _extend_ranking_pool(
            scores, expanded_labels, extra_scores
        )
        cutoff = (
            ranking_scores.size(-1) if k is None else min(k, ranking_scores.size(-1))
        )
        if cutoff <= 0:
            return finish(scores.new_zeros(batch_size, rollout_count))
        if reward_type.startswith("ndcg"):
            return finish(_ndcg_reward(ranking_scores, ranking_labels, cutoff))
        return finish(
            _mrr_reward(
                ranking_scores,
                ranking_labels,
                relevance_labels,
                cutoff,
                relevance_scheme,
            )
        )
    return finish(
        _contrastive_reward(
            scores,
            relevance_labels,
            reward_type,
            in_batch_positive_scores,
            contrastive_use_in_batch_negatives,
            contrastive_temperature,
        )
    )


@dataclass(frozen=True)
class RewardEvaluator:
    """Evaluate one reward configuration behind a policy-independent boundary.

    GRPO supplies score tensors through :class:`RewardInputs` and receives
    generic :class:`RewardSignal` objects.  Term dispatch, candidate-pool needs,
    and raw weighting stay on this side of the boundary. Advantage normalization
    and combination remain policy estimator choices.
    """

    terms: tuple[RewardTerm, ...]
    boundary_cutoff: int | None = None

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        if not terms:
            raise ValueError("RewardEvaluator requires at least one reward term")
        if self.boundary_cutoff is not None and self.boundary_cutoff <= 0:
            raise ValueError("boundary_cutoff must be positive when provided")
        object.__setattr__(self, "terms", terms)

    @property
    def requirements(self) -> RewardRequirements:
        return RewardRequirements(
            representative_candidates=reward_terms_need_in_batch_positives(self.terms),
            all_candidates=reward_terms_need_in_batch_candidates(self.terms),
        )

    def evaluate(self, inputs: RewardInputs) -> RewardEvaluation:
        """Evaluate configured terms and expose their policy-facing signals."""
        if inputs.fixed_cross_scores is not None:
            term_rewards = compute_reward_terms_over_fixed_pool(
                self.terms,
                scores=inputs.scores,
                relevance_labels=inputs.relevance_labels,
                candidate_mask=inputs.candidate_mask,
                rank_labels=inputs.rank_labels,
                cross_scores=inputs.fixed_cross_scores,
            )
        else:
            term_rewards = compute_reward_terms(
                self.terms,
                scores=inputs.scores,
                relevance_labels=inputs.relevance_labels,
                candidate_mask=inputs.candidate_mask,
                rank_labels=inputs.rank_labels,
                in_batch_positive_scores=inputs.in_batch_positive_scores,
                in_batch_candidate_scores=inputs.in_batch_candidate_scores,
            )
        return self.from_term_rewards(term_rewards)

    def from_term_rewards(
        self, term_rewards: Mapping[str, torch.Tensor]
    ) -> RewardEvaluation:
        """Assemble chunked or specialized term values using one combine rule."""
        missing = [term.name for term in self.terms if term.name not in term_rewards]
        if missing:
            raise ValueError(f"Missing reward term values: {', '.join(missing)}")

        values = {term.name: term_rewards[term.name].float() for term in self.terms}
        contributions = [values[term.name] * term.weight for term in self.terms]
        combined = contributions[0]
        for contribution in contributions[1:]:
            combined = combined + contribution

        signals = tuple(
            RewardSignal(term.name, term.weight, values[term.name])
            for term in self.terms
        )
        return RewardEvaluation(values, combined, signals)

    @torch.no_grad()
    def counterfactual_document_advantages(
        self,
        inputs: RewardInputs,
        *,
        reference_scores: torch.Tensor,
        rewards: torch.Tensor,
        document_axis: int,
    ) -> torch.Tensor:
        """Re-evaluate rewards after replacing each document action in turn."""
        if inputs.fixed_cross_scores is not None:
            raise ValueError(
                "Counterfactual document baselines do not support fixed cross pools"
            )
        other_dims = tuple(
            dim for dim in range(1, rewards.ndim) if dim != document_axis
        )
        marginal = rewards.mean(dim=other_dims) if other_dims else rewards
        result = inputs.scores.new_zeros(
            (*marginal.shape, inputs.scores.size(-1)), dtype=torch.float32
        )
        replaced_scores = inputs.scores.clone()
        for document_index in range(inputs.scores.size(-1)):
            replaced_scores[..., document_index] = reference_scores[..., document_index]
            baseline = self.evaluate(replace(inputs, scores=replaced_scores)).combined
            delta = rewards - baseline
            result[..., document_index] = (
                delta.mean(dim=other_dims) if other_dims else delta
            )
            replaced_scores[..., document_index] = inputs.scores[..., document_index]
        if inputs.candidate_mask is not None:
            result.masked_fill_(~inputs.candidate_mask.unsqueeze(1), 0.0)
        return result

    def cross_candidate_mask(
        self,
        all_candidates: torch.Tensor,
        representative_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Select the cross-query candidate union required by all terms."""
        requirements = self.requirements
        selected = torch.zeros_like(all_candidates)
        if requirements.all_candidates:
            selected |= all_candidates
        if requirements.representative_candidates:
            selected |= representative_candidates
        return selected

    @staticmethod
    def distinct_levels(values: torch.Tensor) -> torch.Tensor:
        return realized_reward_levels(values)

    def statistics(self, evaluation: RewardEvaluation) -> dict[str, torch.Tensor]:
        """Return reward-only diagnostics without rollout or estimator details."""
        flattened = evaluation.combined.detach().reshape(-1).float()
        statistics = {
            "reward_mean": flattened.mean(),
            "reward_std": flattened.std(unbiased=False),
            "reward_min": flattened.min(),
            "reward_max": flattened.max(),
        }
        batch_size = evaluation.combined.size(0)
        for term in self.terms:
            values = evaluation.term_rewards[term.name].detach()
            prefix = f"reward/{term.name}"
            statistics[f"{prefix}/n_distinct"] = realized_reward_levels(values)
            if len(self.terms) > 1:
                flattened_term = values.reshape(-1).float()
                statistics.update(
                    {
                        f"{prefix}/mean": flattened_term.mean(),
                        f"{prefix}/std": flattened_term.std(unbiased=False),
                        f"{prefix}/min": flattened_term.min(),
                        f"{prefix}/max": flattened_term.max(),
                        f"{prefix}/group_std": values.reshape(batch_size, -1)
                        .std(dim=-1, unbiased=False)
                        .mean(),
                    }
                )
        return statistics

    @torch.no_grad()
    def exploration_statistics(
        self,
        mean_scores: torch.Tensor,
        sampled_scores: torch.Tensor,
        relevance_labels: torch.Tensor,
        candidate_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Measure how often exploration flips reward-visible score boundaries."""
        valid = (
            torch.ones_like(relevance_labels, dtype=torch.bool)
            if candidate_mask is None
            else candidate_mask.bool()
        )
        order = mean_scores.masked_fill(~valid, -torch.inf).argsort(
            dim=-1, descending=True
        )
        grades = relevance_labels.gather(1, order)
        ordered_valid = valid.gather(1, order)
        ordered_mean = mean_scores.gather(1, order)
        boundary_count = max(mean_scores.size(1) - 1, 0)
        cutoff = min(self.boundary_cutoff or boundary_count, boundary_count)
        eligible = (
            ordered_valid[:, :-1]
            & ordered_valid[:, 1:]
            & (grades[:, :-1] != grades[:, 1:])
            & ((ordered_mean[:, :-1] - ordered_mean[:, 1:]).abs() > 1e-8)
        )
        eligible[:, cutoff:] = False
        sampled = sampled_scores.reshape(mean_scores.size(0), -1, mean_scores.size(1))
        ordered = sampled.gather(-1, order.unsqueeze(1).expand_as(sampled))
        flips = ordered[:, :, :-1] < ordered[:, :, 1:]
        count = eligible.sum()
        return {
            "exploration/own_boundary_pairs": count.float() / mean_scores.size(0),
            "exploration/own_boundary_flip_rate": (
                (flips & eligible.unsqueeze(1)).sum().float()
                / (count * sampled.size(1)).clamp_min(1)
            ),
        }
