"""RELER's shared baselines and dimension-aware, externally scheduled vMF exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import torch
import torch.nn.functional as F


class PolicyObjectiveResult(NamedTuple):
    """Common result contract for every GRPO policy-objective path."""

    loss: torch.Tensor
    statistics: dict[str, torch.Tensor]
    advantages: torch.Tensor
    degenerate_fraction: torch.Tensor


def bessel_ratio(
    nu: float,
    kappa: torch.Tensor,
    num_iters: int | None = None,
) -> torch.Tensor:
    """Return ``I_nu(kappa) / I_(nu-1)(kappa)`` via backward recurrence."""
    kappa = torch.as_tensor(kappa).double()
    if num_iters is None:
        num_iters = max(32, int(float(kappa.detach().max()) / 2.0 - nu) + 64)
    tail_order = nu + num_iters
    ratio = kappa / (tail_order + torch.sqrt(kappa.pow(2) + tail_order**2))
    for step in range(num_iters - 1, -1, -1):
        order = nu + step
        ratio = 1.0 / (2.0 * order / kappa + ratio)
    return ratio.float()


def sample_vmf(
    mean_directions: torch.Tensor,
    kappa: float,
    num_samples: int,
    max_rejection_rounds: int = 256,
) -> torch.Tensor:
    """Draw exact vMF samples with Wood's rejection sampler."""
    if mean_directions.dim() != 2:
        raise ValueError(
            f"mean_directions must be [n, d], got shape {tuple(mean_directions.shape)}"
        )
    kappa = float(kappa)
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    n, dim = mean_directions.shape
    if dim < 3:
        raise ValueError(f"sample_vmf requires embedding dim >= 3, got {dim}")
    device = mean_directions.device

    # Rationalizing the envelope parameter avoids cancellation at large kappa.
    b = (dim - 1) / (2.0 * kappa + math.sqrt(4.0 * kappa**2 + (dim - 1) ** 2))
    x0 = (1.0 - b) / (1.0 + b)
    log_c = kappa * x0 + (dim - 1) * math.log(1.0 - x0 * x0)

    total = n * num_samples
    w = torch.empty(total, dtype=torch.float64, device=device)
    pending = torch.ones(total, dtype=torch.bool, device=device)
    half = torch.tensor((dim - 1) / 2.0, dtype=torch.float64, device=device)
    beta = torch.distributions.Beta(half, half)
    for _ in range(max_rejection_rounds):
        num_pending = int(pending.sum())
        if num_pending == 0:
            break
        z = beta.sample((num_pending,))
        candidate = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        uniform = torch.rand(num_pending, dtype=torch.float64, device=device)
        accept = kappa * candidate + (dim - 1) * torch.log1p(
            -x0 * candidate
        ) - log_c >= torch.log(uniform)
        accepted_indices = pending.nonzero(as_tuple=True)[0][accept]
        w[accepted_indices] = candidate[accept]
        pending[accepted_indices] = False
    if pending.any():
        raise RuntimeError("sample_vmf rejection sampling did not converge")

    means = F.normalize(mean_directions.float(), dim=-1)
    tangent = torch.randn(
        n,
        num_samples,
        dim,
        dtype=torch.float32,
        device=device,
    )
    tangent = tangent - (tangent * means.unsqueeze(1)).sum(
        dim=-1, keepdim=True
    ) * means.unsqueeze(1)
    tangent = F.normalize(tangent, dim=-1)
    w = w.reshape(n, num_samples, 1).to(torch.float32)
    samples = (
        w * means.unsqueeze(1) + torch.sqrt((1.0 - w.pow(2)).clamp_min(0.0)) * tangent
    )
    return samples.to(mean_directions.dtype)


def sample_projected_gaussian(
    mean_directions: torch.Tensor,
    sigma: float,
    num_samples: int,
) -> torch.Tensor:
    """Draw ``normalize(mu + sigma * eps)`` samples for the Gaussian ablation."""
    if mean_directions.dim() != 2:
        raise ValueError(
            f"mean_directions must be [n, d], got shape {tuple(mean_directions.shape)}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    means = F.normalize(mean_directions.float(), dim=-1)
    n, dim = means.shape
    noise = torch.randn(
        n,
        num_samples,
        dim,
        dtype=torch.float32,
        device=means.device,
    )
    samples = F.normalize(means.unsqueeze(1) + float(sigma) * noise, dim=-1)
    return samples.to(mean_directions.dtype)


def projected_gaussian_mean_alignment(sigma: float, dim: int) -> float:
    """Approximate ``E[mu^T normalize(mu + sigma * eps)]``."""
    return 1.0 / math.sqrt(1.0 + float(sigma) ** 2 * int(dim))


def vmf_kl(
    policy_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    kappa: torch.Tensor,
    document_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL between vMF distributions with equal concentration."""
    reference_embeddings = F.normalize(reference_embeddings.float(), dim=-1).detach()
    alignment = (policy_embeddings.float() * reference_embeddings).sum(dim=-1)
    kappa = kappa.float()
    mean_resultant_length = bessel_ratio(policy_embeddings.size(-1) / 2.0, kappa)
    distances = 1.0 - alignment
    if document_mask is not None:
        distances = (distances * document_mask).sum(-1) / document_mask.sum(
            -1
        ).clamp_min(1)
    return kappa * mean_resultant_length * distances.mean()


@lru_cache(maxsize=4096)
def mean_alignment(dim: int, kappa: float) -> float:
    """I_(d/2)(kappa) / I_(d/2-1)(kappa), via a convergent continued fraction."""
    if dim < 3 or not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("vMF requires dimension >= 3 and finite positive kappa")
    value = c = float(dim)
    inverse = 0.0
    for j in range(1, 10001):
        b = dim + 2.0 * j
        inverse = 1.0 / (b + kappa * kappa * inverse)
        c = b + kappa * kappa / c
        factor = c * inverse
        value *= factor
        if abs(factor - 1.0) < 2e-15:
            return kappa / value
    raise RuntimeError("vMF mean-alignment continued fraction did not converge")


@lru_cache(maxsize=4096)
def kappa_for_alignment(dim: int, alignment: float) -> float:
    if dim < 3 or not math.isfinite(alignment) or not 0 < alignment < 1:
        raise ValueError(
            "dimension >= 3 and target alignment strictly between 0 and 1 required"
        )
    low, high = 0.0, max(1.0, dim * alignment / (1.0 - alignment**2))
    while mean_alignment(dim, high) < alignment:
        high *= 2.0
    for _ in range(48):
        middle = (low + high) / 2.0
        if mean_alignment(dim, middle) < alignment:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def score_gap_variance(dim: int, kappa: float, gap, perpendicular_norm):
    """Exact vMF projected-score variance, including longitudinal fluctuations."""
    a = mean_alignment(dim, float(kappa))
    derivative = max(0.0, 1.0 - a * a - (dim - 1) * a / kappa)
    return (a / kappa) * perpendicular_norm**2 + derivative * gap**2


def validate_exploration(target_alignment, final_alignment, schedule):
    if schedule not in {"fixed", "linear"}:
        raise ValueError("exploration_schedule must be fixed or linear")
    for value in (target_alignment, final_alignment):
        if value is not None and (not math.isfinite(value) or not 0 < value < 1):
            raise ValueError(
                "exploration alignments must be finite and strictly between 0 and 1"
            )
    if schedule == "linear":
        if target_alignment is None or final_alignment is None:
            raise ValueError(
                "linear exploration requires target_alignment and final_alignment"
            )
        if final_alignment < target_alignment:
            raise ValueError("linear exploration must maintain or increase alignment")
    elif final_alignment is not None:
        raise ValueError("final_alignment requires a linear exploration schedule")


class ExplorationSchedule:
    """A predetermined schedule indexed by completed optimizer steps, never reward feedback.

    target_alignment overrides kappa; linear interpolation ends at the last update.
    HF Trainer restores global_step/max_steps before the first resumed forward pass.
    """

    def __init__(
        self, kappa=755.0, target_alignment=None, final_alignment=None, schedule="fixed"
    ):
        validate_exploration(target_alignment, final_alignment, schedule)
        if not math.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        self.kappa = float(kappa)
        self.target_alignment = target_alignment
        self.final_alignment = final_alignment
        self.schedule = schedule
        self.step = 0
        self.total_steps = 0

    def set_step(self, step: int, total_steps: int):
        if step < 0 or total_steps < 0:
            raise ValueError("training progress cannot be negative")
        self.step, self.total_steps = int(step), int(total_steps)

    def resolve(self, dim: int) -> float:
        if self.target_alignment is None:
            return self.kappa
        alignment = self.target_alignment
        if self.schedule == "linear":
            if self.total_steps <= 0:
                raise ValueError(
                    "linear exploration requires total optimizer steps before sampling"
                )
            progress = min(self.step / max(self.total_steps - 1, 1), 1.0)
            alignment += progress * (self.final_alignment - alignment)
        return kappa_for_alignment(dim, float(alignment))

    def state_dict(self):
        return dict(
            kappa=self.kappa,
            target_alignment=self.target_alignment,
            final_alignment=self.final_alignment,
            schedule=self.schedule,
            step=self.step,
            total_steps=self.total_steps,
        )

    def load_state_dict(self, state):
        for key in ("kappa", "target_alignment", "final_alignment", "schedule"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Checkpoint exploration setting differs: {key}")
        self.set_step(state["step"], state["total_steps"])


def group_advantages(
    rewards,
    baseline="group",
    normalization="none",
    shared_std=None,
    external_baseline=None,
):
    """Center without hiding degeneracy when normalization is disabled.

    LOO uses the same opposite-role samples for every marginal. Only unnormalized
    group/LOO estimators have the stated constant-factor/unbiased guarantees.
    """
    rewards = rewards.float()
    if rewards.ndim != 2 or rewards.size(1) < 2:
        raise ValueError("Rewards must have shape [batch, group >= 2]")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    spread = centered.std(dim=1, keepdim=True, unbiased=False)
    tolerance = 1e-4 * rewards.abs().mean(dim=1, keepdim=True).clamp_min(1.0)
    degenerate = spread <= tolerance
    if baseline == "leave_one_out":
        advantages = centered * (rewards.size(1) / (rewards.size(1) - 1))
    elif baseline == "group":
        advantages = centered
    elif baseline == "ema" and external_baseline is not None:
        advantages = rewards - external_baseline
    else:
        raise ValueError("Unsupported or missing advantage baseline")
    if normalization == "none":
        return advantages, degenerate.squeeze(1)
    if normalization not in {"per_component", "shared"}:
        raise ValueError("Unsupported advantage normalization")
    # Preserve legacy group scaling; LOO changes only the baseline, not this divisor.
    divisor = (
        shared_std if normalization == "shared" and shared_std is not None else spread
    )
    if baseline == "ema":
        advantages = advantages / divisor.clamp_min(tolerance)
    else:
        advantages = torch.where(
            divisor > tolerance,
            advantages / divisor.clamp_min(tolerance),
            torch.zeros_like(advantages),
        )
    return advantages, degenerate.squeeze(1)


def tensor_statistics(
    values: torch.Tensor,
    prefix: str,
    separator: str = "_",
) -> dict[str, torch.Tensor]:
    """Detached scalar moments with stable metric names."""
    flattened = values.detach().reshape(-1).float()
    return {
        f"{prefix}{separator}mean": flattened.mean(),
        f"{prefix}{separator}std": flattened.std(unbiased=False),
        f"{prefix}{separator}min": flattened.min(),
        f"{prefix}{separator}max": flattened.max(),
    }


@dataclass(frozen=True)
class FactorizedAdvantages:
    """Leave-one-out credit for a [batch, query draw, document draw] reward."""

    query: torch.Tensor
    documents: torch.Tensor
    flattened: torch.Tensor
    degenerate_fraction: torch.Tensor
    statistics: dict[str, torch.Tensor]


def factorized_leave_one_out(
    rewards: torch.Tensor,
    term_rewards: dict[str, torch.Tensor],
) -> FactorizedAdvantages:
    """Compute the fixed LOO contract shared by shortlist and cross-query policies."""
    if rewards.ndim != 3:
        raise ValueError(
            "Factorized rewards must be [batch, query draw, document draw]"
        )

    advantages = {}
    degenerate = []
    statistics = {}
    for marginal_axis, name in ((2, "query"), (1, "documents")):
        marginal = rewards.mean(marginal_axis)
        advantage, collapsed = group_advantages(
            marginal, baseline="leave_one_out", normalization="none"
        )
        advantages[name] = advantage
        degenerate.append(collapsed)
        for term_name, values in term_rewards.items():
            term_marginal = values.mean(marginal_axis)
            _, term_collapsed = group_advantages(
                term_marginal, baseline="leave_one_out", normalization="none"
            )
            prefix = f"reward/{term_name}/{name}"
            statistics[f"{prefix}/group_std"] = term_marginal.std(
                -1, unbiased=False
            ).mean()
            statistics[f"{prefix}/degenerate_frac"] = term_collapsed.float().mean()

    query = advantages["query"]
    documents = advantages["documents"]
    return FactorizedAdvantages(
        query=query,
        documents=documents,
        flattened=torch.cat((query, documents), dim=1),
        degenerate_fraction=torch.cat(degenerate).float().mean(),
        statistics=statistics,
    )
