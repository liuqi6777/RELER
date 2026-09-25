from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from reler.config.arguments import RLArguments
from reler.data.embedding import build_slate_inputs
from reler.data.protocol import encode_valid_candidates, pool_embeddings
from reler.objectives.contrastive import (
    aux_infonce_contract,
    auxiliary_infonce_loss,
    cross_query_document_pool,
)
from reler.objectives.cross_query import (
    shared_document_policy_objective,
)
from reler.objectives.policy_math import (
    ExplorationSchedule,
    PolicyObjectiveResult,
    bessel_ratio,
    group_advantages,
    mean_alignment,
    projected_gaussian_mean_alignment,
    sample_projected_gaussian,
    sample_vmf,
    tensor_statistics,
    vmf_kl,
)
from reler.objectives.precision import fp32_scores
from reler.objectives.projection import (
    conditional_projection_loss,
    fixed_projection_candidates,
)
from reler.objectives.rewards import (
    RewardEvaluation,
    RewardEvaluator,
    RewardInputs,
)
from reler.objectives.rollout_rng import RolloutRNG
from reler.objectives.shortlists import (
    ShortlistSettings,
    shortlist_contract,
    shortlist_policy_objective,
)


@dataclass
class _ActionComponent:
    role: str
    rollout_embeddings: torch.Tensor
    policy_embeddings: torch.Tensor | None = None
    sampled_embeddings: torch.Tensor | None = None
    kappa: torch.Tensor | None = None
    # Metric-key name ("query", "documents", "positive", "negative"). Derived from the
    # action_components config, so it is identical on every rank -- a requirement of the
    # cross-rank metric reduce, which walks accumulator keys in insertion order.
    name: str = ""
    document_mask: torch.Tensor | None = None

    @property
    def is_active(self) -> bool:
        return self.sampled_embeddings is not None


@dataclass(frozen=True)
class _RolloutLayout:
    """Tensor-axis contract for one ordinary product or diagonal rollout."""

    query: _ActionComponent
    documents: tuple[_ActionComponent, ...]
    active: tuple[_ActionComponent, ...]
    log_probs: tuple[torch.Tensor, ...]
    active_axis_by_id: dict[int, int]
    diagonal: bool
    num_axes: int
    group_size: int
    frozen_document_scale: torch.Tensor | None


@dataclass(frozen=True)
class _ScoreContext:
    """Every candidate score table needed to evaluate a reward configuration."""

    own: torch.Tensor
    in_batch_positives: torch.Tensor | None
    in_batch_candidates: torch.Tensor | None
    fixed_cross_scores: torch.Tensor | None
    fixed_cross_pool: torch.Tensor | None
    fixed_cross_mask: torch.Tensor | None
    loss_weight: float

    def reward_inputs(
        self,
        relevance_labels: torch.Tensor,
        rank_labels: torch.Tensor | None,
        candidate_mask: torch.Tensor | None,
    ) -> RewardInputs:
        """Expose only reward-owned inputs at the policy/reward boundary."""
        return RewardInputs(
            scores=self.own,
            relevance_labels=relevance_labels,
            candidate_mask=candidate_mask,
            rank_labels=rank_labels,
            in_batch_positive_scores=self.in_batch_positives,
            in_batch_candidate_scores=self.in_batch_candidates,
            fixed_cross_scores=self.fixed_cross_scores,
        )


@dataclass(frozen=True)
class _AdvantageInput:
    name: str
    weight: float
    rewards: torch.Tensor
    shared_std: torch.Tensor | None


@dataclass(frozen=True)
class _SurrogateResult:
    loss: torch.Tensor
    advantages: torch.Tensor
    degenerate_fraction: torch.Tensor
    statistics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class _PolicyInputs:
    query: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    documents: torch.Tensor
    policy_query: torch.Tensor | None
    policy_positive: torch.Tensor | None
    policy_negative: torch.Tensor | None
    candidate_mask: torch.Tensor


def pool_last_token_embedding(
    last_hidden_states: Tensor,
    attention_mask: Tensor,
    normalize: bool = True,
) -> Tensor:
    """Backward-compatible last-token helper used by older scripts."""
    return pool_embeddings(
        last_hidden_states,
        attention_mask,
        pooling_method="last",
        normalize=normalize,
    )


@dataclass
class GRPOModelOutput(ModelOutput):
    loss: Optional[Tensor] = None
    reward: Optional[Tensor] = None
    reward_mean: Optional[Tensor] = None
    reward_std: Optional[Tensor] = None
    reward_min: Optional[Tensor] = None
    reward_max: Optional[Tensor] = None
    advantages_mean: Optional[Tensor] = None
    advantages_std: Optional[Tensor] = None
    advantages_min: Optional[Tensor] = None
    advantages_max: Optional[Tensor] = None
    advantages_degenerate_frac: Optional[Tensor] = None
    sigma: Optional[Tensor] = None
    kl: Optional[Tensor] = None
    # Per-reward-term scalars, already namespaced ("reward/<term>/mean"). Config-driven, so the
    # key set is identical on every rank -- which the trainer's cross-rank reduce relies on.
    reward_terms: Optional[Dict[str, Tensor]] = None


class GRPO(nn.Module):
    def __init__(
        self,
        action_components=None,
        *,
        config: RLArguments | None = None,
        **overrides,
    ):
        """Build the policy from the single normalized RL configuration contract.

        Production code passes ``RLArguments`` directly. Keyword overrides remain
        supported for focused tests and small programmatic uses; they are normalized
        by the same dataclass, so validation cannot drift between CLI and model paths.
        The first positional argument remains the legacy ``action_components`` value.
        """
        super().__init__()
        if isinstance(action_components, RLArguments):
            if config is not None:
                raise TypeError("Policy configuration was provided twice")
            config = action_components
            action_components = None
        if action_components is not None:
            overrides["action_components"] = action_components
        if config is not None and overrides:
            raise TypeError(
                "Pass either config=RLArguments(...) or keyword overrides, not both"
            )
        if config is None:
            config = RLArguments(**overrides).resolve()
        elif isinstance(config, RLArguments):
            config = config.resolve()
        else:
            raise TypeError("config must be an RLArguments instance")

        self.config = config
        self.reward_evaluator = RewardEvaluator(
            config.reward_terms, boundary_cutoff=config.reward_ndcg_k
        )
        self.rollout_rng = RolloutRNG(config.rollout_seed)
        self.exploration = ExplorationSchedule(
            self.kappa if self.kappa is not None else self.sigma**-2,
            self.target_alignment,
            self.final_alignment,
            self.exploration_schedule,
        )

        if self.sigma_learnable:
            self.log_sigma = nn.Parameter(
                torch.log(torch.tensor(float(self.sigma), dtype=torch.float32))
            )
        else:
            self.register_buffer(
                "fixed_sigma", torch.tensor(float(self.sigma), dtype=torch.float32)
            )

        if self.advantage_baseline == "ema":
            self.register_buffer(
                "reward_baseline", torch.zeros((), dtype=torch.float32)
            )
            self.register_buffer(
                "reward_baseline_initialized", torch.zeros((), dtype=torch.bool)
            )

    def __getattr__(self, name: str):
        """Read algorithm settings from the immutable spec unless explicitly overridden."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            config = self.__dict__.get("config")
            if config is not None and name in config.__dataclass_fields__:
                return getattr(config, name)
            raise

    def current_sigma(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if self.sigma_learnable:
            # Hard bounds: REINFORCE on the exploration scale has no restoring force (the
            # normalizer term cancels under centered advantages), so an unbounded sigma can
            # collapse exploration entirely.
            log_sigma = self.log_sigma.clamp(
                math.log(self.sigma_min), math.log(self.sigma_max)
            )
            sigma = torch.exp(log_sigma)
        else:
            sigma = self.fixed_sigma
        return sigma.to(device=device, dtype=dtype)

    summarize_tensor = staticmethod(tensor_statistics)

    def _current_reward_baseline(self, component_rewards: torch.Tensor) -> torch.Tensor:
        """Global running baseline for the REINFORCE ablation (advantage_baseline='ema').

        Returns the baseline to subtract *before* updating it, so the baseline is
        independent of the rewards it baselines and the estimator stays unbiased.
        The update uses the local-rank batch mean; ranks therefore hold slightly
        different baselines, which is the usual REINFORCE practice and is why this
        exists only as a comparison point for the group baseline.
        """
        batch_mean = component_rewards.detach().mean()
        if not bool(self.reward_baseline_initialized):
            baseline = self.reward_baseline.clone()
            if self.training:
                self.reward_baseline.copy_(batch_mean)
                self.reward_baseline_initialized.fill_(True)
            return baseline

        baseline = self.reward_baseline.clone()
        if self.training:
            momentum = self.advantage_baseline_momentum
            self.reward_baseline.mul_(momentum).add_(batch_mean, alpha=1.0 - momentum)
        return baseline

    def _compute_advantages(
        self,
        component_rewards: torch.Tensor,
        shared_std: torch.Tensor | None = None,
        external_baseline: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        baseline = external_baseline
        if self.advantage_baseline == "ema" and baseline is None:
            baseline = self._current_reward_baseline(component_rewards.float())
        return group_advantages(
            component_rewards,
            self.advantage_baseline,
            self.advantage_norm,
            shared_std,
            baseline,
        )

    @property
    def rollout_seed(self):
        return self.rollout_rng.seed

    @property
    def sample_query(self) -> bool:
        return ("query",) in self.action_components

    @property
    def sample_positive(self) -> bool:
        return any("positive" in group for group in self.action_components)

    @property
    def sample_negative(self) -> bool:
        return any("negative" in group for group in self.action_components)

    def set_training_progress(self, step: int, total_steps: int) -> None:
        """Synchronize exploration with completed optimizer steps."""
        self.exploration.set_step(step, total_steps)

    def rollout_rng_contract(self, world_size: int | None = None) -> dict | None:
        if self.rollout_seed is None:
            return None
        if world_size is None:
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
        return {
            "seed": self.rollout_seed,
            "version": self.rollout_rng.version,
            "world_size": world_size,
        }

    @property
    def requires_checkpoint_contract(self) -> bool:
        """Whether resuming without a policy contract could change the estimator."""
        return any(
            (
                self.exploration.target_alignment is not None,
                self.advantage_baseline == "leave_one_out",
                self.rollout_seed is not None,
                aux_infonce_contract(self) is not None,
                self.reward_cross_device_negatives,
                shortlist_contract(self) is not None,
                self.cross_query_document_gradients,
            )
        )

    def checkpoint_contract(self, step: int, total_steps: int) -> dict:
        """Serialize the estimator choices that must stay fixed across resume."""
        exploration = self.exploration.state_dict()
        exploration.update(step=step, total_steps=total_steps)
        estimator = {
            key: getattr(self, key)
            for key in (
                "advantage_baseline",
                "advantage_norm",
                "document_log_prob_reduction",
                "group_size",
            )
        }
        estimator["gradient_estimator"] = self.gradient_estimator
        if self.document_advantage_baseline != "shared":
            estimator["document_advantage_baseline"] = self.document_advantage_baseline

        payload = {
            "config": self.config.as_checkpoint_dict(self),
            "exploration": exploration,
            "estimator": estimator,
        }
        optional_contracts = {
            "reward_cross_device_negatives": (
                True if self.reward_cross_device_negatives else None
            ),
            "cross_query_document_gradients": (
                True if self.cross_query_document_gradients else None
            ),
            "reward_shortlists": shortlist_contract(self),
            "aux_infonce": aux_infonce_contract(self),
            "rollout_rng": self.rollout_rng_contract(),
        }
        payload.update(
            {
                name: value
                for name, value in optional_contracts.items()
                if value is not None
            }
        )
        if self.advantage_baseline == "ema":
            payload.update(
                reward_baseline=float(self.reward_baseline),
                reward_baseline_initialized=bool(self.reward_baseline_initialized),
            )
        return payload

    def load_checkpoint_contract(self, payload: dict) -> None:
        """Validate and restore policy state without depending on trainer internals."""
        saved_config = payload.get("config")
        if saved_config is not None:
            current_config = self.config.as_checkpoint_dict(self)
            changed = sorted(
                key
                for key in set(saved_config) | set(current_config)
                if saved_config.get(key) != current_config.get(key)
            )
            if changed:
                raise ValueError(
                    "Checkpoint GRPO configuration differs: "
                    f"{', '.join(changed)}; start a new run"
                )

        comparisons = {
            "reward_cross_device_negatives": self.reward_cross_device_negatives,
            "cross_query_document_gradients": self.cross_query_document_gradients,
            "aux_infonce": aux_infonce_contract(self),
            "reward_shortlists": shortlist_contract(self),
            "rollout_rng": self.rollout_rng_contract(),
        }
        errors = {
            "reward_cross_device_negatives": "reward candidate pool",
            "cross_query_document_gradients": "cross-query document policy",
            "aux_infonce": "auxiliary InfoNCE objective",
            "reward_shortlists": "reward shortlist sampling",
            "rollout_rng": "rollout RNG seed/version/world size",
        }
        for key, expected in comparisons.items():
            if (
                payload.get(key, False if isinstance(expected, bool) else None)
                != expected
            ):
                raise ValueError(f"Checkpoint {errors[key]} differs; start a new run")

        estimator = payload["estimator"]
        saved_document_baseline = estimator.get("document_advantage_baseline", "shared")
        if saved_document_baseline != self.document_advantage_baseline:
            raise ValueError(
                "Checkpoint estimator differs: document_advantage_baseline; "
                "start a new run"
            )
        if (
            estimator.get("gradient_estimator", "score_function")
            != self.gradient_estimator
        ):
            raise ValueError(
                "Checkpoint estimator differs: gradient_estimator; start a new run"
            )
        for key, value in estimator.items():
            if getattr(self, key, None) != value:
                raise ValueError(
                    f"Checkpoint estimator differs: {key}; start a new run"
                )

        self.exploration.load_state_dict(payload["exploration"])
        if self.advantage_baseline == "ema":
            self.reward_baseline.fill_(payload["reward_baseline"])
            self.reward_baseline_initialized.fill_(
                payload["reward_baseline_initialized"]
            )

    def runtime_state(self, sigma: float | None = None) -> dict:
        """Return mutable GRPO state omitted from the wrapped backbone weights."""
        state = {}
        if self.sigma_learnable and sigma is not None:
            state["sigma"] = sigma
        if self.advantage_baseline == "ema":
            state.update(
                reward_baseline=float(self.reward_baseline),
                reward_baseline_initialized=bool(self.reward_baseline_initialized),
            )
        return state

    def load_runtime_state(self, state: dict) -> None:
        """Restore mutable state before parameter partitioning is initialized."""
        with torch.no_grad():
            if self.sigma_learnable and state.get("sigma") is not None:
                self.log_sigma.fill_(math.log(float(state["sigma"])))
            if self.advantage_baseline == "ema" and "reward_baseline" in state:
                self.reward_baseline.fill_(float(state["reward_baseline"]))
                self.reward_baseline_initialized.fill_(
                    bool(state.get("reward_baseline_initialized", True))
                )

    def _draw(self, mean_directions: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        with self.rollout_rng.draw(
            mean_directions.device, self.exploration.step, self.training
        ):
            return self._draw_actions(mean_directions, kappa)

    def _draw_actions(
        self, mean_directions: torch.Tensor, kappa: torch.Tensor
    ) -> torch.Tensor:
        if self.sampling_law == "gaussian":
            return sample_projected_gaussian(
                mean_directions,
                sigma=float(kappa.detach()) ** -0.5,
                num_samples=self.group_size,
            )
        return sample_vmf(
            mean_directions,
            kappa=float(kappa.detach()),
            num_samples=self.group_size,
        )

    def _sample_document_embeddings(
        self, rollout_document_embeddings, kappa, document_mask=None
    ):
        batch_size, slate_length, dim = rollout_document_embeddings.shape
        if document_mask is None:
            document_mask = torch.ones(
                (batch_size, slate_length),
                device=rollout_document_embeddings.device,
                dtype=torch.bool,
            )
        flat_mask = document_mask.reshape(-1)
        valid = rollout_document_embeddings.detach().reshape(-1, dim)[flat_mask]
        samples = self._draw(valid, kappa)
        full = samples.new_zeros(batch_size * slate_length, self.group_size, dim)
        full = full.index_copy(0, flat_mask.nonzero(as_tuple=True)[0], samples)
        return full.reshape(batch_size, slate_length, self.group_size, dim).permute(
            0, 2, 1, 3
        )

    def _document_component(
        self,
        rollout_embeddings,
        policy_embeddings,
        kappa,
        sample,
        role_name,
        name,
        document_mask=None,
    ):
        if not sample:
            return _ActionComponent(
                role="document",
                rollout_embeddings=rollout_embeddings,
                name=name,
                document_mask=document_mask,
            )
        if policy_embeddings is None:
            raise ValueError(
                f"policy_{role_name}_document_embeddings required when sampled"
            )
        return _ActionComponent(
            role="document",
            rollout_embeddings=rollout_embeddings,
            policy_embeddings=policy_embeddings,
            sampled_embeddings=self._sample_document_embeddings(
                rollout_embeddings, kappa, document_mask
            ),
            kappa=kappa,
            name=name,
            document_mask=document_mask,
        )

    @staticmethod
    def _vmf_log_prob(
        policy_embeddings: torch.Tensor,
        sampled_embeddings: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        # log pi(e | h) = kappa * h^T e + log C_d(kappa). The normalizer is constant within each
        # group, so it cancels exactly against group-centered advantages — both in the policy-mean
        # gradient and in the learnable-kappa gradient — and is omitted. Cosines are computed in
        # fp32: kappa is large, so bf16 round-off in h^T e would dominate the log-prob differences.
        cosine = (
            sampled_embeddings.detach().float() * policy_embeddings.float().unsqueeze(1)
        ).sum(dim=-1)
        return kappa.float() * cosine

    @staticmethod
    def _document_vmf_log_prob(
        policy_document_embeddings: torch.Tensor,
        sampled_document_embeddings: torch.Tensor,
        kappa: torch.Tensor,
        document_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if policy_document_embeddings.size(1) == 0:
            return torch.zeros(
                sampled_document_embeddings.size(0),
                sampled_document_embeddings.size(1),
                device=sampled_document_embeddings.device,
                dtype=torch.float32,
            )

        cosine = (
            sampled_document_embeddings.detach().float()
            * policy_document_embeddings.float().unsqueeze(1)
        ).sum(dim=-1)
        log_prob = kappa.float() * cosine
        if document_mask is not None:
            log_prob = log_prob * document_mask.unsqueeze(1).to(log_prob.dtype)
        return log_prob.sum(dim=-1)

    @staticmethod
    def _per_document_vmf_log_prob(component: _ActionComponent) -> torch.Tensor:
        """Keep [batch, group, document] credit until after advantage weighting."""
        cosine = (
            component.sampled_embeddings.detach().float()
            * component.policy_embeddings.float().unsqueeze(1)
        ).sum(dim=-1)
        log_prob = component.kappa.float() * cosine
        if component.document_mask is not None:
            log_prob = log_prob * component.document_mask.unsqueeze(1)
        return log_prob

    @staticmethod
    def _resolve_group_size(active_components: Sequence[_ActionComponent]) -> int:
        if not active_components:
            raise ValueError("At least one action component is required")
        group_size = active_components[0].sampled_embeddings.size(1)
        for component in active_components:
            if component.sampled_embeddings.size(1) != group_size:
                raise ValueError("action component group sizes must match")
        return group_size

    @staticmethod
    def _compute_score_table(
        query_component: _ActionComponent,
        document_component: _ActionComponent,
        cross: bool = False,
        diagonal: bool = False,
    ) -> torch.Tensor:
        query_embeddings = (
            query_component.sampled_embeddings
            if query_component.is_active
            else query_component.rollout_embeddings.detach()
        )
        document_embeddings = (
            document_component.sampled_embeddings
            if document_component.is_active
            else document_component.rollout_embeddings.detach()
        )
        # fp32: the reward ranks candidates by these scores, and bf16 resolves cosines only
        # to ~4e-3 near 1.0 — coarser than the score gaps inside a hard-negative slate. The
        # resulting ties can make the reward depend on candidate order and the ranking
        # implementation instead of the actual embedding geometry.
        query_embeddings = F.normalize(query_embeddings.float(), dim=-1)
        document_embeddings = F.normalize(document_embeddings.float(), dim=-1)

        # query: 'bqd' if active else 'bd'; document (same-batch): 'bksd' if active else 'bsd';
        # document (cross-batch): swaps the leading 'b' for 'c' to pair every query with every other batch's docs.
        # Under the diagonal rollout the two sides share one group index, so the contraction
        # emits only the paired entries r^(g,g) instead of the full G x G grid.
        document_index = "q" if diagonal else "k"
        query_spec = "bqd" if query_component.is_active else "bd"
        doc_batch = "c" if cross else "b"
        doc_spec = (
            f"{doc_batch}{document_index}sd"
            if document_component.is_active
            else f"{doc_batch}sd"
        )
        out_spec = "b"
        if cross:
            out_spec += "c"
        if query_component.is_active or (diagonal and document_component.is_active):
            out_spec += "q"
        if document_component.is_active and not diagonal:
            out_spec += "k"
        out_spec += "s"
        with fp32_scores(query_embeddings.device):
            return torch.einsum(
                f"{query_spec},{doc_spec}->{out_spec}",
                query_embeddings,
                document_embeddings,
            )

    @staticmethod
    def _mask_cross_batch_diagonal(
        score_table: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        # Drop each sample's own documents from its cross-batch distractor pool. Masking the
        # compact [batch, candidate_batch, ...] table here rather than the expanded grid avoids
        # materializing a second copy of a tensor that carries one axis per action component.
        diagonal_mask = torch.eye(
            batch_size, device=score_table.device, dtype=torch.bool
        )
        trailing_dims = [1] * (score_table.dim() - 2)
        return score_table.masked_fill(
            diagonal_mask.reshape(batch_size, batch_size, *trailing_dims),
            float("-inf"),
        )

    @staticmethod
    def _expand_score_table(
        score_table: torch.Tensor,
        query_component: _ActionComponent,
        document_component: _ActionComponent,
        active_index_by_id: dict[int, int],
        num_components: int,
        group_size: int,
        cross: bool = False,
    ) -> torch.Tensor:
        # _compute_score_table emits dims in the order (query, document); the reshape below relies on
        # active_index_by_id[query] < active_index_by_id[document] so the non-singleton slots line up.
        if query_component.is_active and document_component.is_active:
            assert (
                active_index_by_id[id(query_component)]
                <= active_index_by_id[id(document_component)]
            ), "query component must precede document component in active_index_by_id"

        slate_length = score_table.size(-1)
        if cross:
            batch_size, candidate_batch_size = score_table.shape[:2]
            leading = [batch_size, candidate_batch_size]
            offset = 2
        else:
            batch_size = score_table.size(0)
            leading = [batch_size]
            offset = 1

        view_shape = [*leading, *([1] * num_components), slate_length]
        if query_component.is_active:
            view_shape[offset + active_index_by_id[id(query_component)]] = group_size
        if document_component.is_active:
            view_shape[offset + active_index_by_id[id(document_component)]] = group_size
        expanded = score_table.reshape(view_shape).expand(
            *leading,
            *([group_size] * num_components),
            slate_length,
        )
        if cross:
            return expanded.permute(
                0, *range(2, 2 + num_components), 1, 2 + num_components
            )
        return expanded

    def _cross_batch_component(self, component: _ActionComponent) -> _ActionComponent:
        # In-batch candidates from OTHER samples enter each sample's slate as distractors. Scoring
        # them with their sampled embeddings would leak every other sample's perturbation into this
        # sample's advantages through the shared group index (cross-sample credit contamination),
        # so by default they are represented by their detached mean embeddings.
        if self.in_batch_use_sampled_documents:
            return component
        return _ActionComponent(
            role=component.role,
            rollout_embeddings=component.rollout_embeddings,
            name=component.name,
            document_mask=component.document_mask,
        )

    def _frozen_doc_scale(
        self, document_components: Sequence[_ActionComponent]
    ) -> torch.Tensor | None:
        # A sampled vMF embedding is attenuated toward the origin in expectation: E[e] = A_d(kappa) mu.
        # Any candidate scored at its frozen mean embedding (an unsampled slate component, or in-batch
        # candidates under the default in_batch_use_sampled_documents=False) therefore lands on a score
        # scale 1/A_d(kappa) above the sampled documents it competes with (~3x at kappa=400, d=1024)
        # and systematically outranks them, collapsing ranking rewards to a constant. Whenever at
        # least one document component is sampled, every frozen-document score table is rescaled by
        # A_d(kappa) so all candidates share the same expected score scale.
        if not self.frozen_doc_rescale:
            return None
        sampled = [
            component for component in document_components if component.is_active
        ]
        if not sampled:
            return None
        dim = sampled[0].rollout_embeddings.size(-1)
        kappa = sampled[0].kappa.detach()
        if self.sampling_law == "gaussian":
            # Match the law actually sampled from, so the ablation compares sampling fidelity
            # rather than an incidental miscalibration of the frozen candidates.
            alignment = projected_gaussian_mean_alignment(float(kappa) ** -0.5, dim)
            return torch.as_tensor(alignment, dtype=torch.float32, device=kappa.device)
        return bessel_ratio(dim / 2.0, kappa)

    @staticmethod
    def _split_components(
        components: Sequence[_ActionComponent],
    ) -> tuple[_ActionComponent, tuple[_ActionComponent, ...]]:
        queries = tuple(
            component for component in components if component.role == "query"
        )
        documents = tuple(
            component for component in components if component.role == "document"
        )
        if len(queries) != 1:
            raise ValueError("Exactly one query component is required")
        if not documents:
            raise ValueError("At least one document component is required")
        return queries[0], documents

    def _build_rollout_layout(
        self,
        components: Sequence[_ActionComponent],
    ) -> _RolloutLayout:
        """Resolve action roles, rollout axes, and policy log probabilities once."""
        active = tuple(component for component in components if component.is_active)
        query, documents = self._split_components(components)

        diagonal = self.rollout == "diagonal"
        group_size = self._resolve_group_size(active)
        active_axis_by_id = {
            id(component): (0 if diagonal else index)
            for index, component in enumerate(active)
        }
        log_probs = []
        for component in active:
            if component.role == "query":
                log_probs.append(
                    self._vmf_log_prob(
                        policy_embeddings=component.policy_embeddings,
                        sampled_embeddings=component.sampled_embeddings,
                        kappa=component.kappa,
                    )
                )
            elif component.role == "document":
                log_probs.append(
                    self._per_document_vmf_log_prob(component)
                    if self.document_advantage_baseline == "counterfactual"
                    else self._document_vmf_log_prob(
                        policy_document_embeddings=component.policy_embeddings,
                        sampled_document_embeddings=component.sampled_embeddings,
                        kappa=component.kappa,
                        document_mask=component.document_mask,
                    )
                )
            else:
                raise ValueError(f"Unsupported action component role: {component.role}")

        return _RolloutLayout(
            query=query,
            documents=documents,
            active=active,
            log_probs=tuple(log_probs),
            active_axis_by_id=active_axis_by_id,
            diagonal=diagonal,
            num_axes=1 if diagonal else len(active),
            group_size=group_size,
            frozen_document_scale=self._frozen_doc_scale(documents),
        )

    def _score_grid(
        self,
        layout: _RolloutLayout,
        document: _ActionComponent,
        *,
        cross: bool = False,
    ) -> torch.Tensor:
        """Score one document component and broadcast it onto the rollout axes."""
        if cross:
            document = self._cross_batch_component(document)
        score_table = self._compute_score_table(
            query_component=layout.query,
            document_component=document,
            cross=cross,
            diagonal=layout.diagonal,
        )
        if layout.frozen_document_scale is not None and not document.is_active:
            score_table = score_table * layout.frozen_document_scale.to(
                score_table.dtype
            )
        if cross:
            batch_size = layout.query.rollout_embeddings.size(0)
            if document.document_mask is not None:
                mask = document.document_mask
                shape = [1, batch_size] + [1] * (score_table.ndim - 3) + [mask.size(-1)]
                score_table = score_table.masked_fill(
                    ~mask.reshape(shape), float("-inf")
                )
            score_table = self._mask_cross_batch_diagonal(score_table, batch_size)
        return self._expand_score_table(
            score_table=score_table,
            query_component=layout.query,
            document_component=document,
            active_index_by_id=layout.active_axis_by_id,
            num_components=layout.num_axes,
            group_size=layout.group_size,
            cross=cross,
        )

    def _build_score_context(
        self,
        layout: _RolloutLayout,
        *,
        in_batch_positive_mask=None,
        in_batch_candidate_mask=None,
        cross_batch_metadata=None,
    ) -> _ScoreContext:
        """Construct own and cross-query candidate pools before reward evaluation."""
        batch_size = layout.query.rollout_embeddings.size(0)
        own_scores = torch.cat(
            [self._score_grid(layout, document) for document in layout.documents],
            dim=-1,
        )

        fixed_cross_pool = fixed_cross_mask = fixed_cross_scores = None
        loss_weight = 1.0
        if self.reward_cross_device_negatives:
            means = torch.cat(
                [document.rollout_embeddings for document in layout.documents], dim=1
            )
            fixed_cross_pool, fixed_cross_mask, loss_weight = cross_query_document_pool(
                means,
                cross_batch_metadata,
                include_negatives=True,
                cross_device=True,
                detach_documents=True,
            )
            with torch.no_grad(), fp32_scores(own_scores.device):
                queries = F.normalize(layout.query.sampled_embeddings.float(), dim=-1)
                fixed_cross_scores = (
                    queries @ F.normalize(fixed_cross_pool.float(), dim=-1).T
                )
                if layout.frozen_document_scale is not None:
                    fixed_cross_scores = (
                        fixed_cross_scores * layout.frozen_document_scale
                    )
                fixed_cross_scores = fixed_cross_scores.masked_fill(
                    ~fixed_cross_mask[:, None], -torch.inf
                )

        in_batch_positives = None
        requirements = self.reward_evaluator.requirements
        if batch_size > 1 and requirements.representative_candidates:
            in_batch_positives = self._score_grid(
                layout, layout.documents[0], cross=True
            )[..., 0]
            if in_batch_positive_mask is not None:
                shape = [batch_size] + [1] * layout.num_axes + [batch_size]
                in_batch_positives = in_batch_positives.masked_fill(
                    ~in_batch_positive_mask.reshape(shape), float("-inf")
                )

        in_batch_candidates = None
        if (
            not self.reward_cross_device_negatives
            and batch_size > 1
            and requirements.all_candidates
        ):
            tables = []
            start = 0
            for document in layout.documents:
                table = self._score_grid(layout, document, cross=True)
                length = document.rollout_embeddings.size(1)
                if in_batch_candidate_mask is not None:
                    mask = in_batch_candidate_mask[..., start : start + length]
                    shape = [batch_size] + [1] * layout.num_axes + [batch_size, length]
                    table = table.masked_fill(~mask.reshape(shape), float("-inf"))
                tables.append(
                    table.reshape(
                        batch_size,
                        *([layout.group_size] * layout.num_axes),
                        -1,
                    )
                )
                start += length
            in_batch_candidates = torch.cat(tables, dim=-1)

        return _ScoreContext(
            own=own_scores,
            in_batch_positives=in_batch_positives,
            in_batch_candidates=in_batch_candidates,
            fixed_cross_scores=fixed_cross_scores,
            fixed_cross_pool=fixed_cross_pool,
            fixed_cross_mask=fixed_cross_mask,
            loss_weight=loss_weight,
        )

    def _counterfactual_advantages(
        self,
        layout: _RolloutLayout,
        scores: _ScoreContext,
        rewards: torch.Tensor,
        relevance_labels: torch.Tensor,
        rank_labels: torch.Tensor | None,
        candidate_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.document_advantage_baseline != "counterfactual":
            return None

        # Validation guarantees one active document component containing the slate.
        (document,) = layout.documents
        reference = _ActionComponent(
            role="document",
            rollout_embeddings=document.rollout_embeddings,
        )
        reference_table = self._compute_score_table(layout.query, reference)
        reference_scores = self._expand_score_table(
            reference_table,
            layout.query,
            reference,
            layout.active_axis_by_id,
            layout.num_axes,
            layout.group_size,
        )
        return self.reward_evaluator.counterfactual_document_advantages(
            scores.reward_inputs(relevance_labels, rank_labels, candidate_mask),
            reference_scores=reference_scores,
            rewards=rewards,
            document_axis=layout.active_axis_by_id[id(document)] + 1,
        )

    def _advantage_inputs(
        self,
        rewards: RewardEvaluation,
    ) -> tuple[_AdvantageInput, ...]:
        """Describe which reward tensor supplies each independently scaled advantage."""
        if len(rewards.signals) == 1:
            raw = ((rewards.signals[0].name, 1.0, rewards.combined),)
        elif self.reward_combine == "sum":
            raw = (("combined", 1.0, rewards.combined),)
        else:
            raw = tuple(
                (signal.name, signal.weight, signal.values)
                for signal in rewards.signals
                if signal.weight != 0.0
            )

        batch_size = rewards.combined.size(0)
        return tuple(
            _AdvantageInput(
                name=name,
                weight=weight,
                rewards=values,
                shared_std=(
                    values.reshape(batch_size, -1).std(
                        dim=-1, unbiased=False, keepdim=True
                    )
                    if self.advantage_norm == "shared"
                    else None
                ),
            )
            for name, weight, values in raw
        )

    def _score_function_surrogate(
        self,
        layout: _RolloutLayout,
        scores: _ScoreContext,
        rewards: RewardEvaluation,
        relevance_labels: torch.Tensor,
        rank_labels: torch.Tensor | None,
        candidate_mask: torch.Tensor | None,
    ) -> _SurrogateResult:
        """Marginalize rewards per action axis and build the policy surrogate."""
        batch_size = relevance_labels.size(0)
        per_document_advantages = self._counterfactual_advantages(
            layout,
            scores,
            rewards.combined,
            relevance_labels,
            rank_labels,
            candidate_mask,
        )
        inputs = self._advantage_inputs(rewards)
        ema_baseline = (
            self._current_reward_baseline(rewards.combined)
            if self.advantage_baseline == "ema"
            else None
        )

        losses = []
        all_advantages = []
        degenerate_masks = []
        statistics: dict[str, torch.Tensor] = {}
        sample_dims = tuple(range(1, 1 + layout.num_axes))

        for component_index, (component, log_prob) in enumerate(
            zip(layout.active, layout.log_probs)
        ):
            axis = 0 if layout.diagonal else component_index
            other_dims = tuple(dim for dim in sample_dims if dim != axis + 1)
            component_advantages = None
            for item in inputs:
                component_rewards = (
                    item.rewards.mean(dim=other_dims) if other_dims else item.rewards
                )
                term_advantages, term_degenerate = self._compute_advantages(
                    component_rewards,
                    shared_std=item.shared_std,
                    external_baseline=ema_baseline,
                )
                if item.weight != 1.0:
                    term_advantages = term_advantages * item.weight
                component_advantages = (
                    term_advantages
                    if component_advantages is None
                    else component_advantages + term_advantages
                )
                degenerate_masks.append(term_degenerate)
                prefix = f"reward/{item.name}/{component.name}"
                statistics[f"{prefix}/group_std"] = (
                    component_rewards.detach()
                    .reshape(batch_size, -1)
                    .std(dim=-1, unbiased=False)
                    .mean()
                )
                statistics[f"{prefix}/degenerate_frac"] = (
                    term_degenerate.detach().float().mean()
                )

            weighted_log_prob = log_prob
            if component.role == "document" and per_document_advantages is not None:
                valid = (
                    torch.ones_like(relevance_labels, dtype=torch.bool)
                    if component.document_mask is None
                    else component.document_mask
                )
                valid_advantages = per_document_advantages[
                    valid.unsqueeze(1).expand_as(per_document_advantages)
                ]
                prefix = f"baseline/{component.name}/counterfactual"
                statistics.update(
                    self.summarize_tensor(
                        valid_advantages,
                        prefix=prefix,
                        separator="/",
                    )
                )
                statistics[f"{prefix}/zero_frac"] = (
                    (valid_advantages == 0).float().mean()
                )
                component_advantages = per_document_advantages

            if (
                component.role == "document"
                and self.document_log_prob_reduction == "mean"
            ):
                count = (
                    log_prob.new_full(
                        (batch_size, 1), component.policy_embeddings.size(1)
                    )
                    if component.document_mask is None
                    else component.document_mask.sum(-1, keepdim=True).clamp_min(1)
                )
                weighted_log_prob = log_prob / (
                    count.unsqueeze(-1) if log_prob.ndim == 3 else count
                )

            weighted = component_advantages.detach() * weighted_log_prob
            losses.append(
                -(weighted.sum(-1) if weighted.ndim == 3 else weighted).mean()
            )
            all_advantages.append(component_advantages.reshape(batch_size, -1))

        return _SurrogateResult(
            loss=sum(losses),
            advantages=torch.cat(all_advantages, dim=1),
            degenerate_fraction=torch.cat(degenerate_masks).float().mean(),
            statistics=statistics,
        )

    def _conditional_projection_surrogate(
        self,
        layout: _RolloutLayout,
        scores: _ScoreContext,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(layout.active) != 2 or len(layout.documents) != 1:
            raise ValueError(
                "Conditional projection requires one query and one joint document group"
            )
        document = layout.documents[0]
        for component in (layout.query, document):
            if not torch.allclose(
                component.policy_embeddings.detach(),
                component.rollout_embeddings,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    "Conditional projection requires on-policy mean directions"
                )

        frozen_documents, frozen_mask = fixed_projection_candidates(
            document.rollout_embeddings,
            in_batch_positive_scores=scores.in_batch_positives,
            in_batch_candidate_scores=scores.in_batch_candidates,
            fixed_cross_pool=scores.fixed_cross_pool,
            fixed_cross_mask=scores.fixed_cross_mask,
        )
        loss, ranks = conditional_projection_loss(
            layout.query.policy_embeddings,
            document.policy_embeddings,
            layout.query.sampled_embeddings,
            document.sampled_embeddings,
            rewards,
            layout.query.kappa,
            document.document_mask,
            frozen_documents=frozen_documents,
            frozen_mask=frozen_mask,
            stream_frozen=self.reward_cross_device_negatives,
        )
        return loss, {
            "projection/query_span_rank_mean": ranks.float().mean(),
            "projection/query_span_rank_max": ranks.max().float(),
        }

    def _reward_statistics(
        self,
        layout: _RolloutLayout,
        scores: _ScoreContext,
        rewards: RewardEvaluation,
        relevance_labels: torch.Tensor,
        candidate_mask: torch.Tensor | None,
        estimator_statistics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        statistics = self.reward_evaluator.statistics(rewards)
        if scores.fixed_cross_mask is not None:
            counts = scores.fixed_cross_mask.sum(-1).float()
            statistics["reward_pool/cross_candidates_mean"] = counts.mean()
            statistics["reward_pool/cross_candidates_max"] = counts.max()
            statistics["reward_pool/zero_reward_frac"] = (
                (rewards.combined == 0).float().mean()
            )

        with torch.no_grad(), fp32_scores(rewards.combined.device):
            means = torch.cat(
                [document.rollout_embeddings for document in layout.documents], dim=1
            )
            deterministic = torch.einsum(
                "bd,bnd->bn",
                layout.query.rollout_embeddings.float(),
                means.float(),
            )
            statistics.update(
                self.reward_evaluator.exploration_statistics(
                    deterministic,
                    scores.own,
                    relevance_labels,
                    candidate_mask,
                )
            )

        statistics.update(estimator_statistics)
        return statistics

    def _compute_component_loss(
        self,
        relevance_labels: torch.Tensor,
        rank_labels: torch.Tensor | None,
        components: Sequence[_ActionComponent],
        candidate_mask=None,
        in_batch_positive_mask=None,
        in_batch_candidate_mask=None,
        cross_batch_metadata=None,
        positive_mask=None,
    ) -> PolicyObjectiveResult:
        query, documents = self._split_components(components)
        if self.reward_shortlist_count:
            return shortlist_policy_objective(
                settings=ShortlistSettings.from_source(self),
                evaluator=self.reward_evaluator,
                rollout_rng=self.rollout_rng,
                step=self.exploration.step,
                training=self.training,
                query=query,
                documents=documents,
                labels=relevance_labels,
                rank_labels=rank_labels,
                metadata=cross_batch_metadata,
                positive_mask=positive_mask,
                frozen_document_scale=self._frozen_doc_scale(documents),
            )
        if self.cross_query_document_gradients:
            return shared_document_policy_objective(
                evaluator=self.reward_evaluator,
                query=query,
                documents=documents,
                labels=relevance_labels,
                cross_device=self.reward_cross_device_negatives,
                gradient_estimator=self.gradient_estimator,
                in_batch_positive_mask=in_batch_positive_mask,
                in_batch_candidate_mask=in_batch_candidate_mask,
                metadata=cross_batch_metadata,
            )
        layout = self._build_rollout_layout(components)
        score_context = self._build_score_context(
            layout,
            in_batch_positive_mask=in_batch_positive_mask,
            in_batch_candidate_mask=in_batch_candidate_mask,
            cross_batch_metadata=cross_batch_metadata,
        )
        reward_evaluation = self.reward_evaluator.evaluate(
            score_context.reward_inputs(relevance_labels, rank_labels, candidate_mask)
        )
        surrogate = self._score_function_surrogate(
            layout,
            score_context,
            reward_evaluation,
            relevance_labels,
            rank_labels,
            candidate_mask,
        )
        loss = surrogate.loss
        estimator_statistics = dict(surrogate.statistics)

        if self.gradient_estimator == "conditional_projection":
            loss, projection_statistics = self._conditional_projection_surrogate(
                layout,
                score_context,
                reward_evaluation.combined,
            )
            estimator_statistics.update(projection_statistics)

        reward_statistics = self._reward_statistics(
            layout,
            score_context,
            reward_evaluation,
            relevance_labels,
            candidate_mask,
            estimator_statistics,
        )
        return PolicyObjectiveResult(
            loss=loss * score_context.loss_weight,
            statistics=reward_statistics,
            advantages=surrogate.advantages,
            degenerate_fraction=surrogate.degenerate_fraction,
        )

    _kl_term = staticmethod(vmf_kl)

    @staticmethod
    def _normalize_policy(
        policy_embeddings: torch.Tensor | None,
        rollout_embeddings: torch.Tensor,
        policy_name: str,
        rollout_name: str,
    ) -> torch.Tensor | None:
        """Validate a policy tensor against its rollout twin and unit-normalize it."""
        if policy_embeddings is None:
            return None
        if policy_embeddings.shape != rollout_embeddings.shape:
            raise ValueError(
                f"{policy_name} shape must match {rollout_name}, "
                f"got policy={tuple(policy_embeddings.shape)} "
                f"rollout={tuple(rollout_embeddings.shape)}"
            )
        return F.normalize(policy_embeddings.float(), dim=-1)

    def _prepare_policy_inputs(
        self,
        rollout_query: torch.Tensor,
        rollout_positive: torch.Tensor,
        rollout_negative: torch.Tensor,
        relevance_labels: torch.Tensor,
        candidate_mask: torch.Tensor | None,
        *,
        policy_query: torch.Tensor | None,
        policy_positive: torch.Tensor | None,
        policy_negative: torch.Tensor | None,
        rank_labels: torch.Tensor | None,
    ) -> _PolicyInputs:
        """Validate the slate contract and normalize every embedding exactly once."""
        if relevance_labels is None:
            raise ValueError("relevance_labels are required for GRPO training")
        if rollout_positive.dim() != 3:
            raise ValueError(
                "positive_document_embeddings must be [batch, 1, dim], "
                f"got shape {tuple(rollout_positive.shape)}"
            )
        if rollout_positive.size(1) != 1:
            raise ValueError(
                "positive_document_embeddings must contain exactly one document per sample"
            )
        if rollout_negative.dim() != 3:
            raise ValueError(
                "negative_document_embeddings must be [batch, negatives, dim], "
                f"got shape {tuple(rollout_negative.shape)}"
            )
        if rollout_negative.size(0) != rollout_positive.size(0):
            raise ValueError("positive and negative document batch sizes must match")
        if rollout_negative.size(-1) != rollout_positive.size(-1):
            raise ValueError("positive and negative document embedding dims must match")

        documents = torch.cat((rollout_positive, rollout_negative), dim=1)
        if relevance_labels.shape != documents.shape[:2]:
            raise ValueError(
                "relevance_labels shape must match [batch, slate], "
                f"got labels={tuple(relevance_labels.shape)} "
                f"documents={tuple(documents.shape)}"
            )
        if rank_labels is not None and rank_labels.shape != relevance_labels.shape:
            raise ValueError(
                "rank_labels shape must match relevance_labels [batch, slate], "
                f"got ranks={tuple(rank_labels.shape)} "
                f"labels={tuple(relevance_labels.shape)}"
            )

        if candidate_mask is None:
            candidate_mask = torch.ones_like(relevance_labels, dtype=torch.bool)
        if (
            candidate_mask.shape != relevance_labels.shape
            or not candidate_mask[:, 0].all()
        ):
            raise ValueError(
                "candidate_mask must match labels and keep the positive at position 0"
            )
        if candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be bool")

        query = F.normalize(rollout_query.float(), dim=-1)
        positive = F.normalize(rollout_positive.float(), dim=-1)
        negative = F.normalize(rollout_negative.float(), dim=-1)
        return _PolicyInputs(
            query=query,
            positive=positive,
            negative=negative,
            documents=torch.cat((positive, negative), dim=1),
            policy_query=self._normalize_policy(
                policy_query,
                query,
                "policy_query_embeddings",
                "rollout_query_embeddings",
            ),
            policy_positive=self._normalize_policy(
                policy_positive,
                positive,
                "policy_positive_document_embeddings",
                "positive_document_embeddings",
            ),
            policy_negative=self._normalize_policy(
                policy_negative,
                negative,
                "policy_negative_document_embeddings",
                "negative_document_embeddings",
            ),
            candidate_mask=candidate_mask,
        )

    def _exploration_parameters(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve the scalar exploration scale used by this rollout."""
        sigma = self.current_sigma(device=query.device, dtype=torch.float32)
        if not self.sigma_learnable:
            kappa_value = self.exploration.resolve(query.size(-1))
            sigma = sigma.new_tensor(kappa_value**-0.5)
        return sigma, sigma.pow(-2)

    def _build_action_components(
        self,
        inputs: _PolicyInputs,
        kappa: torch.Tensor,
    ) -> tuple[tuple[_ActionComponent, ...], bool]:
        """Materialize the configured query/document action groups."""
        if self.sample_query and inputs.policy_query is None:
            raise ValueError(
                "policy_query_embeddings are required when query is sampled"
            )
        components = [
            _ActionComponent(
                role="query",
                rollout_embeddings=inputs.query,
                policy_embeddings=inputs.policy_query,
                sampled_embeddings=self._draw(inputs.query.detach(), kappa),
                kappa=kappa,
                name="query",
            )
            if self.sample_query
            else _ActionComponent(
                role="query",
                rollout_embeddings=inputs.query,
                name="query",
            )
        ]

        joint_documents = any(
            set(group) == {"positive", "negative"} for group in self.action_components
        )
        if joint_documents:
            for role_name, policy in (
                ("positive", inputs.policy_positive),
                ("negative", inputs.policy_negative),
            ):
                if policy is None:
                    raise ValueError(
                        f"policy_{role_name}_document_embeddings are required "
                        f"when {role_name} is sampled"
                    )
            components.append(
                self._document_component(
                    rollout_embeddings=inputs.documents,
                    policy_embeddings=torch.cat(
                        (inputs.policy_positive, inputs.policy_negative), dim=1
                    ),
                    kappa=kappa,
                    sample=True,
                    role_name="positive",
                    name="documents",
                    document_mask=inputs.candidate_mask,
                )
            )
            return tuple(components), True

        components.extend(
            (
                self._document_component(
                    rollout_embeddings=inputs.positive,
                    policy_embeddings=inputs.policy_positive,
                    kappa=kappa,
                    sample=self.sample_positive,
                    role_name="positive",
                    name="positive",
                    document_mask=inputs.candidate_mask[:, :1],
                ),
                self._document_component(
                    rollout_embeddings=inputs.negative,
                    policy_embeddings=inputs.policy_negative,
                    kappa=kappa,
                    sample=self.sample_negative,
                    role_name="negative",
                    name="negative",
                    document_mask=inputs.candidate_mask[:, 1:],
                ),
            )
        )
        return tuple(components), False

    def _kl_loss(
        self,
        inputs: _PolicyInputs,
        kappa: torch.Tensor,
        *,
        joint_documents: bool,
        reference_query: torch.Tensor | None,
        reference_positive: torch.Tensor | None,
        reference_negative: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute the configured KL regularizer over sampled action groups."""
        if self.kl_coef == 0:
            return kappa.new_zeros(())

        terms = []
        if self.sample_query:
            if reference_query is None:
                raise ValueError(
                    "reference_query_embeddings are required when kl_coef > 0 "
                    "and query is sampled"
                )
            terms.append(self._kl_term(inputs.policy_query, reference_query, kappa))

        if joint_documents:
            if reference_positive is None or reference_negative is None:
                raise ValueError(
                    "reference_positive_document_embeddings and "
                    "reference_negative_document_embeddings are required when "
                    "kl_coef > 0 and the joint document group is sampled"
                )
            terms.append(
                self._kl_term(
                    torch.cat((inputs.policy_positive, inputs.policy_negative), dim=1),
                    torch.cat((reference_positive, reference_negative), dim=1),
                    kappa,
                    inputs.candidate_mask,
                )
            )
        else:
            for role_name, sampled, policy, reference, mask in (
                (
                    "positive",
                    self.sample_positive,
                    inputs.policy_positive,
                    reference_positive,
                    inputs.candidate_mask[:, :1],
                ),
                (
                    "negative",
                    self.sample_negative,
                    inputs.policy_negative,
                    reference_negative,
                    inputs.candidate_mask[:, 1:],
                ),
            ):
                if not sampled:
                    continue
                if reference is None:
                    raise ValueError(
                        f"reference_{role_name}_document_embeddings required when "
                        f"kl_coef > 0 and {role_name} is sampled"
                    )
                terms.append(self._kl_term(policy, reference, kappa, mask))
        return torch.stack(terms).sum() if terms else kappa.new_zeros(())

    def _auxiliary_loss(
        self,
        inputs: _PolicyInputs,
        positive_mask: torch.Tensor | None,
        in_batch_positive_mask: torch.Tensor | None,
        index_route_ids: torch.Tensor | None,
        cross_batch_metadata: list[dict] | None,
    ) -> torch.Tensor | None:
        """Compute direct InfoNCE without expanding the RL action scope."""
        if self.aux_infonce_coef == 0:
            return None
        queries = inputs.policy_query if self.sample_query else inputs.query.detach()
        documents = torch.cat(
            (
                inputs.policy_positive
                if self.sample_positive
                else inputs.positive.detach(),
                inputs.policy_negative
                if self.sample_negative
                else inputs.negative.detach(),
            ),
            dim=1,
        )
        return auxiliary_infonce_loss(
            queries,
            documents,
            positive_mask,
            inputs.candidate_mask,
            temperature=self.aux_infonce_temperature,
            use_in_batch_negatives=self.aux_infonce_use_in_batch_negatives,
            in_batch_positive_mask=in_batch_positive_mask,
            index_route_ids=index_route_ids,
            strong_negatives=self.aux_infonce_strong_negatives,
            cross_batch_metadata=cross_batch_metadata,
        )

    def forward(
        self,
        rollout_query_embeddings: torch.Tensor,
        rollout_positive_document_embeddings: torch.Tensor,
        rollout_negative_document_embeddings: torch.Tensor,
        relevance_labels: torch.Tensor,
        policy_query_embeddings: torch.Tensor | None = None,
        policy_positive_document_embeddings: torch.Tensor | None = None,
        policy_negative_document_embeddings: torch.Tensor | None = None,
        reference_query_embeddings: torch.Tensor | None = None,
        reference_positive_document_embeddings: torch.Tensor | None = None,
        reference_negative_document_embeddings: torch.Tensor | None = None,
        rank_labels: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        in_batch_positive_mask: torch.Tensor | None = None,
        in_batch_candidate_mask: torch.Tensor | None = None,
        positive_mask: torch.Tensor | None = None,
        index_route_ids: torch.Tensor | None = None,
        cross_batch_metadata: list[dict] | None = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        inputs = self._prepare_policy_inputs(
            rollout_query_embeddings,
            rollout_positive_document_embeddings,
            rollout_negative_document_embeddings,
            relevance_labels,
            candidate_mask,
            policy_query=policy_query_embeddings,
            policy_positive=policy_positive_document_embeddings,
            policy_negative=policy_negative_document_embeddings,
            rank_labels=rank_labels,
        )
        sigma, kappa = self._exploration_parameters(inputs.query)
        components, joint_documents = self._build_action_components(inputs, kappa)
        loss, reward_stats, advantages, degenerate_frac = self._compute_component_loss(
            relevance_labels=relevance_labels,
            rank_labels=rank_labels,
            components=components,
            candidate_mask=inputs.candidate_mask,
            in_batch_positive_mask=in_batch_positive_mask,
            in_batch_candidate_mask=in_batch_candidate_mask,
            cross_batch_metadata=cross_batch_metadata,
            positive_mask=positive_mask,
        )

        policy_loss = loss
        kl = self._kl_loss(
            inputs,
            kappa,
            joint_documents=joint_documents,
            reference_query=reference_query_embeddings,
            reference_positive=reference_positive_document_embeddings,
            reference_negative=reference_negative_document_embeddings,
        )
        loss = loss + self.kl_coef * kl

        auxiliary = self._auxiliary_loss(
            inputs,
            positive_mask,
            in_batch_positive_mask,
            index_route_ids,
            cross_batch_metadata,
        )
        if auxiliary is not None:
            weighted_auxiliary = self.aux_infonce_coef * auxiliary
            reward_stats.update(
                {
                    "train/loss_rl": policy_loss.detach(),
                    "train/loss_infonce": auxiliary.detach(),
                    "train/loss_infonce_weighted": weighted_auxiliary.detach(),
                }
            )
            loss = loss + weighted_auxiliary

        reward_stats["exploration/kappa"] = kappa.detach()
        reward_stats["exploration/mean_alignment"] = kappa.new_tensor(
            mean_alignment(inputs.query.size(-1), float(kappa.detach()))
            if self.sampling_law == "vmf"
            else projected_gaussian_mean_alignment(
                float(sigma.detach()), inputs.query.size(-1)
            )
        )
        advantage_stats = self.summarize_tensor(advantages, prefix="advantages")
        advantage_stats["advantages_degenerate_frac"] = degenerate_frac.detach()
        return loss, reward_stats, advantage_stats, sigma.detach(), kl.detach()


class GRPOModel(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        rl_args: RLArguments,
        pooling_method: str = "last",
    ):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.pooling_method = pooling_method
        self.grpo = GRPO(config=rl_args)

    def encode(self, model_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return pool_embeddings(
            self.model(**model_inputs).last_hidden_state,
            model_inputs["attention_mask"],
            pooling_method=self.pooling_method,
            normalize=True,
        )

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        positive_document: Dict[str, torch.Tensor] = None,
        negative_document: Dict[str, torch.Tensor] = None,
        relevance_labels: torch.Tensor = None,
        rank_labels: torch.Tensor = None,
        positive_mask: torch.Tensor = None,
        candidate_mask: torch.Tensor = None,
        in_batch_positive_mask: torch.Tensor = None,
        in_batch_candidate_mask: torch.Tensor = None,
        cross_batch_metadata: list[dict] = None,
    ) -> GRPOModelOutput:
        if query is None:
            raise ValueError("query inputs are required for GRPO training")
        if positive_document is None:
            raise ValueError("positive document inputs are required for GRPO training")
        if negative_document is None:
            raise ValueError("negative document inputs are required for GRPO training")
        if relevance_labels is None:
            raise ValueError("relevance_labels are required for GRPO training")

        batch_size, slate_length = relevance_labels.shape
        if candidate_mask is None:
            candidate_mask = torch.ones_like(relevance_labels, dtype=torch.bool)
        if self.grpo.sample_query:
            policy_query_embeddings = self.encode(query)
            rollout_query_embeddings = policy_query_embeddings.detach()
        else:
            with torch.no_grad():
                rollout_query_embeddings = self.encode(query)
            rollout_query_embeddings = rollout_query_embeddings.detach()
            policy_query_embeddings = None

        document_inputs = build_slate_inputs(
            positive_document=positive_document,
            negative_document=negative_document,
            batch_size=batch_size,
            slate_length=slate_length,
        )

        # Legacy role names describe slots: representative vs remaining documents.
        # G1's document component samples both, including all additional positives.
        sample_document = self.grpo.sample_positive or self.grpo.sample_negative
        if sample_document:
            encoded_document_embeddings = encode_valid_candidates(
                self.encode, document_inputs, candidate_mask
            )
            policy_document_embeddings = encoded_document_embeddings.reshape(
                batch_size, slate_length, -1
            )
            rollout_document_embeddings = policy_document_embeddings.detach()
            policy_positive_document_embeddings = (
                policy_document_embeddings[:, :1] if self.grpo.sample_positive else None
            )
            policy_negative_document_embeddings = (
                policy_document_embeddings[:, 1:] if self.grpo.sample_negative else None
            )
        else:
            with torch.no_grad():
                encoded_document_embeddings = encode_valid_candidates(
                    self.encode, document_inputs, candidate_mask
                )
            rollout_document_embeddings = encoded_document_embeddings.detach().reshape(
                batch_size, slate_length, -1
            )
            policy_positive_document_embeddings = None
            policy_negative_document_embeddings = None

        rollout_positive_document_embeddings = rollout_document_embeddings[:, :1]
        rollout_negative_document_embeddings = rollout_document_embeddings[:, 1:]

        reference_query_embeddings = None
        reference_positive_document_embeddings = None
        reference_negative_document_embeddings = None
        if self.grpo.kl_coef > 0:
            if not hasattr(self.model, "disable_adapter"):
                raise RuntimeError(
                    "kl_coef > 0 requires a PEFT/LoRA model exposing .disable_adapter(); "
                    "either enable LoRA or set kl_coef=0."
                )
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    with self.model.disable_adapter():
                        if self.grpo.sample_query:
                            reference_query_embeddings = self.encode(query)
                        if sample_document:
                            encoded_reference_documents = encode_valid_candidates(
                                self.encode, document_inputs, candidate_mask
                            ).reshape(batch_size, slate_length, -1)
                            if self.grpo.sample_positive:
                                reference_positive_document_embeddings = (
                                    encoded_reference_documents[:, :1]
                                )
                            if self.grpo.sample_negative:
                                reference_negative_document_embeddings = (
                                    encoded_reference_documents[:, 1:]
                                )
            finally:
                self.model.train(was_training)

        loss, reward_stats, advantage_stats, sigma, kl = self.grpo(
            rollout_query_embeddings=rollout_query_embeddings,
            rollout_positive_document_embeddings=rollout_positive_document_embeddings,
            rollout_negative_document_embeddings=rollout_negative_document_embeddings,
            relevance_labels=relevance_labels,
            rank_labels=rank_labels,
            candidate_mask=candidate_mask,
            in_batch_positive_mask=in_batch_positive_mask,
            in_batch_candidate_mask=in_batch_candidate_mask,
            policy_query_embeddings=policy_query_embeddings,
            policy_positive_document_embeddings=policy_positive_document_embeddings,
            policy_negative_document_embeddings=policy_negative_document_embeddings,
            reference_query_embeddings=reference_query_embeddings,
            reference_positive_document_embeddings=reference_positive_document_embeddings,
            reference_negative_document_embeddings=reference_negative_document_embeddings,
            positive_mask=positive_mask,
            cross_batch_metadata=cross_batch_metadata,
        )
        # Namespaced keys are the per-term diagnostics; the flat ones (reward_mean/std/min/max
        # and advantages_*) are named exactly like the GRPOModelOutput fields they fill.
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

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
