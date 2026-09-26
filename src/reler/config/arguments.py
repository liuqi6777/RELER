import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments

from reler.data.protocol import validate_embedding_protocol
from reler.objectives.contrastive import validate_aux_infonce
from reler.objectives.rewards import (
    SUPPORTED_REWARD_TYPES,
    normalize_reward_combine_mode,
    normalize_reward_terms,
    reward_terms_mix_scales,
)
from reler.objectives.shortlists import (
    validate_shortlist_objectives,
    validate_shortlist_sampling,
)

logger = logging.getLogger(__name__)


def _checkpoint_value(value):
    """Convert normalized config values to a stable JSON-compatible structure."""
    if is_dataclass(value):
        return {
            item.name: _checkpoint_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_checkpoint_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _checkpoint_value(item) for key, item in value.items()}
    return value


SUPPORTED_ACTION_COMPONENTS = {"query", "positive", "negative"}

SUPPORTED_ADVANTAGE_NORM_MODES = ("per_component", "shared", "none")

SUPPORTED_ADVANTAGE_BASELINES = ("group", "leave_one_out", "ema")

SUPPORTED_DOCUMENT_ADVANTAGE_BASELINES = ("shared", "counterfactual")

SUPPORTED_SAMPLING_LAWS = ("vmf", "gaussian")

SUPPORTED_ROLLOUTS = ("product", "diagonal")

SUPPORTED_BASELINE_LOSSES = ("infonce", "ranknet", "lambdaloss")

SUPPORTED_GRADIENT_ESTIMATORS = ("score_function", "conditional_projection")


@dataclass(frozen=True)
class _PolicyConstraints:
    """Normalized policy fields shared by cross-feature validators."""

    action_components: tuple[tuple[str, ...], ...]
    sampling_law: str
    sigma_learnable: bool
    rollout: str
    advantage_baseline: str
    advantage_norm: str
    reward_combine: str
    in_batch_use_sampled_documents: bool
    document_advantage_baseline: str
    document_log_prob_reduction: str

    @classmethod
    def from_arguments(cls, arguments) -> "_PolicyConstraints":
        return cls(**{item.name: getattr(arguments, item.name) for item in fields(cls)})


def validate_reward_shortlists(
    count,
    size,
    hard_count,
    hard_pool_size,
    *,
    reward_cross_device_negatives,
    cross_query_document_gradients,
    rollout_seed,
    policy: _PolicyConstraints,
    pool_source="cross_device_all",
):
    if pool_source not in {
        "cross_device_all",
        "local_all",
        "cross_device_representatives",
    }:
        raise ValueError("Unknown reward_shortlist_pool_source")
    if not count and pool_source != "cross_device_all":
        raise ValueError(
            "A custom shortlist pool source requires reward_shortlist_count > 0"
        )
    validate_shortlist_sampling(count, size, hard_count, hard_pool_size)
    if not count:
        return
    if reward_cross_device_negatives != (pool_source != "local_all"):
        raise ValueError(
            "reward_cross_device_negatives must match reward_shortlist_pool_source"
        )
    if cross_query_document_gradients or rollout_seed is None:
        raise ValueError(
            "Reward shortlists require fixed documents and an explicit rollout_seed"
        )
    # Both SF and CP must estimate the same fixed-kappa, unnormalized LOO target.
    validate_gradient_estimator("conditional_projection", policy)


def validate_cross_query_document_gradients(
    enabled,
    *,
    reward_terms,
    policy: _PolicyConstraints,
    reward_cross_device_negatives=False,
    rollout_seed=None,
):
    if not enabled:
        return
    if (
        len(policy.action_components) != 2
        or ("query",) not in policy.action_components
        or not any(
            set(group) == {"positive", "negative"} for group in policy.action_components
        )
        or policy.sampling_law != "vmf"
        or policy.sigma_learnable
        or policy.rollout != "product"
        or policy.advantage_baseline != "leave_one_out"
        or policy.advantage_norm != "none"
        or policy.reward_combine != "sum"
        or policy.document_advantage_baseline != "shared"
        or policy.document_log_prob_reduction != "sum"
        or policy.in_batch_use_sampled_documents
    ):
        raise ValueError(
            "Cross-query document gradients require static joint vMF product rollouts, "
            "fixed kappa, LOO without normalization, shared document baseline, sum reductions, "
            "and the legacy in_batch_use_sampled_documents flag disabled"
        )
    if not reward_terms or any(
        term.type not in {"ndcg_in_batch", "mrr_in_batch"}
        or term.k is None
        or term.k <= 0
        for term in reward_terms
    ):
        raise ValueError(
            "Cross-query document gradients require in-batch nDCG/MRR with a positive cutoff"
        )
    if reward_cross_device_negatives and rollout_seed is None:
        raise ValueError(
            "Cross-device document policies require an explicit rollout_seed for independent rank-local actions"
        )


def validate_reward_cross_device_negatives(
    enabled,
    *,
    reward_terms,
    policy: _PolicyConstraints,
):
    if not enabled:
        return
    if (
        policy.rollout != "product"
        or policy.in_batch_use_sampled_documents
        or policy.document_advantage_baseline != "shared"
        or tuple(map(tuple, policy.action_components))
        != (("query",), ("positive", "negative"))
    ):
        raise ValueError(
            "Cross-device reward negatives require static joint product rollouts with fixed cross documents"
        )
    if not reward_terms or any(
        term.type not in {"ndcg_in_batch", "mrr_in_batch"}
        or not term.ndcg_in_batch_include_negatives
        or term.k is None
        or term.k <= 0
        for term in reward_terms
    ):
        raise ValueError(
            "Cross-device reward negatives require in-batch nDCG/MRR with all candidates and a positive cutoff"
        )


def validate_gradient_estimator(
    mode,
    policy: _PolicyConstraints,
):
    if mode not in SUPPORTED_GRADIENT_ESTIMATORS:
        raise ValueError(f"Unsupported gradient_estimator: {mode!r}")
    if mode == "score_function":
        return
    requirements = {
        "joint query and full-document actions": (
            len(policy.action_components) == 2
            and ("query",) in policy.action_components
            and any(
                set(group) == {"positive", "negative"}
                for group in policy.action_components
            )
        ),
        "sampling_law='vmf'": policy.sampling_law == "vmf",
        "sigma_learnable=false": not policy.sigma_learnable,
        "rollout='product'": policy.rollout == "product",
        "advantage_baseline='leave_one_out'": policy.advantage_baseline
        == "leave_one_out",
        "advantage_norm='none'": policy.advantage_norm == "none",
        "reward_combine='sum'": policy.reward_combine == "sum",
        "in_batch_use_sampled_documents=false": not policy.in_batch_use_sampled_documents,
        "document_advantage_baseline='shared'": policy.document_advantage_baseline
        == "shared",
        "document_log_prob_reduction='sum'": policy.document_log_prob_reduction
        == "sum",
    }
    missing = [name for name, valid in requirements.items() if not valid]
    if missing:
        raise ValueError(
            "gradient_estimator='conditional_projection' requires " + ", ".join(missing)
        )


def validate_document_advantage_baseline(
    mode,
    policy: _PolicyConstraints,
):
    """Limit the first factorized baseline to the estimator it was derived for."""
    if mode not in SUPPORTED_DOCUMENT_ADVANTAGE_BASELINES:
        raise ValueError(f"Unsupported document_advantage_baseline: {mode!r}")
    if mode == "shared":
        return
    requirements = {
        "a joint positive/negative document action group": any(
            set(group) == {"positive", "negative"} for group in policy.action_components
        ),
        "sampling_law='vmf'": policy.sampling_law == "vmf",
        # Per-document differences do not sum to zero; the dropped vMF normalizer
        # would then contribute a missing gradient if kappa were learnable.
        "sigma_learnable=false": not policy.sigma_learnable,
        "rollout='product'": policy.rollout == "product",
        "advantage_baseline='leave_one_out'": policy.advantage_baseline
        == "leave_one_out",
        "advantage_norm='none'": policy.advantage_norm == "none",
        "reward_combine='sum'": policy.reward_combine == "sum",
        "in_batch_use_sampled_documents=false": not policy.in_batch_use_sampled_documents,
    }
    missing = [name for name, valid in requirements.items() if not valid]
    if missing:
        raise ValueError(
            "document_advantage_baseline='counterfactual' requires "
            + ", ".join(missing)
        )


def normalize_advantage_norm_mode(mode) -> str:
    """Map legacy bool values (and their YAML/CLI string forms) onto the mode names."""
    if isinstance(mode, bool):
        return "per_component" if mode else "none"
    if isinstance(mode, str):
        lowered = mode.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return "per_component"
        if lowered in {"false", "0", "no"}:
            return "none"
        if lowered in SUPPORTED_ADVANTAGE_NORM_MODES:
            return lowered
    raise ValueError(
        f"Unsupported advantage_norm mode: {mode!r}. "
        f"Expected one of {SUPPORTED_ADVANTAGE_NORM_MODES} (or a legacy bool)."
    )


def normalize_action_components(action_components) -> tuple[tuple[str, ...], ...]:
    if isinstance(action_components, str):
        groups = []
        for raw_group in action_components.split(";"):
            raw_group = raw_group.strip()
            if not raw_group:
                continue
            groups.append(
                [
                    component.strip()
                    for component in raw_group.split(",")
                    if component.strip()
                ]
            )
    elif isinstance(action_components, Sequence):
        groups = action_components
    else:
        raise ValueError(
            "action_components must be a semicolon string or a list of component groups"
        )

    normalized_groups: list[tuple[str, ...]] = []
    seen_components: set[str] = set()
    for group in groups:
        if isinstance(group, str):
            components = tuple(
                component.strip() for component in group.split(",") if component.strip()
            )
        elif isinstance(group, Sequence):
            components = tuple(
                str(component).strip() for component in group if str(component).strip()
            )
        else:
            raise ValueError(
                "Each action component group must be a string or a list of strings"
            )

        if not components:
            raise ValueError("action component groups must not be empty")
        invalid_components = sorted(set(components) - SUPPORTED_ACTION_COMPONENTS)
        if invalid_components:
            raise ValueError(
                f"Unsupported action component(s): {', '.join(invalid_components)}"
            )
        if "query" in components and len(components) > 1:
            raise ValueError("query must be its own action component group")
        if len(set(components)) != len(components):
            raise ValueError(f"Duplicate component in action group: {components}")
        duplicate_components = sorted(seen_components.intersection(components))
        if duplicate_components:
            raise ValueError(
                f"Duplicate action component(s): {', '.join(duplicate_components)}"
            )

        seen_components.update(components)
        normalized_groups.append(components)

    if not normalized_groups:
        raise ValueError("At least one action component group must be configured")

    return tuple(normalized_groups)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to a pretrained model or a Hugging Face model ID"}
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pinned revision shared by the backbone, config and tokenizer"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional config path if it differs from model_name_or_path"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional tokenizer path if it differs from model_name_or_path"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory used to cache downloaded model files"},
    )
    pooling_method: str = field(
        default="last",
        metadata={"help": "Embedding pooling rule: last, mean, or cls"},
    )
    padding_side: str = field(
        default="left",
        metadata={"help": "Tokenizer padding side used by this embedding checkpoint"},
    )
    append_token: str = field(
        default="pad",
        metadata={
            "help": (
                "Terminal token appended after truncating raw content IDs: none, eos, or pad. "
                "Explicit terminals disable automatic special tokens; none uses the native tokenizer."
            )
        },
    )
    query_prompt_template: str = field(
        default="Instruct: {task_description}\nQuery:{query}",
        metadata={
            "help": (
                "Query formatting template. Supports {text}/{query} and {task_description}."
            )
        },
    )
    document_prompt_template: str = field(
        default="{document}",
        metadata={"help": "Document formatting template. Supports {text}/{document}."},
    )
    embedding_max_length: int = field(
        default=8192,
        metadata={
            "help": (
                "Maximum sequence length supported by the embedding checkpoint. Training "
                "query/document limits are clamped to this value."
            )
        },
    )

    def __post_init__(self) -> None:
        self.pooling_method = self.pooling_method.strip().lower()
        self.padding_side = self.padding_side.strip().lower()
        self.append_token = self.append_token.strip().lower()
        if self.embedding_max_length <= 0:
            raise ValueError(
                f"embedding_max_length must be positive, got {self.embedding_max_length}"
            )
        validate_embedding_protocol(
            pooling_method=self.pooling_method,
            padding_side=self.padding_side,
            append_token=self.append_token,
            query_prompt_template=self.query_prompt_template,
            document_prompt_template=self.document_prompt_template,
        )


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to the embedding training dataset"})
    per_dataset_max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of samples to keep from each source dataset. Use null to keep all samples."
        },
    )
    q_max_len: int = field(
        default=512,
        metadata={"help": "Maximum token length for query inputs"},
    )
    d_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum token length for document inputs"},
    )
    relevance_scheme: str = field(
        default="binary",
        metadata={"help": "Relevance labels: binary or graded"},
    )
    slate_size: int = field(
        default=8,
        metadata={
            "help": (
                "Maximum candidate documents per sample (1 positive + up to "
                "slate_size-1 negatives). Shorter samples are padded and masked per batch."
            )
        },
    )
    file_glob: str = field(
        default="*_len-0-500.jsonl",
        metadata={
            "help": (
                "Filename glob(s) used to pick which length bucket(s) to read from each "
                "source subdirectory when data_path is a directory. Accepts a "
                "comma-separated list to mix buckets, e.g. "
                "'*_len-0-500.jsonl,*_len-500-1000.jsonl'; matches are unioned and "
                "de-duplicated."
            )
        },
    )
    batch_per_length_bucket: bool = field(
        default=False,
        metadata={
            "help": (
                "When file_glob matches several length buckets, batch each bucket "
                "separately so every micro-batch holds documents of one length range "
                "(reduces padding waste). The per-source training cap "
                "(per_dataset_max_samples) still applies per source, not per bucket. "
                "Off by default, keeping single-bucket runs unchanged."
            )
        },
    )
    include_sources: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated source subdirectory names to include (e.g. "
                "'MSMARCO,NQ,HotpotQA'). None includes every subdirectory under data_path."
            )
        },
    )
    index_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory to cache the per-file byte-offset index used for lazy loading. "
                "None writes a sidecar '<file>.reler_idx.json' next to each data file."
            )
        },
    )

    def __post_init__(self) -> None:
        if self.relevance_scheme not in {"binary", "graded"}:
            raise ValueError(f"Unsupported relevance_scheme: {self.relevance_scheme}")
        if self.slate_size < 2:
            raise ValueError(f"slate_size must be >= 2, got {self.slate_size}")


@dataclass(frozen=True)
class FrozenCorpusArguments:
    """Location of immutable document embeddings."""

    index_dir: str = field(
        metadata={"help": "Frozen document-vector directory built by reler-index"},
    )

    def __post_init__(self) -> None:
        if not self.index_dir or not self.index_dir.strip():
            raise ValueError("index_dir must be a non-empty path")


@dataclass
class TrainingArguments(HFTrainingArguments):
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Allow training into a non-empty output directory"},
    )


@dataclass
class BaselineArguments:
    """Objective settings for supervised post-training controls."""

    baseline_loss: str = field(
        default="infonce",
        metadata={"help": "Supervised objective: infonce, ranknet, or lambdaloss"},
    )
    baseline_temperature: float = field(
        default=0.03,
        metadata={
            "help": (
                "Temperature applied to cosine scores before the supervised loss. "
                "Used by InfoNCE and RankNet."
            )
        },
    )
    baseline_ndcg_k: int = field(
        default=10,
        metadata={"help": "nDCG cutoff used by the LambdaRank LambdaLoss variant"},
    )
    lambdaloss_sigma: float = field(
        default=1.0,
        metadata={"help": "Pairwise logistic scale used by LambdaLoss"},
    )
    baseline_use_in_batch_negatives: bool = field(
        default=False,
        metadata={
            "help": (
                "Append the other samples' positive documents to each query's "
                "candidate pool. Their embeddings are detached in the cross-query "
                "scores, matching the frozen in-batch candidates used by RL."
            )
        },
    )
    baseline_in_batch_include_negatives: bool = field(
        default=False,
        metadata={
            "help": "Reuse all other queries' candidates, rather than only representative positives"
        },
    )
    baseline_cross_device_negatives: bool = field(
        default=False,
        metadata={
            "help": "Gather cross-query candidates across the data-parallel process group"
        },
    )
    baseline_detach_in_batch_documents: bool = field(
        default=True,
        metadata={"help": "Stop document gradients from cross-query negative scores"},
    )

    @property
    def extended_negative_pool(self) -> bool:
        return (
            self.baseline_in_batch_include_negatives
            or self.baseline_cross_device_negatives
            or not self.baseline_detach_in_batch_documents
        )

    def __post_init__(self) -> None:
        self.baseline_loss = self.baseline_loss.strip().lower()
        if self.baseline_loss not in SUPPORTED_BASELINE_LOSSES:
            raise ValueError(
                f"Unsupported baseline_loss: {self.baseline_loss!r}. "
                f"Expected one of {SUPPORTED_BASELINE_LOSSES}."
            )
        if self.baseline_temperature <= 0:
            raise ValueError(
                "baseline_temperature must be positive, "
                f"got {self.baseline_temperature}"
            )
        if self.extended_negative_pool and (
            self.baseline_loss != "infonce" or not self.baseline_use_in_batch_negatives
        ):
            raise ValueError(
                "Extended baseline negatives require InfoNCE and baseline_use_in_batch_negatives=true"
            )
        if self.baseline_ndcg_k <= 0:
            raise ValueError(
                f"baseline_ndcg_k must be positive, got {self.baseline_ndcg_k}"
            )
        if self.lambdaloss_sigma <= 0:
            raise ValueError(
                f"lambdaloss_sigma must be positive, got {self.lambdaloss_sigma}"
            )


@dataclass
class LoraArguments:
    lora_enabled: bool = False
    lora_path: Optional[str] = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    )
    lora_bias: str = "none"


@dataclass(frozen=True)
class RLArguments:
    gradient_estimator: str = field(
        default="score_function",
        metadata={
            "help": "score_function or conditional_projection (joint static vMF product rollouts)"
        },
    )
    rollout_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "Independent action-sampling seed; None preserves the legacy global RNG"
        },
    )
    action_components: str = field(
        default="query",
        metadata={
            "help": (
                "GRPO action component groups. YAML may use nested lists, e.g. "
                "[[query], [positive, negative]]. CLI may use 'query;positive,negative'."
            )
        },
    )
    group_size: int = field(
        default=8,
        metadata={"help": "Number of sampled actions per input"},
    )
    sigma: float = field(
        default=0.05,
        metadata={
            "help": (
                "Exploration scale; the vMF policy concentration is kappa = 1/sigma^2. "
                "Ignored when kappa is set explicitly."
            )
        },
    )
    kappa: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "vMF concentration override (unless target_alignment is set). sigma = 1/sqrt(kappa). "
                "E.g. kappa=755 matches the sampling concentration (mean cosine 0.53 at d=1024) "
                "of the legacy projected-Gaussian sampler with sigma=0.05."
            )
        },
    )
    target_alignment: Optional[float] = field(
        default=None,
        metadata={
            "help": "Target vMF mean cosine; overrides kappa using the actual embedding dimension"
        },
    )
    final_alignment: Optional[float] = field(
        default=None,
        metadata={"help": "Final mean cosine for linear exploration shrinkage"},
    )
    exploration_schedule: str = field(
        default="fixed",
        metadata={"help": "fixed or linear, indexed by optimizer steps"},
    )
    document_log_prob_reduction: str = field(
        default="sum",
        metadata={
            "help": "sum is the joint policy density; mean applies legacy 1/n role weighting"
        },
    )
    document_advantage_baseline: str = field(
        default="shared",
        metadata={
            "help": "shared uses the component baseline; counterfactual replaces one document action with its unit mean direction"
        },
    )
    sigma_learnable: bool = field(
        default=False,
        metadata={"help": "Learn a global sigma scalar for GRPO"},
    )
    sigma_min: float = field(
        default=1e-3,
        metadata={
            "help": "Lower clamp for learnable sigma (prevents exploration collapse)"
        },
    )
    sigma_max: float = field(
        default=0.5,
        metadata={"help": "Upper clamp for learnable sigma"},
    )
    reward_type: str = field(
        default="ndcg",
        metadata={
            "help": (
                "Reward type: ndcg, ndcg_in_batch, top_weighted_pairwise, rbo, "
                "contrastive, infonce, mrr, or mrr_in_batch. Used when "
                "reward_terms is empty, and as the per-term default elsewhere."
            )
        },
    )
    reward_terms: str = field(
        default="",
        metadata={
            "help": (
                "Additive reward terms. Empty falls back to the single reward_type. YAML may "
                "use a list of mappings, e.g. [{type: ndcg_in_batch, weight: 1.0, k: 16}, "
                "{type: contrastive, weight: 0.5, in_batch_negatives: true}]. CLI may use "
                "'ndcg_in_batch:1.0,k=16;contrastive:0.5,in_batch_negatives=true'. Unset "
                "per-term fields inherit reward_ndcg_k / reward_rbo_p / "
                "contrastive_temperature / the "
                "in-batch flags below."
            )
        },
    )
    reward_combine: str = field(
        default="sum",
        metadata={
            "help": (
                "How multiple reward terms are combined. 'sum' adds the raw rewards and takes "
                "one advantage, preserving each term's true effect size (so the effective "
                "mixing ratio is weight x within-group std, not the weight alone). "
                "'normalized_sum' standardizes each term's advantage separately before the "
                "weighted sum, making the weights scale-free -- use it whenever bounded "
                "ranking rewards are mixed with contrastive margins."
            )
        },
    )
    reward_ndcg_k: int = field(
        default=10,
        metadata={"help": "Default cutoff for ranking reward terms"},
    )
    reward_rbo_p: float = field(
        default=0.9,
        metadata={
            "help": (
                "Default RBO persistence in [0, 1); larger values spread more reward "
                "weight across deeper prefixes"
            )
        },
    )
    ndcg_in_batch_include_negatives: bool = field(
        default=False,
        metadata={
            "help": "Append all candidates from other samples, not only positives, for ndcg_in_batch"
        },
    )
    reward_cross_device_negatives: bool = field(
        default=False,
        metadata={
            "help": "Use the Strong CL cross-device document pool for static joint ranking RL"
        },
    )
    reward_shortlist_pool_source: str = field(
        default="cross_device_all",
        metadata={
            "help": "Shortlist source: cross_device_all, local_all, or cross_device_representatives"
        },
    )
    reward_shortlist_count: int = field(
        default=0,
        metadata={
            "help": "Number of separately rewarded/projected cross-pool shortlists; 0 disables"
        },
    )
    reward_shortlist_size: int = field(
        default=15,
        metadata={
            "help": "Cross-query negatives per shortlist; 0 keeps only own candidates (count > 0)"
        },
    )
    reward_shortlist_hard_count: int = field(
        default=8,
        metadata={
            "help": "Negatives per shortlist drawn from the highest-scoring stratum; 0 is uniform"
        },
    )
    reward_shortlist_hard_pool_size: int = field(
        default=64,
        metadata={
            "help": "Size of the high-score stratum, ranked using detached means"
        },
    )
    reward_shortlist_binary_weight: float = field(
        default=0.0,
        metadata={
            "help": "Mix original-positive binary nDCG into shortlist nDCG: (1-alpha)*base + alpha*binary"
        },
    )
    reward_shortlist_pairwise_coef: float = field(
        default=0.0,
        metadata={
            "help": "Add original-positive pair rewards with per-pair RLOO, using the selected gradient_estimator"
        },
    )
    cross_query_document_gradients: bool = field(
        default=False,
        metadata={
            "help": "Share sampled document actions across queries and accumulate all reward gradients; supports local and cross-device pools"
        },
    )
    contrastive_use_in_batch_negatives: bool = field(
        default=False,
        metadata={
            "help": "Use positives from other samples as extra negatives for contrastive reward"
        },
    )
    contrastive_temperature: float = field(
        default=0.03,
        metadata={"help": "Temperature used by the contrastive/infonce reward"},
    )
    aux_infonce_coef: float = field(
        default=0.0,
        metadata={
            "help": "Weight of direct multi-positive InfoNCE on unperturbed embeddings; 0 disables it"
        },
    )
    aux_infonce_temperature: float = field(
        default=0.03,
        metadata={
            "help": "Temperature of the direct InfoNCE auxiliary loss, independent of reward temperature"
        },
    )
    aux_infonce_use_in_batch_negatives: bool = field(
        default=False,
        metadata={
            "help": "Append masked, detached cross-query representatives to the auxiliary InfoNCE candidates"
        },
    )
    aux_infonce_strong_negatives: bool = field(
        default=False,
        metadata={
            "help": "Use all cross-device candidates and full document gradients within the RL action scope"
        },
    )
    advantage_norm: str = field(
        default="none",
        metadata={
            "help": (
                "Advantage normalization: 'per_component' divides each component's group by its own "
                "std (legacy true), 'shared' divides all components by the per-sample std of the raw "
                "reward tensor (preserves relative effect sizes between components), 'none' only "
                "centers (legacy false)."
            )
        },
    )
    sampling_law: str = field(
        default="vmf",
        metadata={
            "help": (
                "Law the actions are drawn from. 'vmf' samples exactly (Wood's rejection "
                "sampler). 'gaussian' is the projected-Gaussian shortcut e = normalize(h + "
                "sigma*eps), which is scored under the vMF log-density it was NOT drawn from; "
                "it exists as an ablation of sampling fidelity, not as a supported mode."
            )
        },
    )
    rollout: str = field(
        default="product",
        metadata={
            "help": (
                "'product' evaluates the full cross product of per-component samples (one "
                "reward-tensor axis per sampled component). 'diagonal' evaluates only the paired "
                "entries r^(g,g,...,g), so every component shares one group index and one "
                "advantage vector."
            )
        },
    )
    frozen_doc_rescale: bool = field(
        default=True,
        metadata={
            "help": (
                "Rescale frozen-document score tables by the policy's mean resultant length so "
                "frozen and sampled candidates share a score scale. Disabling it reproduces the "
                "reward-collapse failure mode and exists only as an ablation."
            )
        },
    )
    advantage_baseline: str = field(
        default="leave_one_out",
        metadata={
            "help": (
                "Baseline: 'leave_one_out' excludes the current action; 'group' is the group mean; "
                "'ema' is a single global exponential-moving-average scalar, which reduces the "
                "method to REINFORCE with a running baseline and exists as an ablation."
            )
        },
    )
    advantage_baseline_momentum: float = field(
        default=0.99,
        metadata={
            "help": "Momentum of the running baseline when advantage_baseline='ema'"
        },
    )
    in_batch_use_sampled_documents: bool = field(
        default=False,
        metadata={
            "help": (
                "Legacy behavior: score in-batch candidates with their sampled (perturbed) embeddings, "
                "which leaks other samples' perturbations into each sample's advantages via the shared "
                "group index. Default False scores them with detached mean embeddings so per-sample "
                "credit assignment stays exact."
                " Use cross_query_document_gradients for shared actions with complete document gradients."
            )
        },
    )
    kl_coef: float = field(
        default=0.0,
        metadata={
            "help": (
                "KL penalty coefficient between the adapter-enabled policy and the base "
                "(LoRA-disabled) reference. 0 disables the KL term. Requires a PEFT/LoRA model."
            )
        },
    )

    def __post_init__(self) -> None:
        self._normalize_and_validate_policy()
        self._normalize_reward_options()
        policy = _PolicyConstraints.from_arguments(self)
        self._normalize_and_validate_rewards(policy)
        self._validate_special_objectives(policy)
        self._validate_final_compatibility()

    def _normalize_and_validate_policy(self) -> None:
        from reler.objectives.policy_math import validate_exploration
        from reler.objectives.rollout_rng import validate_rollout_seed

        validate_aux_infonce(self.aux_infonce_coef, self.aux_infonce_temperature)
        validate_rollout_seed(self.rollout_seed)
        validate_exploration(
            self.target_alignment, self.final_alignment, self.exploration_schedule
        )
        if self.document_log_prob_reduction not in {"sum", "mean"}:
            raise ValueError("document_log_prob_reduction must be sum or mean")
        if self.target_alignment is not None and (
            self.sigma_learnable or self.sampling_law != "vmf"
        ):
            raise ValueError(
                "Alignment-based exploration requires vMF with sigma_learnable=false"
            )
        object.__setattr__(
            self,
            "action_components",
            normalize_action_components(self.action_components),
        )
        object.__setattr__(
            self,
            "advantage_norm",
            normalize_advantage_norm_mode(self.advantage_norm),
        )
        if self.sampling_law not in SUPPORTED_SAMPLING_LAWS:
            raise ValueError(
                f"Unsupported sampling_law: {self.sampling_law!r}. "
                f"Expected one of {SUPPORTED_SAMPLING_LAWS}."
            )
        if self.rollout not in SUPPORTED_ROLLOUTS:
            raise ValueError(
                f"Unsupported rollout: {self.rollout!r}. Expected one of {SUPPORTED_ROLLOUTS}."
            )
        if self.advantage_baseline not in SUPPORTED_ADVANTAGE_BASELINES:
            raise ValueError(
                f"Unsupported advantage_baseline: {self.advantage_baseline!r}. "
                f"Expected one of {SUPPORTED_ADVANTAGE_BASELINES}."
            )
        if not 0.0 <= self.advantage_baseline_momentum < 1.0:
            raise ValueError(
                f"advantage_baseline_momentum must lie in [0, 1), got {self.advantage_baseline_momentum}"
            )
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < 2
        ):
            raise ValueError("group_size must be an integer of at least 2 for GRPO")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if self.kappa is not None:
            if not math.isfinite(self.kappa) or self.kappa <= 0:
                raise ValueError(f"kappa must be positive, got {self.kappa}")
            object.__setattr__(self, "sigma", self.kappa**-0.5)
        if self.sigma_learnable:
            if not 0 < self.sigma_min < self.sigma_max:
                raise ValueError(
                    "Expected 0 < sigma_min < sigma_max, "
                    f"got [{self.sigma_min}, {self.sigma_max}]"
                )
            if not self.sigma_min <= self.sigma <= self.sigma_max:
                raise ValueError(
                    f"Initial sigma {self.sigma} must lie within the learnable bounds "
                    f"[{self.sigma_min}, {self.sigma_max}]"
                )
        if self.kl_coef < 0:
            raise ValueError(f"kl_coef must be non-negative, got {self.kl_coef}")

    def _normalize_reward_options(self) -> None:
        object.__setattr__(self, "reward_type", str(self.reward_type).strip().lower())
        if self.reward_type not in SUPPORTED_REWARD_TYPES:
            raise ValueError(
                f"Unsupported reward type: {self.reward_type}. "
                f"Supported types: {sorted(SUPPORTED_REWARD_TYPES)}"
            )
        if self.contrastive_temperature <= 0:
            raise ValueError(
                f"contrastive_temperature must be positive, got {self.contrastive_temperature}"
            )
        if not 0.0 <= self.reward_rbo_p < 1.0:
            raise ValueError(
                f"reward_rbo_p must lie in [0, 1), got {self.reward_rbo_p}"
            )
        object.__setattr__(
            self,
            "reward_combine",
            normalize_reward_combine_mode(self.reward_combine),
        )

    def _normalize_and_validate_rewards(self, policy: _PolicyConstraints) -> None:
        validate_document_advantage_baseline(
            self.document_advantage_baseline,
            policy,
        )
        validate_gradient_estimator(self.gradient_estimator, policy)
        # An empty spec means "single term from reward_type", which resolves to exactly the
        # arguments the pre-combination code passed, so legacy configs are untouched.
        object.__setattr__(
            self,
            "reward_terms",
            normalize_reward_terms(
                self.reward_terms if self.reward_terms else self.reward_type,
                default_k=self.reward_ndcg_k,
                default_temperature=self.contrastive_temperature,
                default_rbo_p=self.reward_rbo_p,
                default_ndcg_in_batch_include_negatives=self.ndcg_in_batch_include_negatives,
                default_contrastive_use_in_batch_negatives=self.contrastive_use_in_batch_negatives,
            ),
        )
        if (
            len(self.reward_terms) > 1
            and self.reward_combine == "normalized_sum"
            and self.advantage_norm == "none"
        ):
            raise ValueError(
                "reward_combine='normalized_sum' requires advantage_norm="
                "'per_component' or 'shared' so each reward term is actually normalized"
            )

    def _validate_special_objectives(self, policy: _PolicyConstraints) -> None:
        validate_reward_cross_device_negatives(
            self.reward_cross_device_negatives or self.reward_shortlist_count > 0,
            reward_terms=self.reward_terms,
            policy=policy,
        )
        validate_cross_query_document_gradients(
            self.cross_query_document_gradients,
            reward_terms=self.reward_terms,
            policy=policy,
            reward_cross_device_negatives=self.reward_cross_device_negatives,
            rollout_seed=self.rollout_seed,
        )
        validate_reward_shortlists(
            self.reward_shortlist_count,
            self.reward_shortlist_size,
            self.reward_shortlist_hard_count,
            self.reward_shortlist_hard_pool_size,
            pool_source=self.reward_shortlist_pool_source,
            reward_cross_device_negatives=self.reward_cross_device_negatives,
            cross_query_document_gradients=self.cross_query_document_gradients,
            rollout_seed=self.rollout_seed,
            policy=policy,
        )
        validate_shortlist_objectives(
            self.reward_shortlist_count,
            self.reward_terms,
            self.reward_shortlist_binary_weight,
            self.reward_shortlist_pairwise_coef,
            self.gradient_estimator,
        )

    def _validate_final_compatibility(self) -> None:
        if (
            len(self.reward_terms) > 1
            and self.reward_combine == "sum"
            and reward_terms_mix_scales(self.reward_terms)
        ):
            logger.warning(
                "reward_combine='sum' mixes bounded ranking rewards with unbounded contrastive "
                "margins (%s). Advantages are divided by the group std of the COMBINED reward, "
                "so each term contributes in proportion to weight x its own within-group std, "
                "not to its weight alone -- watch reward/<term>/group_std to see the ratio you "
                "actually got. Use reward_combine='normalized_sum' for weights that mean what "
                "they say.",
                ", ".join(
                    f"{term.name}(w={term.weight})" for term in self.reward_terms
                ),
            )
        if (
            self.advantage_baseline == "ema"
            and self.reward_combine == "normalized_sum"
            and len(self.reward_terms) > 1
        ):
            raise ValueError(
                "advantage_baseline='ema' is incompatible with reward_combine='normalized_sum' "
                "for multiple terms: the running baseline is a single global scalar and cannot "
                "track per-term reward scales. Use reward_combine='sum' or the group baseline."
            )
        if self.sigma_learnable and self.advantage_baseline == "ema":
            raise ValueError(
                "advantage_baseline='ema' is incompatible with sigma_learnable=True: "
                "the vMF normalizer only cancels under group-centered advantages."
            )

    def resolve(self) -> "RLArguments":
        """Return the normalized immutable configuration consumed by the policy."""
        return self

    @classmethod
    def from_arguments(cls, arguments: "RLArguments") -> "RLArguments":
        if isinstance(arguments, cls):
            return arguments
        return cls(**{item.name: getattr(arguments, item.name) for item in fields(cls)})

    def as_checkpoint_dict(self, source=None) -> dict:
        """Serialize every algorithm setting, optionally reading live overrides."""
        source = self if source is None else source
        return {
            item.name: _checkpoint_value(getattr(source, item.name))
            for item in fields(self)
        }

    @property
    def sample_query(self) -> bool:
        return ("query",) in self.action_components

    @property
    def sample_positive(self) -> bool:
        return any("positive" in group for group in self.action_components)

    @property
    def sample_negative(self) -> bool:
        return any("negative" in group for group in self.action_components)


# Backward-compatible public name for the resolved policy configuration. Keeping
# one immutable dataclass prevents the CLI schema and runtime spec from drifting.
GRPOSpec = RLArguments
