# GRPO objective

RELER applies GRPO directly in normalized embedding space. The joint policy can
sample a query action and a document-slate action, score their Cartesian product,
and optimize a retrieval reward without generating text.

## Default policy

`configs/grpo/default.yaml` uses:

- a vMF policy over unit embeddings;
- separate query and joint-document action groups;
- product rollouts;
- a leave-one-out group baseline;
- summed document log-probabilities;
- fixed exploration concentration;
- the score-function gradient estimator.

`group_size` controls the number of actions sampled for each component. `kappa`
is the vMF concentration; alternatively, `target_alignment` expresses exploration
as the expected cosine similarity between a policy mean and its sample.

## Estimators

The score-function estimator uses detached leave-one-out advantages. For a joint
document action, log-probabilities are summed across valid documents so the loss
matches the joint policy density.

`gradient_estimator: conditional_projection` selects the conditional-projection
estimator. RELER validates the assumptions required by its derivation: joint
query/document vMF product actions, fixed concentration, leave-one-out advantages,
no advantage normalization, and static candidate slates. Unsupported combinations
fail during configuration parsing.

## Rewards and candidate pools

Reward fragments under `configs/reward/` select MRR, nDCG, contrastive, RBO, or
pairwise ranking signals. Candidate masks and known-positive filters are applied
consistently to local, in-batch, and cross-device pools. Optional shortlists reduce
large pools while preserving the configured estimator contract.

The policy and reward implementations meet at `RewardEvaluator`: GRPO constructs
only the score tables declared by the evaluator's candidate-pool requirements, then
consumes generic weighted reward signals. Reward dispatch and raw weighting stay in
`objectives/rewards.py`; baselines, normalization, and policy-gradient estimators stay
in the RL layer. Adding a score-based reward therefore does not require changing the
GRPO rollout or loss orchestration.

`GRPOModel` only dispatches policy objectives. The regular product-action objective
stays with the model because it owns the shared rollout state; shortlist and
shared-document objectives live in `objectives/shortlists.py` and
`objectives/cross_query.py`. Each path returns the same `PolicyObjectiveResult`, so
the trainer is independent of candidate-pool-specific bookkeeping.

All reward ranking and similarity scoring is FP32. Padding candidates contribute
neither reward nor gradient.

## Reproducibility

`rollout_seed` creates independent per-rank, per-step random streams without
mutating the process-global RNG. Checkpoints store the exploration schedule,
estimator contract, shortlist contract, and rollout position so resumed training
continues from the same policy state.

`RLArguments` is normalized once and then used as the immutable runtime and
checkpoint configuration. The legacy `GRPOSpec` name aliases that same contract,
so CLI/YAML fields and checkpoint fields cannot drift between duplicate schemas.

The numerical contracts are covered by the conditional-projection, shortlist,
cross-query-policy, rollout, and score-precision tests in `tests/`.
