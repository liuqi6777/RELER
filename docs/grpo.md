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

Set `reward_shortlist_count: 1`, `reward_shortlist_size: 0`, and
`reward_shortlist_hard_count: 0` to retain only each query's own candidates.
The shortlist objective stays active, including pairwise feedback. It skips
cross-query document gathering and shortlist sampling, while preserving the
global query mean for uneven distributed batches. `reward_shortlist_count: 0`
instead disables this objective and uses the ordinary reward path.

`reward_shortlist_pairwise_coef` adds binary positive-versus-negative pair
rewards using the annotated positive mask, independently of the teacher grades.
The same `gradient_estimator` selects CMP or unprojected RLOO for both the
listwise and pairwise terms. Pairwise document credit applies only to the pair's
trainable endpoints; fixed negatives receive no gradient. Both estimators share
pair selection, tie handling, per-pair leave-one-out weights, and per-query
normalization. The unprojected path retains the same fixed-vMF product-policy
constraints as CMP.

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

## Training configuration

`configs/examples/reler.yaml` composes the existing model, dataset, training,
policy, and reward fragments for joint own-candidate training: graded nDCG@10
plus pairwise weight 0.5, CMP, 64 actions per side, fixed alignment 0.70, and
113 optimizer steps. The inherited batch size is 16 per GPU, giving a global
batch of 128 with eight GPUs and no accumulation. Supply prepared training data
in `embedding_candidates_v2` format with teacher labels and positive identities
through `--data_path`; see the [data contract](../README.md#training-data).

```bash
NPROC_PER_NODE=8 bash scripts/run_grpo.sh configs/examples/reler.yaml \
  --data_path /path/to/train.ready.jsonl

# Same reward and sampled policy, with CMP disabled in both reward components.
NPROC_PER_NODE=8 bash scripts/run_grpo.sh configs/examples/reler.yaml \
  --data_path /path/to/train.ready.jsonl \
  --gradient_estimator score_function \
  --output_dir checkpoints/reler-rloo
```

Set `--reward_shortlist_pairwise_coef 0` for the listwise-only objective, or
override `--target_alignment` to change exploration. Set `--seed`, `--data_seed`,
and `--rollout_seed` together when changing the training seed. Model selection,
LoRA, step budget, and batch size remain ordinary configuration/CLI overrides.

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
