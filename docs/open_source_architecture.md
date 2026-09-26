# Open-source embedding branch architecture

This branch turns the research workspace into a small, auditable package for
training embedding models. The refactor is intentionally behavior-preserving:
it changes ownership and public interfaces before changing any numerical code.

## Release scope

The first open-source release includes:

- shared-weight query/document embedding training;
- prepared `embedding_candidates_v2` training data;
- InfoNCE, RankNet, and LambdaLoss supervised objectives;
- the current GRPO objective, including score-function and conditional-projection
  estimators, shortlist rewards, and cross-query document policies;
- checkpoint resume state and the embedding protocol sidecar;
- optional immutable-document lookup with query-only GRPO;
- fixed-E0 BRIGHT evaluation with sequential per-subset caches;
- standalone MTEB evaluation.

The first release does not include:

- RAG data preparation, generation, answer rewards, or RAG evaluation;
- dynamic full-corpus retrieval RL, corpus routers, or FAISS search;
- paper experiment matrices, cluster queues, result analysis, or audit artifacts;
- legacy artifact-verification orchestration.

This is a source-tree boundary, not a hidden feature flag: excluded research code
must not remain importable from the release package.

## Target layout

```text
src/reler/
  cli.py                     # one public training command with three modes
  config/
    arguments.py             # model, data, training, and objective schemas
    loader.py                # YAML composition and CLI overrides
  data/
    candidates.py            # shared prepared-record validation
    embedding.py             # lazy reads, source batching, and collation
    protocol.py              # text formatting, pooling, and checkpoint metadata
  fixed_corpus/
    build.py                 # immutable corpus encoder and directory builder
    index.py                 # validated lookup-only vector shards
    data.py                  # ordinal collator without document tokenization
    training.py              # query-only GRPO adapter
  objectives/
    contrastive.py           # supervised and auxiliary contrastive primitives
    cross_query.py           # cross-query document policy
    policy_math.py           # vMF schedules and advantage baselines
    precision.py             # FP32 ranking contraction
    projection.py            # conditional and pairwise projection estimators
    rewards.py               # retrieval reward implementations
    rollout_rng.py           # isolated rollout sampling stream
    shortlists.py            # shortlist sampling and reward contracts
  training/
    common.py                # backbone, tokenizer, LoRA, and dataset setup
    supervised.py            # InfoNCE, RankNet, and LambdaLoss entrypoint
    grpo.py                   # GRPO entrypoint
    grpo_model.py             # policy sampling and surrogate objective
    trainer.py                # Trainer integration and resume state
  evaluation/
    run_mteb.py              # standalone MTEB runner
    qwen3_embedding_model.py # protocol-aware MTEB adapter
    fixed_corpus.py          # fixed-E0 BRIGHT cache and query/document routing
    summary.py               # result summarizer
configs/
  examples/
    supervised.yaml
    grpo.yaml
tests/
  test_*.py
```

The final package obeys these dependency directions:

```text
cli -> training -> {config, data, objectives}
evaluation -> {config.loader, data.protocol}
fixed_corpus.training -> {training, config, data, fixed_corpus.index}
evaluation.fixed_corpus -> data.protocol
config.arguments -> {data.protocol, objective validators}
objectives -> {numerical helpers inside objectives}
data.embedding -> {data.candidates, data.protocol}
fixed_corpus.build -> {data.candidates, data.protocol}
```

`training/common.py` must not import a corpus index, RAG module, or experiment
runner. The optional fixed-corpus entrypoint depends on the ordinary training
stack, while ordinary training and objective modules remain unaware of it.

## Behavior contract

The following properties define algorithmic compatibility and must be covered by
regression tests before the old modules are removed.

1. Prepared candidate order, binary labels, teacher grades, and higher-is-better
   rank labels are preserved without conversion or candidate sampling.
2. The annotated positive used by contrastive learning remains independent of
   teacher grades. Known positives must still be excluded from negative pools.
3. Query/document prompt rendering, padding side, terminal-token insertion,
   truncation, pooling, and L2 normalization must not change.
4. Pooling and similarity scores remain FP32 under mixed precision.
5. Candidate padding is inert for every loss, reward, and gradient.
6. InfoNCE, RankNet, LambdaLoss, reward, advantage, score-function, and
   conditional-projection outputs and gradients match the pre-refactor code on
   fixed inputs and RNG states.
7. Rollout RNG streams and checkpointed exploration state preserve exact resume
   behavior.
8. `embedding_protocol.json` keeps its current fields and evaluation semantics.

Fixed-corpus training uses a deliberately small directory contract: vector shards,
the document-to-ordinal mapping, and `embedding_protocol.json` must agree. Prepared
text keys identify documents across sources; source-scoped IDs and known-document
sets also filter cross-query candidates without changing binary positive labels.

## Migration sequence

Each phase must leave the retained tests green. Deletions happen only after the
replacement path is exercised by those tests.

1. Record the test baseline and add focused numerical compatibility tests where
   coverage is indirect.
2. Extract shared training setup from the GRPO entrypoint without changing model,
   data, or objective implementations.
3. Introduce the package CLI and route both supervised and GRPO training through
   the same configuration loader and setup layer.
4. Keep immutable-corpus support behind its dedicated entrypoint and configuration
   slot; do not add corpus branches to the joint model or collator.
5. Remove RAG modules, configurations, scripts, and tests.
6. Remove paper experiment orchestration and legacy artifact gates.
7. Move the retained numerical modules under `reler`, replace compatibility
   imports, and reduce the public configuration surface.
8. Replace the research README and dependency list with install, data, training,
   evaluation, and reproducibility documentation for RELER.

## Release gates

The branch is ready only when all of the following are true:

- a clean environment can install the declared runtime and development extras;
- the documented InfoNCE and GRPO configurations parse without downloading a
  model or starting distributed training;
- retained unit and numerical regression tests pass with no RAG/frozen-corpus
  collection dependency;
- the ordinary training stack contains no imports of `rag` or `fixed_corpus`;
- immutable-corpus directory and protocol checks stay outside ordinary training;
- a tiny local model and tiny JSONL fixture complete one supervised update and one
  GRPO update, save a checkpoint, and resume;
- README commands and links are checked against the final tree.
