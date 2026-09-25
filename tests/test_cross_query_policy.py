"""Shared document policies: rewards, analytic credit and distributed gradients."""

import json
from datetime import timedelta

import pytest
import torch
import torch.nn.functional as F

from reler.config import RLArguments
from reler.objectives.contrastive import cross_document_mask
from reler.objectives.cross_query import (
    sampled_document_pool,
    sampled_pool_loss,
    sampled_pool_rewards,
)
from reler.objectives.policy_math import mean_alignment
from reler.objectives.rewards import (
    RewardEvaluator,
    compute_reward_from_scores,
    normalize_reward_terms,
)
from reler.training.grpo_model import GRPO, GRPOModel, _ActionComponent, sample_vmf
from reler.training.trainer import GRPOTrainer, restore_exploration_state


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def geometry(batch=3, width=3, dim=8, group=4):
    torch.manual_seed(107)
    source = F.normalize(torch.randn(batch, width + 1, dim), dim=-1)
    queries = sample_vmf(source[:, 0], 8.0, group)
    docs = (
        sample_vmf(source[:, 1:].reshape(-1, dim), 8.0, group)
        .reshape(batch, width, group, dim)
        .transpose(1, 2)
    )
    valid = torch.ones(batch, width, dtype=torch.bool)
    valid[-1, -1] = False
    docs = docs.masked_fill(~valid[:, None, :, None], 0)
    labels = torch.zeros(batch, width)
    labels[:, 0] = 3
    labels[0, 1] = 1
    metadata = [
        dict(
            keys=[f"d{i}-{m}" if valid[i, m] else None for m in range(width)],
            ids=[None] * width,
            source="s",
            known_ids=[],
            known_positive_keys=[],
        )
        for i in range(batch)
    ]
    if batch > 1:
        metadata[0]["known_positive_keys"] = ["d1-0"]
    return source, queries, docs, valid, labels, metadata


def full_rewards(q, docs, labels, valid, pool, all_mask, rep_mask, terms):
    own = torch.einsum(
        "bid,bjmd->bijm",
        F.normalize(q.float(), dim=-1),
        F.normalize(docs.float(), dim=-1),
    )
    cross = torch.einsum(
        "bid,jpd->bijp",
        F.normalize(q.float(), dim=-1),
        F.normalize(pool.float(), dim=-1),
    )
    width = docs.size(2)
    return {
        term.name: compute_reward_from_scores(
            own,
            labels,
            reward_type=term.type,
            k=term.k,
            candidate_mask=valid,
            ndcg_in_batch_include_negatives=term.ndcg_in_batch_include_negatives,
            in_batch_candidate_scores=cross.masked_fill(
                ~all_mask[:, None, None], -torch.inf
            ),
            in_batch_positive_scores=cross[..., ::width].masked_fill(
                ~rep_mask[:, None, None, ::width], -torch.inf
            ),
        )
        for term in terms
    }


@pytest.mark.parametrize(
    "all_candidates,cross_device", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("batch", [1, 3])
def test_streamed_rewards_preserve_candidates_ties_and_padding(
    all_candidates, cross_device, batch
):
    source, q, docs, valid, labels, metadata = geometry(batch=batch)
    # Equal actions create exact score ties across own and foreign candidates.
    docs[:, 1] = docs[:, 0]
    docs[0, :, 1] = docs[0, :, 0]
    pool, actions, all_mask, rep_mask, _ = sampled_document_pool(
        source[:, 1:],
        docs,
        valid,
        cross_device=cross_device,
        metadata=metadata,
    )
    terms = normalize_reward_terms(
        [
            dict(
                type="ndcg_in_batch",
                k=2,
                ndcg_in_batch_include_negatives=all_candidates,
            ),
            dict(
                type="mrr_in_batch", k=2, ndcg_in_batch_include_negatives=all_candidates
            ),
        ]
    )
    expected = full_rewards(q, docs, labels, valid, actions, all_mask, rep_mask, terms)
    evaluator = RewardEvaluator(terms)
    for chunk in (1, 3, 8):
        evaluation, _ = sampled_pool_rewards(
            evaluator,
            q,
            docs,
            actions,
            labels,
            valid,
            all_mask,
            rep_mask,
            document_chunk=chunk,
        )
        for name in expected:
            torch.testing.assert_close(
                evaluation.term_rewards[name], expected[name], rtol=0, atol=0
            )
    if batch > 1:
        assert not all_mask[0, 3]  # Known positive from another query.
    assert not all_mask[:, -1].any()  # Padding never becomes a negative.


def explicit_loss(means, q, docs, rewards, valid, foreign_mask, estimator, kappa=8.0):
    """Literal per-cell global policy score; CP projects each score separately."""
    batch, gq, gd = rewards.shape
    width = docs.size(2)
    coefficients = torch.zeros_like(means, dtype=torch.float64)
    for b in range(batch):
        slots = [(b, m) for m in range(width) if valid[b, m]]
        slots += [
            (p // width, p % width)
            for p in foreign_mask[b].nonzero(as_tuple=True)[0].tolist()
        ]
        for i in range(gq):
            for j in range(gd):
                aq = (
                    rewards[b, i, j]
                    - torch.cat((rewards[b, :i, j], rewards[b, i + 1 :, j])).mean()
                )
                ad = (
                    rewards[b, i, j]
                    - torch.cat((rewards[b, i, :j], rewards[b, i, j + 1 :])).mean()
                )
                query = q[b, i].double()
                if estimator == "conditional_projection":
                    columns = (
                        torch.stack(
                            [means[b, 0].detach(), *[docs[c, j, m] for c, m in slots]]
                        )
                        .double()
                        .T
                    )
                    query = columns @ torch.linalg.pinv(columns) @ query
                coefficients[b, 0] += aq * query
                for c, m in slots:
                    document = docs[c, j, m].double()
                    if estimator == "conditional_projection":
                        columns = (
                            torch.stack((means[c, m + 1].detach(), q[b, i])).double().T
                        )
                        document = columns @ torch.linalg.pinv(columns) @ document
                    coefficients[c, m + 1] += ad * document
    return -(means * coefficients.to(means.dtype)).sum() * (kappa / rewards.numel())


@pytest.mark.parametrize("estimator", ["score_function", "conditional_projection"])
@pytest.mark.parametrize(
    "all_candidates,cross_device", [(False, False), (True, False), (True, True)]
)
def test_encoder_gradient_matches_explicit_all_query_credit(
    estimator, all_candidates, cross_device
):
    source, q, docs, valid, labels, metadata = geometry()
    weight = torch.eye(source.size(-1), requires_grad=True)
    means = F.normalize(source @ weight.T, dim=-1)
    pool, actions, all_mask, rep_mask, _ = sampled_document_pool(
        means[:, 1:],
        docs.requires_grad_(),
        valid,
        cross_device=cross_device,
        metadata=metadata,
    )
    terms = normalize_reward_terms(
        "ndcg_in_batch",
        default_k=2,
        default_ndcg_in_batch_include_negatives=all_candidates,
    )
    rewards = full_rewards(
        q, docs.detach(), labels, valid, actions, all_mask, rep_mask, terms
    )[terms[0].name].detach()
    mask = all_mask if all_candidates else rep_mask
    loss, ranks = sampled_pool_loss(
        means[:, 0],
        means[:, 1:],
        pool,
        q.requires_grad_(),
        docs,
        actions,
        rewards.requires_grad_(),
        8.0,
        valid,
        mask,
        estimator=estimator,
    )
    actual = torch.autograd.grad(loss, weight, retain_graph=True)[0]
    expected = torch.autograd.grad(
        explicit_loss(
            means, q.detach(), docs.detach(), rewards.detach(), valid, mask, estimator
        ),
        weight,
    )[0]
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert actual.norm() > 0
    assert q.grad is None and docs.grad is None and rewards.grad is None
    if estimator == "conditional_projection":
        assert (ranks <= source.size(-1)).all()


def test_remote_document_gets_credit_from_another_query_only():
    source, q, docs, valid, _, metadata = geometry(batch=2, width=2, dim=9)
    metadata[0]["known_positive_keys"] = []
    raw = source.clone().requires_grad_()
    means = F.normalize(raw, dim=-1)
    pool, actions, mask, _, _ = sampled_document_pool(
        means[:, 1:], docs, valid, cross_device=False, metadata=metadata
    )
    # Only query 0 has a reward, depending on query 1's representative document.
    reward = torch.zeros(2, q.size(1), docs.size(1))
    reward[0] = q[0] @ docs[1, :, 0].T
    loss, _ = sampled_pool_loss(
        means[:, 0],
        means[:, 1:],
        pool,
        q,
        docs,
        actions,
        reward,
        8.0,
        valid,
        mask,
        estimator="conditional_projection",
    )
    loss.backward()
    assert raw.grad[1, 1].norm() > 0
    torch.testing.assert_close(raw.grad[1, 0], torch.zeros(9), rtol=0, atol=0)
    torch.testing.assert_close(raw.grad[1, 2], torch.zeros(9), rtol=0, atol=0)


@pytest.mark.parametrize("estimator", ["score_function", "conditional_projection"])
def test_mixed_candidate_terms_use_union_for_complete_document_credit(estimator):
    source, q, docs, valid, labels, metadata = geometry()
    raw = source.clone().requires_grad_()
    means = F.normalize(raw, dim=-1)
    pool, actions, all_mask, rep_mask, _ = sampled_document_pool(
        means[:, 1:], docs, valid, cross_device=False, metadata=metadata
    )
    terms = normalize_reward_terms(
        [
            dict(
                type="ndcg_in_batch",
                k=2,
                weight=0.7,
                ndcg_in_batch_include_negatives=True,
            ),
            dict(
                type="mrr_in_batch",
                k=3,
                weight=0.3,
                ndcg_in_batch_include_negatives=False,
            ),
        ]
    )
    evaluator = RewardEvaluator(terms)
    evaluation, _ = sampled_pool_rewards(
        evaluator, q, docs, actions, labels, valid, all_mask, rep_mask
    )
    expected = full_rewards(q, docs, labels, valid, actions, all_mask, rep_mask, terms)
    for name in expected:
        torch.testing.assert_close(
            evaluation.term_rewards[name], expected[name], rtol=0, atol=0
        )
    reward = evaluation.combined
    loss, _ = sampled_pool_loss(
        means[:, 0],
        means[:, 1:],
        pool,
        q,
        docs,
        actions,
        reward,
        8.0,
        valid,
        all_mask | rep_mask,
        estimator=estimator,
    )
    actual_gradient = torch.autograd.grad(loss, raw, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(
        explicit_loss(means, q, docs, reward, valid, all_mask | rep_mask, estimator),
        raw,
    )[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("estimator", ["score_function", "conditional_projection"])
def test_no_foreign_candidates_reduces_to_original_joint_estimator(estimator):
    source, q, docs, valid, labels, metadata = geometry(batch=1)
    raw = source.clone().requires_grad_()
    means = F.normalize(raw, dim=-1)
    original = make_head(estimator, cross_query_document_gradients=False)
    extended = make_head(estimator)
    reference, before, _, _ = head_loss(
        original, means, q, docs, valid, labels, metadata
    )
    loss, after, _, _ = head_loss(extended, means, q, docs, valid, labels, metadata)
    actual = torch.autograd.grad(loss, raw, retain_graph=True)[0]
    expected = torch.autograd.grad(reference, raw)[0]
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(
        before["reward_mean"], after["reward_mean"], atol=0, rtol=0
    )


@pytest.mark.parametrize("rank", [3, 7])
def test_common_pool_full_rank_shortcut_matches_per_cell_projection(rank):
    source, q, docs, valid, _, _ = geometry(batch=5, width=3, dim=7, group=3)
    docs[..., rank:] = 0
    docs = F.normalize(docs, dim=-1)
    # First two queries can use all documents from the other three queries.
    mask = torch.zeros(2, 15, dtype=torch.bool)
    mask[:, 6:] = valid[2:].reshape(-1)
    raw = source.clone().requires_grad_()
    means = F.normalize(raw, dim=-1)
    reward = torch.randn(2, 3, 3)
    loss, ranks = sampled_pool_loss(
        means[:2, 0],
        means[:2, 1:],
        means[:, 1:].reshape(-1, 7),
        q[:2],
        docs[:2],
        docs.transpose(0, 1).reshape(3, -1, 7),
        reward,
        8.0,
        valid[:2],
        mask,
        estimator="conditional_projection",
    )
    actual = torch.autograd.grad(loss, raw, retain_graph=True)[0]
    padded_reward = torch.cat((reward, torch.zeros(3, 3, 3)))
    padded_mask = torch.cat((mask, torch.zeros(3, 15, dtype=torch.bool)))
    reference = explicit_loss(
        means, q, docs, padded_reward, valid, padded_mask, "conditional_projection"
    ) * (5 / 2)
    expected = torch.autograd.grad(reference, raw)[0]
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=4e-6)
    assert (ranks == min(rank + 1, 7)).all()


def test_joint_policy_expected_gradient_matches_analytic_bilinear_reward():
    repeats, group, dim, kappa = 256, 8, 6, 8.0
    source, _, _, valid, _, metadata = geometry(batch=2, width=2, dim=dim, group=group)
    valid[:] = True
    metadata[0]["known_positive_keys"] = []
    metadata[1]["keys"][1] = "d1-1"
    draws = sample_vmf(source.reshape(-1, dim), kappa, repeats * group).reshape(
        2, 3, repeats, group, dim
    )
    raw = source.clone().requires_grad_()
    means = F.normalize(raw, dim=-1)
    coefficients = torch.tensor([[1.0, -0.2, -0.6, 0.3], [-0.4, 0.1, 1.0, -0.3]])
    gradients = []
    for draw in range(repeats):
        q, docs = draws[:, 0, draw], draws[:, 1:, draw].transpose(1, 2)
        pool, actions, mask, _, _ = sampled_document_pool(
            means[:, 1:], docs, valid, cross_device=False, metadata=metadata
        )
        reward = torch.einsum("bid,jpd,bp->bij", q, actions, coefficients)
        loss, _ = sampled_pool_loss(
            means[:, 0],
            means[:, 1:],
            pool,
            q,
            docs,
            actions,
            reward,
            kappa,
            valid,
            mask,
            estimator="conditional_projection",
        )
        gradients.append(torch.autograd.grad(loss, raw, retain_graph=True)[0])
    exact = (
        -(mean_alignment(dim, kappa) ** 2)
        * ((means[:, 0] @ means[:, 1:].reshape(-1, dim).T) * coefficients).sum()
        / 2
    )
    expected = torch.autograd.grad(exact, raw)[0]
    gradients = torch.stack(gradients).double()
    error = (gradients.mean(0) - expected).norm()
    sem = (gradients.var(0, unbiased=True).sum() / repeats).sqrt()
    assert error < 4 * sem
    assert error / expected.norm() < 0.08


def make_head(estimator="conditional_projection", **overrides):
    kwargs = dict(
        action_components="query;positive,negative",
        group_size=4,
        kappa=8.0,
        gradient_estimator=estimator,
        reward_type="ndcg_in_batch",
        reward_ndcg_k=2,
        cross_query_document_gradients=True,
        rollout_seed=42,
    )
    return GRPO(**dict(kwargs, **overrides))


def head_loss(head, means, q, docs, valid, labels, metadata):
    components = (
        _ActionComponent(
            role="query",
            name="query",
            rollout_embeddings=means[:, 0].detach(),
            policy_embeddings=means[:, 0],
            sampled_embeddings=q,
            kappa=torch.tensor(8.0),
        ),
        _ActionComponent(
            role="document",
            name="documents",
            rollout_embeddings=means[:, 1:].detach(),
            policy_embeddings=means[:, 1:],
            sampled_embeddings=docs,
            document_mask=valid,
            kappa=torch.tensor(8.0),
        ),
    )
    return head._compute_component_loss(
        labels, None, components, candidate_mask=valid, cross_batch_metadata=metadata
    )


@pytest.mark.parametrize(
    "all_candidates,cross_device", [(False, False), (True, False), (True, True)]
)
def test_head_ignores_frozen_rescaling_for_sampled_foreign_documents(
    all_candidates, cross_device
):
    source, q, docs, valid, labels, metadata = geometry()
    raw = source.clone().requires_grad_()
    means = F.normalize(raw, dim=-1)
    options = dict(
        ndcg_in_batch_include_negatives=all_candidates,
        reward_cross_device_negatives=cross_device,
    )
    on, on_stats, _, _ = head_loss(
        make_head(**options), means, q, docs, valid, labels, metadata
    )
    off, off_stats, _, _ = head_loss(
        make_head(**options, frozen_doc_rescale=False),
        means,
        q,
        docs,
        valid,
        labels,
        metadata,
    )
    torch.testing.assert_close(on, off, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(on, raw, retain_graph=True)[0],
        torch.autograd.grad(off, raw)[0],
    )
    assert on_stats["reward_mean"] == off_stats["reward_mean"]


def _distributed_worker(rank, rendezvous, estimator, no_local_foreign):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        source, q, docs, valid, labels, metadata = geometry()
        if no_local_foreign:
            for row in metadata[:2]:
                row["known_positive_keys"] = [
                    key for other in metadata for key in other["keys"] if key
                ]
        start, stop, width = (0, 2, 3) if rank == 0 else (2, 3, 2)
        local_metadata = [
            dict(row, keys=row["keys"][:width], ids=row["ids"][:width])
            for row in metadata[start:stop]
        ]
        weight = torch.eye(source.size(-1), requires_grad=True)
        local = F.normalize(source[start:stop, : width + 1] @ weight.T, dim=-1)
        head = make_head(
            estimator,
            reward_cross_device_negatives=True,
            ndcg_in_batch_include_negatives=True,
        )
        # Identical means and identical global RNG state still produce independent
        # document draws across ranks under the configured rollout RNG.
        torch.manual_seed(321)
        draw = head._draw(source[0, :1], torch.tensor(8.0)).contiguous()
        drawn = [torch.empty_like(draw) for _ in range(2)]
        dist.all_gather(drawn, draw)
        assert not torch.allclose(drawn[0], drawn[1])
        loss, _, _, _ = head_loss(
            head,
            local,
            q[start:stop],
            docs[start:stop, :, :width],
            valid[start:stop, :width],
            labels[start:stop, :width],
            local_metadata,
        )
        loss.backward()
        actual = weight.grad.clone()
        dist.all_reduce(actual)
        actual /= 2
        reference_weight = torch.eye(source.size(-1), requires_grad=True)
        means = F.normalize(source @ reference_weight.T, dim=-1)
        mask = cross_document_mask(metadata, metadata, 3, 0, "cpu")
        terms = head.reward_terms
        reward = full_rewards(
            q,
            docs,
            labels,
            valid,
            docs.transpose(0, 1).reshape(4, -1, 8),
            mask,
            torch.zeros_like(mask),
            terms,
        )[terms[0].name]
        reference = explicit_loss(means, q, docs, reward, valid, mask, estimator)
        reference.backward()
        torch.testing.assert_close(actual, reference_weight.grad, atol=5e-6, rtol=4e-5)
        assert actual.norm() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("estimator", ["score_function", "conditional_projection"])
@pytest.mark.parametrize("no_local_foreign", [False, True])
def test_distributed_gradient_matches_global_reference(
    tmp_path, estimator, no_local_foreign
):
    torch.multiprocessing.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"), estimator, no_local_foreign),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        dict(action_components="query"),
        dict(sampling_law="gaussian"),
        dict(sigma_learnable=True),
        dict(rollout="diagonal"),
        dict(advantage_norm="shared"),
        dict(advantage_baseline="group"),
        dict(document_log_prob_reduction="mean"),
        dict(document_advantage_baseline="counterfactual"),
        dict(in_batch_use_sampled_documents=True),
        dict(reward_type="infonce"),
        dict(reward_ndcg_k=0),
    ],
)
def test_configuration_rejects_incomplete_policy_estimators(overrides):
    base = dict(
        action_components="query;positive,negative",
        reward_type="ndcg_in_batch",
        cross_query_document_gradients=True,
    )
    with pytest.raises(ValueError):
        RLArguments(**dict(base, **overrides))
    with pytest.raises(ValueError):
        GRPO(**dict(base, **overrides))


def test_cross_device_policy_requires_independent_rollout_rng():
    options = dict(
        action_components="query;positive,negative",
        reward_type="ndcg_in_batch",
        cross_query_document_gradients=True,
        reward_cross_device_negatives=True,
        ndcg_in_batch_include_negatives=True,
    )
    for constructor in (RLArguments, GRPO):
        with pytest.raises(ValueError):
            constructor(**options)
        constructor(**options, rollout_seed=42)


def test_trainer_step_and_checkpoint_policy_contract(tmp_path):
    from transformers import BertConfig, BertModel, TrainingArguments

    torch.manual_seed(7)
    backbone = BertModel(
        BertConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    )
    wrapper = GRPOModel(
        backbone,
        RLArguments(
            action_components="query;positive,negative",
            group_size=4,
            kappa=8.0,
            reward_type="ndcg_in_batch",
            reward_ndcg_k=2,
            gradient_estimator="conditional_projection",
            cross_query_document_gradients=True,
        ),
    )

    def tokens(ids):
        return dict(
            input_ids=torch.tensor(ids)[:, None],
            attention_mask=torch.ones(len(ids), 1, dtype=torch.long),
        )

    batch = dict(
        query=tokens([0, 1]),
        positive_document=tokens([2, 5]),
        negative_document=tokens([3, 4, 6, 7]),
        relevance_labels=torch.tensor([[3.0, 1.0, 0.0], [3.0, 0.0, 0.0]]),
        candidate_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    before = backbone.embeddings.word_embeddings.weight.detach().clone()
    trainer = GRPOTrainer(
        model=wrapper,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            max_steps=1,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            logging_steps=1,
            save_steps=1,
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=False,
        ),
        train_dataset=[0, 1],
        data_collator=lambda _: batch,
    )
    trainer.train()
    assert (backbone.embeddings.word_embeddings.weight.detach() - before).norm() > 0
    checkpoint = tmp_path / "checkpoint-1"
    payload = json.loads((checkpoint / "exploration_state.json").read_text())
    assert payload["cross_query_document_gradients"] is True
    restore_exploration_state(wrapper, checkpoint)
    wrapper.grpo.cross_query_document_gradients = False
    with pytest.raises(ValueError):
        restore_exploration_state(wrapper, checkpoint)
    payload.pop("cross_query_document_gradients")
    payload.pop("config")  # Simulate a legacy checkpoint without the full spec.
    (checkpoint / "exploration_state.json").write_text(json.dumps(payload))
    restore_exploration_state(wrapper, checkpoint)
    wrapper.grpo.cross_query_document_gradients = True
    with pytest.raises(ValueError):
        restore_exploration_state(wrapper, checkpoint)
    with pytest.raises(ValueError):
        restore_exploration_state(wrapper, tmp_path / "missing-contract")
