"""Large-pool ranking rewards and estimators against uncompressed global oracles."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from reler.config import RLArguments
from reler.objectives.contrastive import cross_document_mask
from reler.objectives.policy_math import mean_alignment
from reler.objectives.projection import (
    conditional_projection_loss,
    project_span,
    project_with_fixed_documents,
)
from reler.objectives.rewards import (
    RewardEvaluator,
    RewardInputs,
    compute_reward_from_scores,
    normalize_reward_terms,
)
from reler.training.grpo_model import GRPO, GRPOModel, _ActionComponent
from reler.training.trainer import restore_exploration_state, rollout_rng_contract


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("kind", ["ndcg_in_batch", "mrr_in_batch"])
@pytest.mark.parametrize("k", [1, 3, 10])
def test_chunked_reward_matches_full_pool_including_ties(kind, k):
    torch.manual_seed(61)
    own = torch.randn(3, 4, 4, 5)
    # Quantize to exercise ties between relevant and irrelevant documents too.
    own = own.round()
    labels = torch.tensor(
        [
            [3.0, 2.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    valid = torch.tensor([[True] * 5, [True] * 4 + [False], [True] * 5])
    cross = torch.randn(3, 4, 37).round()
    cross[0] = -torch.inf  # No allowed extra candidates for one query.
    cross[1, :, 7:] = -torch.inf
    kwargs = dict(
        scores=own,
        relevance_labels=labels,
        candidate_mask=valid,
        reward_type=kind,
        k=k,
        ndcg_in_batch_include_negatives=True,
    )
    full = compute_reward_from_scores(
        **kwargs, in_batch_candidate_scores=cross[:, :, None].expand(-1, -1, 4, -1)
    )
    terms = normalize_reward_terms(
        kind, default_k=k, default_ndcg_in_batch_include_negatives=True
    )
    compact = (
        RewardEvaluator(terms)
        .evaluate(
            RewardInputs(
                scores=own,
                relevance_labels=labels,
                candidate_mask=valid,
                fixed_cross_scores=cross,
            )
        )
        .combined
    )
    torch.testing.assert_close(compact, full, rtol=0, atol=0)


@pytest.mark.parametrize("fixed_count,rank", [(30, 12), (30, 4), (3, 3)])
def test_streamed_projection_matches_full_svd(fixed_count, rank):
    torch.manual_seed(19)
    batch, group, dim = 2, 3, 12
    vectors = torch.randn(batch, group, dim)
    moving = torch.randn(batch, group, dim, 4)
    fixed = torch.randn(batch, fixed_count, dim)
    fixed[..., rank:] = 0
    mask = torch.ones(batch, fixed_count, dtype=torch.bool)
    mask[1, -1] = False
    columns = torch.cat(
        (
            moving,
            fixed.masked_fill(~mask[..., None], 0)
            .transpose(-2, -1)[:, None]
            .expand(-1, group, -1, -1),
        ),
        dim=-1,
    )
    expected, expected_rank = project_span(vectors, columns)
    actual, actual_rank = project_with_fixed_documents(vectors, moving, fixed, mask)
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=2e-5)
    torch.testing.assert_close(actual_rank, expected_rank)


def make_head(estimator, **kwargs):
    return GRPO(
        action_components="query;positive,negative",
        group_size=4,
        kappa=9,
        rollout_seed=42,
        gradient_estimator=estimator,
        reward_type="ndcg_in_batch",
        reward_ndcg_k=2,
        ndcg_in_batch_include_negatives=True,
        reward_cross_device_negatives=True,
        **kwargs,
    )


def metadata():
    rows = [
        dict(
            keys=[f"d{3 * i + j}" for j in range(3)],
            ids=[None] * 3,
            source="test",
            known_ids=[],
            known_positive_keys=[],
        )
        for i in range(3)
    ]
    rows[2]["keys"][2] = None
    rows[0]["known_positive_keys"] = ["d7"]
    return rows


@pytest.mark.parametrize("estimator", ["score_function", "conditional_projection"])
def test_full_head_receives_metadata_and_produces_gradients(estimator):
    torch.manual_seed(31)
    raw = torch.randn(3, 4, 12, requires_grad=True)
    valid = torch.tensor([[True] * 3, [True] * 3, [True, True, False]])
    loss, stats, _, _, _ = make_head(estimator)(
        rollout_query_embeddings=raw[:, 0].detach(),
        policy_query_embeddings=raw[:, 0],
        rollout_positive_document_embeddings=raw[:, 1:2].detach(),
        policy_positive_document_embeddings=raw[:, 1:2],
        rollout_negative_document_embeddings=raw[:, 2:].detach(),
        policy_negative_document_embeddings=raw[:, 2:],
        relevance_labels=torch.tensor(
            [[3.0, 1.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        candidate_mask=valid,
        cross_batch_metadata=metadata(),
    )
    loss.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.norm() > 0
    assert raw.grad[2, -1].norm() == 0
    assert stats["reward_pool/cross_candidates_mean"] == 5
    assert "train/loss_infonce" not in stats


def _distributed_worker(rank, rendezvous, estimator):
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
        torch.manual_seed(702)
        dim, group = 12, 4
        source = torch.randn(3, 4, dim)
        query_actions = F.normalize(torch.randn(3, group, dim), dim=-1)
        doc_actions = F.normalize(torch.randn(3, group, 3, dim), dim=-1)
        valid = torch.tensor([[True] * 3, [True] * 3, [True, True, False]])
        doc_actions = doc_actions.masked_fill(~valid[:, None, :, None], 0)
        labels = torch.tensor([[3.0, 1.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        start, stop, width = (0, 2, 3) if rank == 0 else (2, 3, 2)
        local_metadata = [
            dict(row, keys=row["keys"][:width], ids=row["ids"][:width])
            for row in metadata()[start:stop]
        ]
        weight = torch.eye(dim, requires_grad=True)
        directions = F.normalize(source[start:stop, : width + 1] @ weight.T, dim=-1)
        local_valid = valid[start:stop, :width]
        components = (
            _ActionComponent(
                role="query",
                name="query",
                rollout_embeddings=directions[:, 0].detach(),
                policy_embeddings=directions[:, 0],
                sampled_embeddings=query_actions[start:stop],
                kappa=torch.tensor(9.0),
            ),
            _ActionComponent(
                role="document",
                name="documents",
                rollout_embeddings=directions[:, 1:].detach(),
                policy_embeddings=directions[:, 1:],
                sampled_embeddings=doc_actions[start:stop, :, :width],
                kappa=torch.tensor(9.0),
                document_mask=local_valid,
            ),
        )
        loss, _, _, _ = make_head(estimator)._compute_component_loss(
            labels[start:stop, :width],
            None,
            components,
            candidate_mask=local_valid,
            cross_batch_metadata=local_metadata,
        )
        loss.backward()
        actual = weight.grad.clone()
        dist.all_reduce(actual)
        actual /= 2  # DDP/ZeRO mean reduction.

        # Global, uncompressed reward and estimator; same prescribed actions.
        reference_weight = torch.eye(dim, requires_grad=True)
        means = F.normalize(source @ reference_weight.T, dim=-1)
        pool = means[:, 1:].detach().reshape(9, dim)
        allowed = valid.reshape(1, -1).expand(3, -1).clone()
        for i in range(3):
            allowed[i, 3 * i : 3 * i + 3] = False
        allowed[0, 7] = False
        torch.testing.assert_close(
            cross_document_mask(metadata(), metadata(), 3, 0, "cpu"), allowed
        )
        own = torch.einsum("bid,bjmd->bijm", query_actions, doc_actions)
        cross = (query_actions @ pool.T * mean_alignment(dim, 9)).masked_fill(
            ~allowed[:, None], -torch.inf
        )
        reward = compute_reward_from_scores(
            own,
            labels,
            reward_type="ndcg_in_batch",
            k=2,
            candidate_mask=valid,
            ndcg_in_batch_include_negatives=True,
            in_batch_candidate_scores=cross[:, :, None].expand(-1, -1, group, -1),
        )
        if estimator == "conditional_projection":
            expected, _ = conditional_projection_loss(
                means[:, 0],
                means[:, 1:],
                query_actions,
                doc_actions,
                reward,
                9.0,
                valid,
                frozen_documents=pool[None].expand(3, -1, -1),
                frozen_mask=allowed,
            )
        else:
            aq = (reward.mean(2) - reward.mean((1, 2))[:, None]) * group / (group - 1)
            ad = (reward.mean(1) - reward.mean((1, 2))[:, None]) * group / (group - 1)
            qlog = 9 * (query_actions * means[:, None, 0]).sum(-1)
            dlog = 9 * (doc_actions * means[:, None, 1:]).sum((-1, -2))
            expected = -(aq.detach() * qlog).mean() - (ad.detach() * dlog).mean()
        expected.backward()
        torch.testing.assert_close(actual, reference_weight.grad, atol=4e-6, rtol=4e-5)
        assert actual.norm() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("estimator", ["score_function", "conditional_projection"])
def test_cross_device_rl_matches_global_gradient_with_uneven_tails(tmp_path, estimator):
    torch.multiprocessing.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"), estimator),
        nprocs=2,
        join=True,
    )


def test_model_wrapper_passes_large_pool_configuration_and_metadata():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.table = torch.nn.Embedding(16, 12)
            self.calls = 0

        def forward(self, input_ids, attention_mask):
            self.calls += 1
            return SimpleNamespace(last_hidden_state=self.table(input_ids))

    def tokens(ids):
        values = torch.tensor(ids).reshape(-1, 1)
        return dict(input_ids=values, attention_mask=torch.ones_like(values))

    args = RLArguments(
        action_components="query;positive,negative",
        group_size=4,
        kappa=9,
        reward_type="ndcg_in_batch",
        ndcg_in_batch_include_negatives=True,
        reward_cross_device_negatives=True,
        gradient_estimator="conditional_projection",
    )
    torch.manual_seed(83)
    model = Encoder()
    wrapper = GRPOModel(model, args)
    output = wrapper(
        query=tokens([0, 1, 2]),
        positive_document=tokens([3, 6, 9]),
        negative_document=tokens([4, 5, 7, 8, 10, 11]),
        relevance_labels=torch.tensor(
            [[3.0, 1.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        candidate_mask=torch.tensor([[True] * 3, [True] * 3, [True, True, False]]),
        cross_batch_metadata=metadata(),
    )
    output.loss.backward()
    assert model.calls == 2
    assert (
        torch.isfinite(model.table.weight.grad).all()
        and model.table.weight.grad.norm() > 0
    )
    assert model.table.weight.grad[11].norm() == 0
    assert output.reward_terms["reward_pool/cross_candidates_mean"] == 5


def test_resume_rejects_changing_reward_pool(tmp_path):
    head = make_head("conditional_projection")
    state = dict(
        exploration=head.exploration.state_dict(),
        estimator={"gradient_estimator": "conditional_projection"},
        rollout_rng=rollout_rng_contract(head),
        reward_cross_device_negatives=True,
    )
    path = tmp_path / "exploration_state.json"
    path.write_text(json.dumps(state))
    restore_exploration_state(SimpleNamespace(grpo=head), tmp_path)
    head.reward_cross_device_negatives = False
    with pytest.raises(ValueError):
        restore_exploration_state(SimpleNamespace(grpo=head), tmp_path)
    state.pop("reward_cross_device_negatives")
    path.write_text(json.dumps(state))
    restore_exploration_state(SimpleNamespace(grpo=head), tmp_path)
    head.reward_cross_device_negatives = True
    with pytest.raises(ValueError):
        restore_exploration_state(SimpleNamespace(grpo=head), tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        dict(in_batch_use_sampled_documents=True),
        dict(rollout="diagonal"),
        dict(action_components="query"),
        dict(reward_ndcg_k=0),
        dict(reward_type="infonce"),
        dict(ndcg_in_batch_include_negatives=False),
    ],
)
def test_reject_unsupported_large_pool_semantics(changes):
    kwargs = dict(
        action_components="query;positive,negative",
        reward_type="ndcg_in_batch",
        ndcg_in_batch_include_negatives=True,
        reward_cross_device_negatives=True,
    )
    with pytest.raises(ValueError):
        RLArguments(**{**kwargs, **changes})
