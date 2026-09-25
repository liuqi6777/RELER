"""Numerical estimator checks, including an analytic vMF expectation."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reler.config import RLArguments
from reler.objectives.policy_math import mean_alignment
from reler.objectives.projection import conditional_projection_loss, project_span
from reler.training.grpo_model import GRPO, sample_vmf
from reler.training.trainer import restore_exploration_state


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def geometry(batch=2, dim=9, group=4):
    torch.manual_seed(72)
    means = F.normalize(torch.randn(batch, 4, dim), dim=-1)
    q = sample_vmf(means[:, 0], 9, group)
    docs = sample_vmf(means[:, 1:].reshape(-1, dim), 9, group)
    docs = docs.reshape(batch, 3, group, dim).transpose(1, 2)
    return means, q, docs


def test_rank_deficient_span_and_padding():
    columns = torch.tensor([[1., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 1., 0.]])
    vector = torch.tensor([2., 3., 4.])
    projected, rank = project_span(vector, columns)
    torch.testing.assert_close(projected, torch.tensor([2., 0., 4.]))
    assert rank == 2
    zero, rank = project_span(vector, torch.zeros_like(columns))
    assert rank == 0 and zero.norm() == 0


def explicit_projection(means, q, docs, reward, valid, fixed, fixed_mask, kappa):
    """Slow per-cell orthogonal projector oracle; independent of contractions."""
    batch, gq, dim = q.shape
    gd, count = docs.shape[1:3]
    result = torch.zeros_like(means, dtype=torch.float64)
    for b in range(batch):
        for i in range(gq):
            for j in range(gd):
                aq = reward[b, i, j] - torch.cat((reward[b, :i, j], reward[b, i+1:, j])).mean()
                ad = reward[b, i, j] - torch.cat((reward[b, i, :j], reward[b, i, j+1:])).mean()
                columns = torch.cat((means[b, :1], docs[b, j, valid[b]], fixed[b, fixed_mask[b]])).double().T
                result[b, 0] += aq * (columns @ torch.linalg.pinv(columns) @ q[b, i].double())
                for m in range(count):
                    if valid[b, m]:
                        columns = torch.stack((means[b, m+1], q[b, i])).double().T
                        result[b, m+1] += ad * (columns @ torch.linalg.pinv(columns) @ docs[b, j, m].double())
    return result * (-kappa / (batch * gq * gd))


def test_per_cell_projection_matches_oracle_and_masks():
    means, q, docs = geometry()
    valid = torch.tensor([[True, True, False], [True, True, True]])
    fixed = F.normalize(torch.randn(2, 3, 9), dim=-1)
    fixed[:, 1] = fixed[:, 0]  # Duplicate and masked vectors are routine.
    fixed_mask = torch.tensor([[True, True, False], [False, False, False]])
    reward = torch.randn(2, 4, 4)
    live = means.clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = conditional_projection_loss(
            live[:, 0], live[:, 1:], q.requires_grad_(), docs.requires_grad_(), reward.requires_grad_(),
            9., valid, frozen_documents=fixed.requires_grad_(), frozen_mask=fixed_mask,
        )
    loss.backward()
    expected = explicit_projection(means, q.detach(), docs.detach(), reward.detach(), valid, fixed.detach(), fixed_mask, 9.)
    torch.testing.assert_close(live.grad.double(), expected, atol=2e-6, rtol=2e-5)
    assert q.grad is None and docs.grad is None and reward.grad is None and fixed.grad is None
    assert live.grad[0, 3].norm() == 0


def test_parallel_and_nearly_parallel_document_directions_are_finite():
    means = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]], requires_grad=True)
    q = F.normalize(torch.tensor([[[1., 0., 0.], [1., 1e-7, 0.]]]), dim=-1)
    docs = F.normalize(torch.randn(1, 2, 2, 3), dim=-1)
    loss, _ = conditional_projection_loss(
        F.normalize(means[:, 0], dim=-1), F.normalize(means[:, 1:], dim=-1),
        q, docs, torch.tensor([[[1., 0.], [0., 1.]]]), 12., torch.ones(1, 2, dtype=torch.bool),
    )
    loss.backward()
    assert torch.isfinite(means.grad).all()
    assert means.grad[:, 1:, 2].abs().max() < 1e-6


def test_bilinear_expected_gradient_through_shared_encoder():
    # E[R] = A_d(kappa)^2 sum_m c_m hq.hdm + A_d(kappa) hq.f.
    # Includes fixed distractors and shared query/document Jacobian covariance.
    repeats, dim, group, kappa = 1024, 12, 12, 18.
    means, _, _ = geometry(batch=1, dim=dim, group=group)
    source = means[0]
    live = source.expand(repeats, -1, -1).clone().requires_grad_()
    directions = F.normalize(live, dim=-1)
    actions = sample_vmf(source, kappa, repeats * group).reshape(4, repeats, group, dim)
    q, docs = actions[0], actions[1:].permute(1, 2, 0, 3)
    coefficients = torch.tensor([1., -.7, .4])
    fixed = F.normalize(torch.randn(dim), dim=0)
    reward = (torch.einsum("bid,bjmd->bijm", q, docs) * coefficients).sum(-1)
    reward += torch.einsum("bid,d->bi", q, fixed)[..., None]
    loss, _ = conditional_projection_loss(
        directions[:, 0], directions[:, 1:], q, docs, reward, kappa,
        torch.ones(repeats, 3, dtype=torch.bool),
        frozen_documents=fixed.expand(repeats, 1, dim), frozen_mask=torch.ones(repeats, 1, dtype=torch.bool),
    )
    gradients = torch.autograd.grad(loss, live)[0] * repeats
    shared = torch.einsum("bmd,mk->bdk", gradients, source).double()
    weight = torch.eye(dim, requires_grad=True)
    encoded = F.normalize(source @ weight.T, dim=-1)
    alignment = mean_alignment(dim, kappa)
    expected_loss = -alignment**2 * ((encoded[0] * encoded[1:]).sum(-1) * coefficients).sum()
    expected_loss -= alignment * (encoded[0] * fixed).sum()
    expected = torch.autograd.grad(expected_loss, weight)[0].double()
    error = (shared.mean(0) - expected).norm()
    standard_error = (shared.var(0, unbiased=True).sum() / repeats).sqrt()
    assert error < 4 * standard_error
    assert error / expected.norm() < .06


@pytest.mark.parametrize("override", [
    {"action_components": "query"}, {"sampling_law": "gaussian"}, {"sigma_learnable": True},
    {"rollout": "diagonal"}, {"advantage_baseline": "group"}, {"advantage_norm": "shared"},
    {"reward_combine": "normalized_sum"}, {"in_batch_use_sampled_documents": True},
    {"document_advantage_baseline": "counterfactual"}, {"document_log_prob_reduction": "mean"},
])
def test_reject_unsupported_estimators_in_both_entrypoints(override):
    config = dict(action_components="query;positive,negative", gradient_estimator="conditional_projection")
    config.update(override)
    for constructor in (RLArguments, GRPO):
        with pytest.raises(ValueError):
            constructor(**config)


def test_reject_unknown_estimator():
    with pytest.raises(ValueError):
        GRPO(gradient_estimator="typo")


def test_resume_estimator_contract(tmp_path):
    head = GRPO(action_components="query;positive,negative", gradient_estimator="conditional_projection")
    state = {"exploration": head.exploration.state_dict(), "estimator": {"gradient_estimator": "conditional_projection"}}
    path = tmp_path / "exploration_state.json"
    path.write_text(json.dumps(state))
    restore_exploration_state(SimpleNamespace(grpo=head), tmp_path)
    head.gradient_estimator = "score_function"
    with pytest.raises(ValueError):
        restore_exploration_state(SimpleNamespace(grpo=head), tmp_path)

    state["estimator"] = {}
    path.write_text(json.dumps(state))
    restore_exploration_state(SimpleNamespace(grpo=head), tmp_path)
    head.gradient_estimator = "conditional_projection"
    with pytest.raises(ValueError):
        restore_exploration_state(SimpleNamespace(grpo=head), tmp_path)


def test_projection_uses_actual_union_of_masked_cross_candidates(monkeypatch):
    import reler.training.grpo_model as module

    torch.manual_seed(13)
    means = torch.randn(2, 4, 12, requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    positive_cross = torch.tensor([[True, True], [False, True]])
    candidate_cross = torch.tensor([[[True, True, True], [False, True, False]],
                                    [[True, True, True], [True, True, True]]])
    captured = []
    original = module.conditional_projection_loss

    def inspect(*args, **kwargs):
        captured.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "conditional_projection_loss", inspect)
    head = GRPO(action_components="query;positive,negative", gradient_estimator="conditional_projection",
                group_size=4, kappa=9, rollout_seed=43,
                reward_terms=[{"type": "mrr_in_batch"}, {"type": "ndcg_in_batch", "k": 2,
                              "ndcg_in_batch_include_negatives": True}])
    kwargs = dict(rollout_query_embeddings=means[:, 0].detach(), policy_query_embeddings=means[:, 0],
                  rollout_positive_document_embeddings=means[:, 1:2].detach(), policy_positive_document_embeddings=means[:, 1:2],
                  rollout_negative_document_embeddings=means[:, 2:].detach(), policy_negative_document_embeddings=means[:, 2:],
                  relevance_labels=torch.tensor([[1., 0., 0.], [1., 1., 0.]]), candidate_mask=valid,
                  in_batch_positive_mask=positive_cross, in_batch_candidate_mask=candidate_cross)
    loss, *_ = head(**kwargs)
    actual_grad = torch.autograd.grad(loss, means)[0]
    args, fixed = captured[0]
    off = ~torch.eye(2, dtype=torch.bool)
    expected_mask = torch.cat((positive_cross & off,
                              (candidate_cross & off[..., None] & valid[None]).reshape(2, -1)), dim=1)
    torch.testing.assert_close(fixed["frozen_mask"], expected_mask)
    unit = F.normalize(means.detach(), dim=-1)
    expected_fixed = torch.cat((unit[:, 1][None].expand(2, -1, -1), unit[:, 1:].reshape(1, 6, 12).expand(2, -1, -1)), dim=1)
    torch.testing.assert_close(fixed["frozen_documents"], expected_fixed)
    oracle = explicit_projection(unit, args[2], args[3], args[4], valid, expected_fixed, expected_mask, float(args[5]))
    # Compose the independent oracle with the live normalization Jacobian.
    expected_loss = (F.normalize(means, dim=-1) * oracle.float()).sum()
    expected_grad = torch.autograd.grad(expected_loss, means)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=2e-5)
    assert actual_grad[0, 3].norm() == 0
    with pytest.raises(ValueError):
        head(**{**kwargs, "policy_query_embeddings": means[:, 0] + 1})
