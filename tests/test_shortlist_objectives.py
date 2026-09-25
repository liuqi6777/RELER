"""Independent labels and per-cell gradient references for shortlist objectives."""

import math

import pytest
import torch
import torch.nn.functional as F
from test_shortlists import (
    _distributed_worker,
    components,
    metadata,
    options,
    reference_loss,
)

import reler.objectives.shortlists as shortlist_module
from reler.config import RLArguments
from reler.objectives.policy_math import mean_alignment
from reler.objectives.rewards import RewardEvaluator, normalize_reward_terms
from reler.objectives.shortlists import sample_shortlists, shortlist_rewards
from reler.training.grpo_model import GRPO


def test_binary_reward_uses_original_positives_despite_opposing_grades_and_padding():
    # Ranking: annotated negative, positive, cross negative, second positive.
    scores = torch.tensor([0.4, 0.2, 0.5, 99.0]).reshape(1, 1, 1, 4).expand(1, 2, 3, 4)
    kwargs = dict(
        scores=scores,
        labels=torch.tensor([[0.0, 3.0, 2.0, 3.0]]),
        valid=torch.tensor([[True, True, True, False]]),
        rank_labels=None,
        cross_scores=torch.tensor([[[0.3, -torch.inf]]]).expand(1, 2, 2),
        positive_mask=torch.tensor([[True, True, False, False]]),
    )
    terms = normalize_reward_terms(
        "ndcg_in_batch", default_k=3, default_ndcg_in_batch_include_negatives=True
    )
    d2 = 1 / math.log2(3)
    expected_graded = 3 / (7 + 3 * d2)
    expected_binary = d2 / (1 + d2)
    evaluator = RewardEvaluator(terms)
    for alpha in (0.0, 0.25, 0.5, 1.0):
        reward, values = shortlist_rewards(evaluator, binary_weight=alpha, **kwargs)
        torch.testing.assert_close(
            reward,
            torch.full_like(
                reward, (1 - alpha) * expected_graded + alpha * expected_binary
            ),
        )
        if alpha:
            torch.testing.assert_close(
                values["binary_ndcg"], torch.full_like(reward, expected_binary)
            )


@pytest.mark.parametrize("estimator", ["score_function", "conditional_projection"])
@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("rescale", [True, False])
def test_mixed_gradient_matches_separate_graded_and_binary_cellwise_oracles(
    monkeypatch, estimator, alpha, rescale
):
    torch.manual_seed(51)
    source = torch.randn(3, 4, 17)
    weight = torch.eye(17, requires_grad=True)
    means = F.normalize(source @ weight.T, dim=-1)
    q = F.normalize(torch.randn(3, 4, 17), dim=-1)
    valid = torch.tensor([[True] * 3, [True] * 3, [True, True, False]])
    docs = F.normalize(torch.randn(3, 4, 3, 17), dim=-1).masked_fill(
        ~valid[:, None, :, None], 0
    )
    labels = torch.tensor([[0.0, 1.0, 3.0], [3.0, 0.0, 1.0], [0.0, 3.0, 0.0]])
    positives = torch.tensor(
        [[True, True, False], [True, False, False], [True, False, False]]
    )
    captured = []

    def sample(*args, **kwargs):
        result = sample_shortlists(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(shortlist_module, "sample_shortlists", sample)
    head = GRPO(
        **options(
            gradient_estimator=estimator,
            frozen_doc_rescale=rescale,
            reward_shortlist_binary_weight=alpha,
        )
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, stats, _, _ = head._compute_component_loss(
            labels,
            None,
            components(means, q, docs, valid),
            candidate_mask=valid,
            cross_batch_metadata=metadata(),
            positive_mask=positives,
        )
    actual = torch.autograd.grad(loss, weight)[0]
    ref_weight = torch.eye(17, requires_grad=True)
    ref = F.normalize(source @ ref_weight.T, dim=-1)
    idx, mask, _ = captured[0]
    args = (ref, q, docs)
    tail = (
        valid,
        ref[:, 1:].detach().reshape(-1, 17),
        idx,
        mask,
        estimator,
        mean_alignment(17, 9.0) if rescale else 1.0,
    )
    graded_loss, graded = reference_loss(*args, labels, *tail)
    binary_loss, binary = reference_loss(*args, positives.float(), *tail)
    expected = (1 - alpha) * graded_loss + alpha * binary_loss
    torch.testing.assert_close(
        actual, torch.autograd.grad(expected, ref_weight)[0], atol=3e-6, rtol=3e-5
    )
    torch.testing.assert_close(
        stats["reward_mean"], ((1 - alpha) * graded + alpha * binary).mean()
    )
    torch.testing.assert_close(stats["reward/binary_ndcg/mean"], binary.mean())
    assert actual.norm() > 0


@pytest.mark.parametrize(
    "changes",
    [
        dict(reward_shortlist_binary_weight=-0.1),
        dict(reward_shortlist_binary_weight=1.1),
        dict(reward_shortlist_binary_weight=float("nan")),
        dict(reward_shortlist_binary_weight=float("inf")),
        dict(reward_shortlist_binary_weight=0.25, reward_shortlist_count=0),
        dict(reward_shortlist_binary_weight=0.25, reward_type="mrr_in_batch"),
        dict(
            reward_shortlist_binary_weight=0.25,
            reward_terms=[dict(type="ndcg_in_batch", weight=2.0)],
        ),
    ],
)
def test_invalid_mixture_rejected_at_both_entrypoints(changes):
    for constructor in (RLArguments, GRPO):
        with pytest.raises(ValueError):
            constructor(**options(**changes))


@pytest.mark.parametrize(
    "positives",
    [
        None,
        torch.ones(1, 3),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
    ],
)
def test_missing_or_invalid_binary_identities_are_never_inferred_from_grades(positives):
    terms = normalize_reward_terms(
        "ndcg_in_batch", default_ndcg_in_batch_include_negatives=True
    )
    with pytest.raises(ValueError):
        shortlist_rewards(
            RewardEvaluator(terms),
            scores=torch.ones(1, 2, 2, 3),
            labels=torch.tensor([[3.0, 2.0, 0.0]]),
            valid=torch.ones(1, 3, dtype=torch.bool),
            rank_labels=None,
            cross_scores=torch.empty(1, 2, 0),
            binary_weight=0.25,
            positive_mask=positives,
        )


@pytest.mark.parametrize("estimator", ["score_function", "conditional_projection"])
def test_mixed_two_rank_gradient_with_uneven_tails_and_empty_cross_pool(
    tmp_path, estimator
):
    torch.multiprocessing.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"), estimator, "cross_device_all", 0.25),
        nprocs=2,
        join=True,
    )
