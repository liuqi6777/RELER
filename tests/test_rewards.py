"""Policy-independent contracts for reward evaluation."""

import torch

from reler.objectives.rewards import (
    RewardEvaluator,
    RewardInputs,
    compute_reward_from_scores,
    normalize_reward_terms,
)


def test_evaluator_declares_candidate_pools_and_combines_terms():
    terms = normalize_reward_terms(
        [
            {
                "type": "ndcg_in_batch",
                "weight": 0.25,
                "k": 2,
                "include_negatives": True,
            },
            {
                "type": "contrastive",
                "weight": 0.75,
                "temperature": 0.2,
                "in_batch_negatives": True,
            },
        ]
    )
    evaluator = RewardEvaluator(terms)
    requirements = evaluator.requirements
    assert requirements.all_candidates
    assert requirements.representative_candidates

    scores = torch.tensor([[[0.7, 0.2, -0.1], [0.1, 0.8, 0.3]]])
    labels = torch.tensor([[3.0, 1.0, 0.0]])
    positive_scores = torch.tensor([[[0.4, -0.2], [0.6, 0.0]]])
    candidate_scores = torch.tensor([[[0.4, -0.2, 0.5], [0.6, 0.0, -0.4]]])
    evaluation = evaluator.evaluate(
        RewardInputs(
            scores=scores,
            relevance_labels=labels,
            in_batch_positive_scores=positive_scores,
            in_batch_candidate_scores=candidate_scores,
        )
    )

    expected_ndcg = compute_reward_from_scores(
        scores,
        labels,
        reward_type="ndcg_in_batch",
        k=2,
        ndcg_in_batch_include_negatives=True,
        in_batch_candidate_scores=candidate_scores,
    )
    expected_contrastive = compute_reward_from_scores(
        scores,
        labels,
        reward_type="contrastive",
        contrastive_temperature=0.2,
        contrastive_use_in_batch_negatives=True,
        in_batch_positive_scores=positive_scores,
    )
    torch.testing.assert_close(evaluation.term_rewards[terms[0].name], expected_ndcg)
    torch.testing.assert_close(
        evaluation.term_rewards[terms[1].name], expected_contrastive
    )
    torch.testing.assert_close(
        evaluation.combined,
        0.25 * expected_ndcg + 0.75 * expected_contrastive,
    )
    torch.testing.assert_close(evaluation.signals[0].values, expected_ndcg)
    torch.testing.assert_close(evaluation.signals[1].values, expected_contrastive)
    assert [signal.weight for signal in evaluation.signals] == [0.25, 0.75]


def test_plain_slate_reward_does_not_request_cross_query_scores():
    evaluator = RewardEvaluator(normalize_reward_terms("mrr"))
    requirements = evaluator.requirements
    assert not requirements.all_candidates
    assert not requirements.representative_candidates


def test_fixed_pool_evaluator_preserves_multi_term_weights():
    terms = normalize_reward_terms(
        [
            {
                "type": "ndcg_in_batch",
                "name": "rank",
                "weight": 0.3,
                "k": 2,
                "include_negatives": True,
            },
            {
                "type": "mrr_in_batch",
                "name": "reciprocal",
                "weight": 0.7,
                "k": 3,
                "include_negatives": True,
            },
        ]
    )
    evaluator = RewardEvaluator(terms)
    scores = torch.tensor(
        [
            [
                [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
                [[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]],
            ]
        ]
    )
    labels = torch.tensor([[3.0, 1.0, 0.0]])
    valid = torch.tensor([[True, True, False]])
    cross_scores = torch.tensor([[[1.0, 0.0, -torch.inf], [0.5, 2.0, 0.5]]])
    evaluation = evaluator.evaluate(
        RewardInputs(
            scores=scores,
            relevance_labels=labels,
            candidate_mask=valid,
            fixed_cross_scores=cross_scores,
        )
    )
    expanded_cross_scores = cross_scores[:, :, None].expand(-1, -1, 2, -1)
    expected_rank = compute_reward_from_scores(
        scores,
        labels,
        reward_type="ndcg_in_batch",
        k=2,
        ndcg_in_batch_include_negatives=True,
        in_batch_candidate_scores=expanded_cross_scores,
        candidate_mask=valid,
    )
    expected_reciprocal = compute_reward_from_scores(
        scores,
        labels,
        reward_type="mrr_in_batch",
        k=3,
        ndcg_in_batch_include_negatives=True,
        in_batch_candidate_scores=expanded_cross_scores,
        candidate_mask=valid,
    )

    torch.testing.assert_close(
        evaluation.term_rewards["rank"], expected_rank, rtol=0, atol=0
    )
    torch.testing.assert_close(
        evaluation.term_rewards["reciprocal"], expected_reciprocal, rtol=0, atol=0
    )
    torch.testing.assert_close(
        evaluation.combined,
        0.3 * expected_rank + 0.7 * expected_reciprocal,
        rtol=0,
        atol=0,
    )


def test_counterfactual_evaluation_is_weighted_masked_and_non_mutating():
    terms = normalize_reward_terms(
        [
            {"type": "mrr", "weight": 0.4, "k": 3},
            {"type": "ndcg", "weight": 0.6, "k": 2},
        ]
    )
    evaluator = RewardEvaluator(terms)
    scores = torch.tensor([[[0.9, 0.4, -2.0], [0.2, 0.8, -3.0], [0.5, 0.1, -4.0]]])
    original_scores = scores.clone()
    reference = torch.tensor([[[0.3, 0.6, -5.0], [0.3, 0.6, -5.0], [0.3, 0.6, -5.0]]])
    labels = torch.tensor([[3.0, 1.0, 0.0]])
    valid = torch.tensor([[True, True, False]])
    inputs = RewardInputs(
        scores=scores,
        relevance_labels=labels,
        candidate_mask=valid,
    )
    rewards = evaluator.evaluate(inputs).combined
    actual = evaluator.counterfactual_document_advantages(
        inputs,
        reference_scores=reference,
        rewards=rewards,
        document_axis=1,
    )

    expected = torch.zeros_like(actual)
    for document_index in range(scores.size(-1)):
        replaced = scores.clone()
        replaced[..., document_index] = reference[..., document_index]
        baseline = evaluator.evaluate(
            RewardInputs(
                scores=replaced,
                relevance_labels=labels,
                candidate_mask=valid,
            )
        ).combined
        expected[..., document_index] = rewards - baseline
    expected[..., ~valid[0]] = 0

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores, original_scores, rtol=0, atol=0)
