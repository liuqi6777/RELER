"""Numerical contracts for the core embedding-space GRPO policy."""

import itertools
import json
import math
from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn.functional as F

from reler.config import RLArguments
from reler.objectives.policy_math import (
    ExplorationSchedule,
    group_advantages,
    kappa_for_alignment,
    mean_alignment,
    score_gap_variance,
)
from reler.objectives.rewards import RewardInputs, compute_reward_from_scores
from reler.training.grpo_model import (
    GRPO,
    _ActionComponent,
    bessel_ratio,
    sample_vmf,
)


@pytest.mark.parametrize(
    "dimension,kappa",
    [(3, 0.1), (8, 4.0), (64, 30.0), (1024, 755.0)],
)
def test_bessel_ratio_matches_mean_alignment_and_derivative(dimension, kappa):
    concentration = torch.tensor(kappa, dtype=torch.float64, requires_grad=True)
    actual = bessel_ratio(dimension / 2, concentration)
    expected = actual.new_tensor(mean_alignment(dimension, kappa))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    (actual_derivative,) = torch.autograd.grad(actual, concentration)
    # d A_d(kappa) / d kappa = 1 - A_d(kappa)^2 - (d-1) A_d(kappa) / kappa.
    expected_derivative = (
        1.0
        - actual.double().square()
        - (dimension - 1) * actual.double() / concentration
    )
    torch.testing.assert_close(
        actual_derivative,
        expected_derivative,
        rtol=1e-4,
        atol=1e-9,
    )


@pytest.mark.parametrize("dimension", [3, 128, 1024, 4096])
@pytest.mark.parametrize("target", [0.1, 0.53, 0.8, 0.95])
def test_alignment_inverse_and_linear_schedule_endpoints(dimension, target):
    concentration = kappa_for_alignment(dimension, target)
    assert mean_alignment(dimension, concentration) == pytest.approx(target, abs=2e-12)

    schedule = ExplorationSchedule(
        target_alignment=target,
        final_alignment=max(target, 0.95),
        schedule="linear",
    )
    schedule.set_step(0, 11)
    assert mean_alignment(dimension, schedule.resolve(dimension)) == pytest.approx(
        target, abs=2e-12
    )
    schedule.set_step(10, 11)
    assert mean_alignment(dimension, schedule.resolve(dimension)) == pytest.approx(
        max(target, 0.95), abs=2e-12
    )


def test_score_gap_variance_includes_longitudinal_vmf_fluctuations():
    concentration = 2.0
    gap = 1.7
    perpendicular_norm = 0.6
    alignment = 1.0 / math.tanh(concentration) - 1.0 / concentration
    longitudinal_variance = 1.0 / concentration**2 - 1.0 / math.sinh(concentration) ** 2
    expected = (
        longitudinal_variance * gap**2
        + alignment / concentration * perpendicular_norm**2
    )
    assert score_gap_variance(
        3, concentration, gap, perpendicular_norm
    ) == pytest.approx(expected, abs=1e-12)


def test_group_advantages_preserve_loo_scale_and_flag_degenerate_rows():
    rewards = torch.tensor([[0.2, 0.6, 0.7], [4.0, 4.0, 4.0]])
    advantages, degenerate = group_advantages(
        rewards, baseline="leave_one_out", normalization="none"
    )
    expected = rewards - (rewards.sum(dim=1, keepdim=True) - rewards) / (
        rewards.size(1) - 1
    )
    torch.testing.assert_close(advantages, expected)
    assert degenerate.tolist() == [False, True]

    normalized, _ = group_advantages(
        rewards, baseline="group", normalization="per_component"
    )
    assert normalized[0].std(unbiased=False) == pytest.approx(1.0)
    torch.testing.assert_close(normalized[1], torch.zeros_like(normalized[1]))


def test_vmf_sampler_has_unit_norm_and_theoretical_first_moment():
    torch.manual_seed(190)
    dimension, kappa, sample_count = 13, 7.0, 30_000
    mean = F.normalize(torch.arange(1, dimension + 1, dtype=torch.float32), dim=0)
    samples = sample_vmf(mean.unsqueeze(0), kappa, sample_count)[0]

    torch.testing.assert_close(
        samples.norm(dim=-1),
        torch.ones(sample_count),
        rtol=0,
        atol=2e-6,
    )
    empirical_mean = samples.mean(dim=0)
    empirical_alignment = empirical_mean @ mean
    expected_alignment = mean_alignment(dimension, kappa)
    assert empirical_alignment.item() == pytest.approx(expected_alignment, abs=0.012)
    tangent_mean = empirical_mean - empirical_alignment * mean
    assert tangent_mean.norm().item() < 0.015


def _joint_inputs():
    torch.manual_seed(91)
    rollout_query = torch.randn(2, 7)
    rollout_positive = torch.randn(2, 1, 7)
    rollout_negative = torch.randn(2, 3, 7)
    policy_query = rollout_query + 0.15 * torch.randn_like(rollout_query)
    policy_positive = rollout_positive + 0.15 * torch.randn_like(rollout_positive)
    policy_negative = rollout_negative + 0.15 * torch.randn_like(rollout_negative)
    return {
        "rollout_query_embeddings": rollout_query,
        "rollout_positive_document_embeddings": rollout_positive,
        "rollout_negative_document_embeddings": rollout_negative,
        "policy_values": (policy_query, policy_positive, policy_negative),
        "relevance_labels": torch.tensor([[2.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]),
        "candidate_mask": torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        ),
    }


def _replay_step(head, inputs, step):
    head.exploration.set_step(step, total_steps=10)
    policies = tuple(
        value.clone().requires_grad_() for value in inputs["policy_values"]
    )
    loss, reward_stats, advantage_stats, sigma, kl = head(
        rollout_query_embeddings=inputs["rollout_query_embeddings"],
        rollout_positive_document_embeddings=inputs[
            "rollout_positive_document_embeddings"
        ],
        rollout_negative_document_embeddings=inputs[
            "rollout_negative_document_embeddings"
        ],
        policy_query_embeddings=policies[0],
        policy_positive_document_embeddings=policies[1],
        policy_negative_document_embeddings=policies[2],
        relevance_labels=inputs["relevance_labels"],
        candidate_mask=inputs["candidate_mask"],
    )
    gradients = torch.autograd.grad(loss, policies)
    return (
        loss.detach(),
        reward_stats["reward_mean"],
        reward_stats["reward_std"],
        advantage_stats["advantages_std"],
        sigma,
        kl,
        *gradients,
    )


def test_seeded_joint_forward_replays_loss_and_gradients_after_resume_boundary():
    inputs = _joint_inputs()
    options = dict(
        action_components="query;positive,negative",
        group_size=4,
        kappa=9.0,
        reward_type="contrastive",
        contrastive_temperature=0.3,
        rollout_seed=1234,
    )
    uninterrupted = GRPO(**options).train()
    resumed = GRPO(**options).train()

    torch.manual_seed(12)
    ambient_before = torch.random.get_rng_state().clone()
    _replay_step(uninterrupted, inputs, step=4)
    expected = _replay_step(uninterrupted, inputs, step=5)
    torch.testing.assert_close(
        torch.random.get_rng_state(), ambient_before, rtol=0, atol=0
    )

    # Checkpointing happens at optimizer-step boundaries. A restored head has no
    # in-step counter history, but its first draw at the next step must be exact.
    torch.manual_seed(9876)
    resumed_ambient = torch.random.get_rng_state().clone()
    actual = _replay_step(resumed, inputs, step=5)
    torch.testing.assert_close(
        torch.random.get_rng_state(), resumed_ambient, rtol=0, atol=0
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    query_gradient, positive_gradient, negative_gradient = actual[-3:]
    assert query_gradient.norm() > 0
    assert positive_gradient.norm() > 0
    assert negative_gradient.norm() > 0
    padded_negatives = ~inputs["candidate_mask"][:, 1:]
    torch.testing.assert_close(
        negative_gradient[padded_negatives],
        torch.zeros_like(negative_gradient[padded_negatives]),
        rtol=0,
        atol=0,
    )


def test_constructor_paths_share_one_normalized_configuration():
    options = dict(
        action_components="query;positive,negative",
        group_size=4,
        kappa=9.0,
        reward_type="contrastive",
        contrastive_temperature=0.2,
        rollout_seed=17,
    )
    configured = GRPO(config=RLArguments(**options))
    direct = GRPO(**options)
    legacy_positional = GRPO(
        options["action_components"],
        **{key: value for key, value in options.items() if key != "action_components"},
    )

    assert configured.config == direct.config == legacy_positional.config


def test_rl_arguments_are_the_normalized_immutable_runtime_config():
    arguments = RLArguments(
        action_components="query;positive,negative",
        kappa=9.0,
    )

    assert arguments.resolve() is arguments
    assert arguments.action_components == (("query",), ("positive", "negative"))
    assert arguments.sigma == pytest.approx(1 / 3)
    with pytest.raises(FrozenInstanceError):
        arguments.group_size = 16


def _query_step(head, inputs, step):
    head.set_training_progress(step, total_steps=10)
    policy_query = inputs["policy_values"][0].clone().requires_grad_()
    loss, reward_stats, _, _, _ = head(
        rollout_query_embeddings=inputs["rollout_query_embeddings"],
        rollout_positive_document_embeddings=inputs[
            "rollout_positive_document_embeddings"
        ],
        rollout_negative_document_embeddings=inputs[
            "rollout_negative_document_embeddings"
        ],
        policy_query_embeddings=policy_query,
        relevance_labels=inputs["relevance_labels"],
        candidate_mask=inputs["candidate_mask"],
    )
    (gradient,) = torch.autograd.grad(loss, policy_query)
    return (
        loss.detach(),
        reward_stats["reward_mean"],
        gradient,
        head.reward_baseline.clone(),
        head.reward_baseline_initialized.clone(),
    )


def test_ema_checkpoint_contract_replays_next_step_exactly():
    inputs = _joint_inputs()
    options = dict(
        action_components="query",
        group_size=4,
        kappa=9.0,
        reward_type="contrastive",
        contrastive_temperature=0.3,
        advantage_baseline="ema",
        advantage_baseline_momentum=0.8,
        rollout_seed=321,
    )
    uninterrupted = GRPO(**options).train()
    resumed = GRPO(**options).train()

    _query_step(uninterrupted, inputs, step=4)
    payload = uninterrupted.checkpoint_contract(step=4, total_steps=10)
    resumed.load_checkpoint_contract(payload)

    expected = _query_step(uninterrupted, inputs, step=5)
    actual = _query_step(resumed, inputs, step=5)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_components", (("positive", "negative"),)),
        ("sampling_law", "gaussian"),
        ("rollout", "diagonal"),
        ("reward_combine", "normalized_sum"),
        ("kl_coef", 0.5),
    ],
)
def test_checkpoint_contract_rejects_any_core_configuration_drift(field, value):
    options = dict(
        action_components="query;positive,negative",
        group_size=4,
        kappa=9.0,
        reward_type="contrastive",
        contrastive_temperature=0.3,
        rollout_seed=123,
    )
    source = GRPO(**options)
    payload = source.checkpoint_contract(step=3, total_steps=10)
    changed = GRPO(**options)
    setattr(changed, field, value)

    with pytest.raises(ValueError, match=field):
        changed.load_checkpoint_contract(payload)


def test_learnable_sigma_has_gradient_and_round_trips_runtime_state():
    inputs = _joint_inputs()
    head = GRPO(
        action_components="query",
        group_size=5,
        sigma=0.2,
        sigma_learnable=True,
        sigma_min=0.1,
        sigma_max=0.3,
        reward_type="contrastive",
        contrastive_temperature=0.3,
        advantage_baseline="group",
        rollout_seed=89,
    ).train()
    policy_query = inputs["policy_values"][0].clone().requires_grad_()
    loss, _, _, sigma, _ = head(
        rollout_query_embeddings=inputs["rollout_query_embeddings"],
        rollout_positive_document_embeddings=inputs[
            "rollout_positive_document_embeddings"
        ],
        rollout_negative_document_embeddings=inputs[
            "rollout_negative_document_embeddings"
        ],
        policy_query_embeddings=policy_query,
        relevance_labels=inputs["relevance_labels"],
        candidate_mask=inputs["candidate_mask"],
    )
    policy_gradient, sigma_gradient = torch.autograd.grad(
        loss,
        (policy_query, head.log_sigma),
    )
    assert torch.isfinite(policy_gradient).all() and policy_gradient.norm() > 0
    assert torch.isfinite(sigma_gradient) and sigma_gradient.abs() > 0

    restored = GRPO(config=head.config)
    restored.load_runtime_state(head.runtime_state(float(sigma)))
    torch.testing.assert_close(
        restored.current_sigma(torch.device("cpu")),
        sigma,
        rtol=0,
        atol=0,
    )


def test_trainer_checkpoints_post_optimizer_learnable_sigma(tmp_path):
    from transformers import BertConfig, BertModel, TrainingArguments

    from reler.training.grpo_model import GRPOModel
    from reler.training.trainer import GRPOTrainer

    torch.manual_seed(19)
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
            action_components="query",
            group_size=4,
            sigma=0.2,
            sigma_learnable=True,
            sigma_min=0.1,
            sigma_max=0.4,
            reward_type="contrastive",
            contrastive_temperature=0.3,
            advantage_baseline="group",
            rollout_seed=91,
        ),
    )

    def tokens(ids):
        return {
            "input_ids": torch.tensor(ids)[:, None],
            "attention_mask": torch.ones(len(ids), 1, dtype=torch.long),
        }

    batch = {
        "query": tokens([0, 1]),
        "positive_document": tokens([2, 5]),
        "negative_document": tokens([3, 4, 6, 7]),
        "relevance_labels": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "candidate_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    trainer = GRPOTrainer(
        model=wrapper,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            max_steps=1,
            per_device_train_batch_size=2,
            learning_rate=0.05,
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

    state = json.loads((tmp_path / "checkpoint-1" / "grpo_state.json").read_text())
    current = float(wrapper.grpo.current_sigma(torch.device("cpu")))
    assert state["sigma"] == pytest.approx(current, abs=1e-8)
    assert state["sigma"] != pytest.approx(0.2, abs=1e-5)


def test_joint_kl_and_auxiliary_loss_compose_and_keep_padding_gradient_free():
    inputs = _joint_inputs()
    head = GRPO(
        action_components="query;positive,negative",
        group_size=4,
        kappa=9.0,
        reward_type="contrastive",
        contrastive_temperature=0.3,
        kl_coef=0.25,
        aux_infonce_coef=0.4,
        aux_infonce_temperature=0.2,
        rollout_seed=55,
    ).train()
    policies = tuple(
        value.clone().requires_grad_() for value in inputs["policy_values"]
    )
    references = tuple(value + 0.1 for value in inputs["policy_values"])
    positive_mask = torch.zeros_like(inputs["candidate_mask"])
    positive_mask[:, 0] = True

    loss, reward_stats, _, _, kl = head(
        rollout_query_embeddings=inputs["rollout_query_embeddings"],
        rollout_positive_document_embeddings=inputs[
            "rollout_positive_document_embeddings"
        ],
        rollout_negative_document_embeddings=inputs[
            "rollout_negative_document_embeddings"
        ],
        policy_query_embeddings=policies[0],
        policy_positive_document_embeddings=policies[1],
        policy_negative_document_embeddings=policies[2],
        reference_query_embeddings=references[0],
        reference_positive_document_embeddings=references[1],
        reference_negative_document_embeddings=references[2],
        relevance_labels=inputs["relevance_labels"],
        candidate_mask=inputs["candidate_mask"],
        positive_mask=positive_mask,
    )
    expected = (
        reward_stats["train/loss_rl"]
        + 0.25 * kl
        + reward_stats["train/loss_infonce_weighted"]
    )
    torch.testing.assert_close(loss.detach(), expected, rtol=1e-6, atol=1e-7)

    gradients = torch.autograd.grad(loss, policies)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.norm() > 0 for gradient in gradients)
    padded_negatives = ~inputs["candidate_mask"][:, 1:]
    torch.testing.assert_close(
        gradients[2][padded_negatives],
        torch.zeros_like(gradients[2][padded_negatives]),
        rtol=0,
        atol=0,
    )


def test_vmf_kl_matches_closed_form_and_masks_padded_documents():
    torch.manual_seed(72)
    dimension = 6
    concentration = torch.tensor(11.0)
    raw_policy = torch.randn(2, 3, dimension, requires_grad=True)
    raw_reference = torch.randn(2, 3, dimension, requires_grad=True)
    policy = F.normalize(raw_policy, dim=-1)
    valid = torch.tensor([[True, True, False], [True, False, False]])

    actual = GRPO._kl_term(policy, raw_reference, concentration, valid)
    reference = F.normalize(raw_reference.detach(), dim=-1)
    distances = 1.0 - (policy * reference).sum(dim=-1)
    per_sample = (distances * valid).sum(dim=-1) / valid.sum(dim=-1)
    expected = concentration * mean_alignment(dimension, 11.0) * per_sample.mean()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)

    actual_gradient, reference_gradient = torch.autograd.grad(
        actual,
        (raw_policy, raw_reference),
        allow_unused=True,
        retain_graph=True,
    )
    (expected_gradient,) = torch.autograd.grad(expected, raw_policy)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-6, atol=1e-7)
    assert reference_gradient is None
    torch.testing.assert_close(
        actual_gradient[~valid],
        torch.zeros_like(actual_gradient[~valid]),
        rtol=0,
        atol=0,
    )


def _fixed_product_case():
    torch.manual_seed(314)
    batch_size, group_size, slate_size, dimension = 2, 4, 3, 7
    raw_means = torch.randn(
        batch_size,
        slate_size + 1,
        dimension,
        requires_grad=True,
    )
    means = F.normalize(raw_means, dim=-1)
    query_actions = F.normalize(
        torch.randn(batch_size, group_size, dimension),
        dim=-1,
    ).requires_grad_()
    document_actions = F.normalize(
        torch.randn(batch_size, group_size, slate_size, dimension),
        dim=-1,
    ).requires_grad_()
    valid = torch.tensor([[True, True, False], [True, True, True]])
    labels = torch.tensor([[3.0, 2.0, 0.0], [3.0, 1.0, 0.0]])
    kappa = torch.tensor(9.0)
    components = (
        _ActionComponent(
            role="query",
            name="query",
            rollout_embeddings=means[:, 0].detach(),
            policy_embeddings=means[:, 0],
            sampled_embeddings=query_actions,
            kappa=kappa,
        ),
        _ActionComponent(
            role="document",
            name="documents",
            rollout_embeddings=means[:, 1:].detach(),
            policy_embeddings=means[:, 1:],
            sampled_embeddings=document_actions,
            kappa=kappa,
            document_mask=valid,
        ),
    )
    return raw_means, means, query_actions, document_actions, valid, labels, components


def _leave_one_out(values):
    group_size = values.size(1)
    return (values - values.mean(dim=1, keepdim=True)) * (group_size / (group_size - 1))


def _product_scores(query_actions, document_actions):
    return torch.einsum(
        "bqd,bkmd->bqkm",
        query_actions.detach(),
        document_actions.detach(),
    )


def _policy_log_probs(means, query_actions, document_actions, valid, kappa=9.0):
    query_log_prob = kappa * torch.einsum(
        "bd,bgd->bg",
        means[:, 0],
        query_actions.detach(),
    )
    per_document_log_prob = kappa * torch.einsum(
        "bmd,bgmd->bgm",
        means[:, 1:],
        document_actions.detach(),
    )
    per_document_log_prob = per_document_log_prob * valid.unsqueeze(1)
    return query_log_prob, per_document_log_prob


def test_product_score_function_matches_explicit_leave_one_out_oracle():
    (
        raw_means,
        means,
        query_actions,
        document_actions,
        valid,
        labels,
        components,
    ) = _fixed_product_case()
    head = GRPO(
        action_components="query;positive,negative",
        group_size=4,
        kappa=9.0,
        reward_type="ndcg",
        reward_ndcg_k=2,
    )
    actual, _, _, _ = head._compute_component_loss(
        labels,
        None,
        components,
        candidate_mask=valid,
    )

    rewards = compute_reward_from_scores(
        _product_scores(query_actions, document_actions),
        labels,
        reward_type="ndcg",
        k=2,
        candidate_mask=valid,
    )
    query_advantages = _leave_one_out(rewards.mean(dim=2))
    document_advantages = _leave_one_out(rewards.mean(dim=1))
    query_log_prob, per_document_log_prob = _policy_log_probs(
        means,
        query_actions,
        document_actions,
        valid,
    )
    expected = (
        -(query_advantages.detach() * query_log_prob).mean()
        - (document_advantages.detach() * per_document_log_prob.sum(dim=-1)).mean()
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    actual_gradient, query_gradient, document_gradient = torch.autograd.grad(
        actual,
        (raw_means, query_actions, document_actions),
        allow_unused=True,
        retain_graph=True,
    )
    (expected_gradient,) = torch.autograd.grad(expected, raw_means)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    assert query_gradient is None and document_gradient is None
    assert actual_gradient[:, 0].norm() > 0
    assert actual_gradient[:, 1:][valid].norm() > 0
    torch.testing.assert_close(
        actual_gradient[:, 1:][~valid],
        torch.zeros_like(actual_gradient[:, 1:][~valid]),
        rtol=0,
        atol=0,
    )


def test_counterfactual_document_baseline_matches_replacement_reward_oracle():
    (
        raw_means,
        means,
        query_actions,
        document_actions,
        valid,
        labels,
        components,
    ) = _fixed_product_case()
    head = GRPO(
        action_components="query;positive,negative",
        group_size=4,
        kappa=9.0,
        reward_type="ndcg",
        reward_ndcg_k=2,
        document_advantage_baseline="counterfactual",
    )
    actual, _, _, _ = head._compute_component_loss(
        labels,
        None,
        components,
        candidate_mask=valid,
    )

    scores = _product_scores(query_actions, document_actions)
    rewards = compute_reward_from_scores(
        scores,
        labels,
        reward_type="ndcg",
        k=2,
        candidate_mask=valid,
    )
    reference_scores = torch.einsum(
        "bqd,bmd->bqm",
        query_actions.detach(),
        means[:, 1:].detach(),
    )
    document_advantages = rewards.new_zeros(
        rewards.size(0),
        rewards.size(2),
        scores.size(-1),
    )
    for document_index in range(scores.size(-1)):
        replaced_scores = scores.clone()
        replaced_scores[..., document_index] = reference_scores[
            ..., document_index
        ].unsqueeze(-1)
        baseline = compute_reward_from_scores(
            replaced_scores,
            labels,
            reward_type="ndcg",
            k=2,
            candidate_mask=valid,
        )
        document_advantages[..., document_index] = (rewards - baseline).mean(dim=1)
    document_advantages.masked_fill_(~valid.unsqueeze(1), 0)

    query_advantages = _leave_one_out(rewards.mean(dim=2))
    query_log_prob, per_document_log_prob = _policy_log_probs(
        means,
        query_actions,
        document_actions,
        valid,
    )
    expected = (
        -(query_advantages.detach() * query_log_prob).mean()
        - (document_advantages.detach() * per_document_log_prob).sum(dim=-1).mean()
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    actual_gradient = torch.autograd.grad(actual, raw_means, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, raw_means)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    assert (
        document_advantages[valid.unsqueeze(1).expand_as(document_advantages)]
        .abs()
        .sum()
        > 0
    )
    assert actual_gradient[:, 0].norm() > 0
    assert actual_gradient[:, 1:][valid].norm() > 0
    torch.testing.assert_close(
        actual_gradient[:, 1:][~valid],
        torch.zeros_like(actual_gradient[:, 1:][~valid]),
        rtol=0,
        atol=0,
    )


def test_counterfactual_document_baseline_preserves_exact_expected_gradient():
    head = GRPO(
        action_components="positive,negative",
        group_size=4,
        kappa=4.0,
        reward_type="mrr",
        document_advantage_baseline="counterfactual",
    )
    actions = torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)))
    scores = torch.stack(
        (-0.4 + 1.2 * actions[:, 0], -0.2 + 0.8 * actions[:, 1]), dim=-1
    ).unsqueeze(0)
    labels = torch.tensor([[1.0, 0.0]])
    rewards = compute_reward_from_scores(scores, labels, reward_type="mrr")
    advantages = head.reward_evaluator.counterfactual_document_advantages(
        RewardInputs(scores=scores, relevance_labels=labels),
        reference_scores=torch.tensor([[[0.2, 0.1]]]),
        rewards=rewards,
        document_axis=1,
    )

    # Both document policies depend on the same encoder parameters. Enumerating
    # every binary action pair gives the exact expected-reward gradient.
    parameters = torch.tensor([0.4, -0.3], requires_grad=True)
    policy_weights = torch.tensor([[1.0, 0.2], [-0.4, 0.8]])
    probabilities = (policy_weights @ parameters).sigmoid()
    joint_probability = torch.where(
        actions.bool(), probabilities, 1.0 - probabilities
    ).prod(dim=-1)
    (exact_gradient,) = torch.autograd.grad(
        (joint_probability * rewards[0]).sum(), parameters
    )

    score_gradients = (actions - probabilities.detach()).unsqueeze(
        -1
    ) * policy_weights.unsqueeze(0)
    estimated_gradient = (
        joint_probability.detach().unsqueeze(-1).unsqueeze(-1)
        * advantages[0].unsqueeze(-1)
        * score_gradients
    ).sum(dim=(0, 1))
    torch.testing.assert_close(
        estimated_gradient,
        exact_gradient,
        rtol=2e-6,
        atol=2e-8,
    )


def _standardized_leave_one_out(values):
    centered = values - values.mean(dim=1, keepdim=True)
    spread = centered.std(dim=1, keepdim=True, unbiased=False)
    return _leave_one_out(values) / spread


def test_diagonal_normalized_reward_sum_matches_explicit_oracle():
    (
        raw_means,
        means,
        query_actions,
        document_actions,
        valid,
        labels,
        components,
    ) = _fixed_product_case()
    head = GRPO(
        action_components="query;positive,negative",
        group_size=4,
        kappa=9.0,
        rollout="diagonal",
        reward_terms=[
            {"type": "ndcg", "name": "rank", "weight": 0.35, "k": 2},
            {
                "type": "contrastive",
                "name": "margin",
                "weight": 0.65,
                "temperature": 0.3,
            },
        ],
        reward_combine="normalized_sum",
        advantage_norm="per_component",
    )
    actual, _, advantages, _ = head._compute_component_loss(
        labels,
        None,
        components,
        candidate_mask=valid,
    )

    diagonal_scores = torch.einsum(
        "bgd,bgmd->bgm",
        query_actions.detach(),
        document_actions.detach(),
    )
    rank_reward = compute_reward_from_scores(
        diagonal_scores,
        labels,
        reward_type="ndcg",
        k=2,
        candidate_mask=valid,
    )
    margin_reward = compute_reward_from_scores(
        diagonal_scores,
        labels,
        reward_type="contrastive",
        contrastive_temperature=0.3,
        candidate_mask=valid,
    )
    combined_advantage = 0.35 * _standardized_leave_one_out(
        rank_reward
    ) + 0.65 * _standardized_leave_one_out(margin_reward)
    query_log_prob, per_document_log_prob = _policy_log_probs(
        means,
        query_actions,
        document_actions,
        valid,
    )
    expected = (
        -(combined_advantage.detach() * query_log_prob).mean()
        - (combined_advantage.detach() * per_document_log_prob.sum(dim=-1)).mean()
    )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)

    actual_gradient = torch.autograd.grad(actual, raw_means, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, raw_means)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=3e-7)
    torch.testing.assert_close(
        advantages[:, : combined_advantage.size(1)],
        combined_advantage,
        rtol=2e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        advantages[:, combined_advantage.size(1) :],
        combined_advantage,
        rtol=2e-5,
        atol=1e-6,
    )
    assert actual_gradient[:, 0].norm() > 0
    assert actual_gradient[:, 1:][valid].norm() > 0
    torch.testing.assert_close(
        actual_gradient[:, 1:][~valid],
        torch.zeros_like(actual_gradient[:, 1:][~valid]),
        rtol=0,
        atol=0,
    )
