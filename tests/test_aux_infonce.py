"""CPU gradient checks; no model downloads, training data, or accelerator needed."""

import json
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reler.config import BaselineArguments, RLArguments
from reler.objectives.contrastive import (
    aux_infonce_contract,
    auxiliary_infonce_loss,
    compute_infonce_loss,
    cross_query_scores,
)
from reler.training.grpo_model import GRPO, GRPOModel
from reler.training.supervised import BaselineModel
from reler.training.trainer import GRPOTrainer, restore_exploration_state


def test_multi_positive_loss_and_padding_gradients():
    scores = torch.tensor([[1.3, 0.9, -0.2, 99.0]], requires_grad=True)
    positives = torch.tensor([[True, True, False, False]])
    valid = torch.tensor([[True, True, True, False]])
    loss = compute_infonce_loss(scores, positives, temperature=0.5, candidate_mask=valid).sum()
    expected = (F.softplus(torch.tensor(-3.0)) + F.softplus(torch.tensor(-2.2))) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (scores.grad[0, :2] < 0).all()
    assert scores.grad[0, 2] > 0
    assert scores.grad[0, 3] == 0


def test_in_batch_mask_and_detached_cross_document():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    documents = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]], requires_grad=True)
    positives = torch.ones(2, 1, dtype=torch.bool)
    cross = torch.tensor([[False, True], [False, False]])
    loss = auxiliary_infonce_loss(
        queries, documents, positives, temperature=1.0,
        use_in_batch_negatives=True, in_batch_positive_mask=cross,
    )
    torch.testing.assert_close(loss, F.softplus(torch.tensor(1.0)) / 2)
    loss.backward()
    assert queries.grad[0].norm() > 0
    assert documents.grad[0].norm() > 0
    assert queries.grad[1].norm() == 0
    assert documents.grad[1].norm() == 0  # Used by query 0 only as a detached negative.

    zero = auxiliary_infonce_loss(
        queries, documents, positives, use_in_batch_negatives=True,
        in_batch_positive_mask=torch.zeros_like(cross),
    )
    gradients = torch.autograd.grad(zero, (queries, documents))
    assert zero == 0
    assert all(torch.isfinite(g).all() and g.norm() == 0 for g in gradients)


def test_auxiliary_scores_stay_fp32_under_autocast():
    torch.manual_seed(8)
    queries = torch.randn(2, 7, dtype=torch.bfloat16, requires_grad=True)
    documents = torch.randn(2, 3, 7, dtype=torch.bfloat16, requires_grad=True)
    positives = torch.tensor([[True, True, False], [True, False, False]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = auxiliary_infonce_loss(queries, documents, positives, temperature=0.03)
    expected = compute_infonce_loss(
        torch.einsum("bd,bmd->bm", queries.float(), documents.float()), positives, 0.03,
    ).mean()
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert torch.isfinite(queries.grad).all() and torch.isfinite(documents.grad).all()


def test_auxiliary_requires_binary_positive_identities():
    with pytest.raises(ValueError):
        auxiliary_infonce_loss(torch.ones(1, 2), torch.ones(1, 2, 2), None)


@pytest.mark.parametrize("groups", [
    (("query",),), (("positive",),), (("negative",),),
    (("query",), ("positive",)), (("query",), ("positive", "negative")),
    (("query",), ("positive",), ("negative",)),
])
@pytest.mark.parametrize("strong", [False, True])
def test_zero_reward_advantage_still_gets_exact_auxiliary_gradient(groups, strong):
    torch.manual_seed(5)
    query, positive, negative = (
        torch.randn(2, 5, requires_grad=True),
        torch.randn(2, 1, 5, requires_grad=True),
        torch.randn(2, 3, 5, requires_grad=True),
    )
    positives = torch.tensor([[True, False, True, False], [True, False, False, False]])
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    metadata = [negative_metadata(['d0', 'd1', 'd2', None]), negative_metadata(['d3', 'd4', 'd5', 'd6'])]
    coefficient, temperature = 0.4, 0.2
    head = GRPO(
        action_components=groups, group_size=3, kappa=12, reward_type="mrr",
        aux_infonce_coef=coefficient, aux_infonce_temperature=temperature,
        aux_infonce_strong_negatives=strong,
        rollout_seed=17,
    )
    values = dict(
        rollout_query_embeddings=query.detach(),
        rollout_positive_document_embeddings=positive.detach(),
        rollout_negative_document_embeddings=negative.detach(),
        policy_query_embeddings=query,
        policy_positive_document_embeddings=positive,
        policy_negative_document_embeddings=negative,
        # All candidates are relevant to this ranking reward: MRR is identically 1.
        # Binary auxiliary identities remain separate from these reward labels.
        relevance_labels=torch.ones(2, 4), positive_mask=positives, candidate_mask=valid,
        cross_batch_metadata=metadata,
    )
    loss, rewards, advantages, _, _ = head(**values)
    assert rewards["reward_mean"] == 1
    assert advantages["advantages_std"] == 0
    assert rewards["train/loss_rl"] == 0
    actual = torch.autograd.grad(loss, (query, positive, negative), allow_unused=True)

    active = {role for group in groups for role in group}
    q = F.normalize(query, dim=-1) if "query" in active else F.normalize(query.detach(), dim=-1)
    p = F.normalize(positive, dim=-1) if "positive" in active else F.normalize(positive.detach(), dim=-1)
    n = F.normalize(negative, dim=-1) if "negative" in active else F.normalize(negative.detach(), dim=-1)
    expected_loss = coefficient * auxiliary_infonce_loss(
        q, torch.cat((p, n), dim=1), positives, valid, temperature=temperature,
        strong_negatives=strong, cross_batch_metadata=metadata,
    )
    expected = torch.autograd.grad(expected_loss, (query, positive, negative), allow_unused=True)
    torch.testing.assert_close(loss, expected_loss)
    for name, got, want in zip(("query", "positive", "negative"), actual, expected):
        if name not in active:
            assert got is None and want is None
        else:
            assert got.norm() > 0
            torch.testing.assert_close(got, want)
    assert actual[2] is None or actual[2][0, 2].norm() == 0  # Padding never gets a gradient.


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.table = nn.Embedding(16, 5)
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.table(input_ids))


def tokens(ids):
    ids = torch.tensor(ids).reshape(-1, 1)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.mark.parametrize("strong", [False, True])
def test_joint_wrapper_preserves_reward_and_auxiliary_gradient(strong):
    torch.manual_seed(1)
    model = TinyEncoder()
    config = dict(
        action_components="query;positive,negative",
        group_size=3, kappa=12, reward_type="ndcg",
        aux_infonce_temperature=0.2, aux_infonce_use_in_batch_negatives=True,
        aux_infonce_strong_negatives=strong,
    )
    args = RLArguments(**config, aux_infonce_coef=0.3)
    wrapper = GRPOModel(model, rl_args=args)
    batch = dict(
        query=tokens([0, 1]), relevance_labels=torch.tensor([[3., 1., 0.], [3., 0., 0.]]),
        positive_mask=torch.tensor([[True, True, False], [True, False, False]]),
        candidate_mask=torch.ones(2, 3, dtype=torch.bool),
        in_batch_positive_mask=~torch.eye(2, dtype=torch.bool),
        cross_batch_metadata=[negative_metadata([f'd{i}' for i in range(3)]),
                              negative_metadata([f'd{i}' for i in range(3, 6)])],
    )
    batch.update(positive_document=tokens([2, 5]), negative_document=tokens([3, 4, 6, 7]))
    torch.manual_seed(42)
    output = wrapper(**batch)
    assert model.calls == 2
    auxiliary = output.reward_terms["train/loss_infonce_weighted"]
    assert auxiliary > 0
    torch.testing.assert_close(output.loss.detach(), output.reward_terms["train/loss_rl"] + auxiliary)
    output.loss.backward()
    assert model.table.weight.grad.norm() > 0

    # Disabling the loss preserves exactly the same rollout reward and RL objective.
    wrapper.grpo.aux_infonce_coef = 0
    batch.pop("positive_mask")  # Disabled configs don't acquire a new input requirement.
    torch.manual_seed(42)
    disabled = wrapper(**batch)
    for name in ("reward_mean", "reward_std", "advantages_mean", "advantages_std"):
        torch.testing.assert_close(getattr(output, name), getattr(disabled, name), rtol=0, atol=0)
    torch.testing.assert_close(disabled.loss, output.reward_terms["train/loss_rl"], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"aux_infonce_coef": -1}, {"aux_infonce_coef": float("nan")},
    {"aux_infonce_coef": float("inf")}, {"aux_infonce_temperature": 0},
    {"aux_infonce_temperature": float("nan")},
])
def test_invalid_auxiliary_configuration(kwargs):
    with pytest.raises(ValueError):
        RLArguments(**kwargs)


def test_auxiliary_resume_contract(tmp_path):
    head = GRPO(group_size=3, aux_infonce_coef=0.2)
    model = SimpleNamespace(grpo=head)
    payload = {"exploration": head.exploration.state_dict(), "estimator": {}}
    path = tmp_path / "exploration_state.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        restore_exploration_state(model, tmp_path)  # Old RL-only checkpoint.
    payload["aux_infonce"] = aux_infonce_contract(head)
    path.write_text(json.dumps(payload))
    restore_exploration_state(model, tmp_path)
    head.aux_infonce_strong_negatives = True
    with pytest.raises(ValueError):
        restore_exploration_state(model, tmp_path)
    head.aux_infonce_strong_negatives = False
    head.aux_infonce_temperature = 0.1
    with pytest.raises(ValueError):
        restore_exploration_state(model, tmp_path)
    head.aux_infonce_coef = 0
    with pytest.raises(ValueError):
        restore_exploration_state(model, tmp_path)
    payload.pop("aux_infonce")
    path.write_text(json.dumps(payload))
    restore_exploration_state(model, tmp_path)


@pytest.mark.parametrize("estimator", ["score_function", "conditional_projection"])
def test_trainer_updates_logs_and_saves_auxiliary_contract(tmp_path, estimator):
    from transformers import BertConfig, BertModel, TrainingArguments

    torch.manual_seed(7)
    backbone = BertModel(BertConfig(
        vocab_size=16, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, hidden_dropout_prob=0, attention_probs_dropout_prob=0,
    ))
    wrapper = GRPOModel(backbone, RLArguments(
        action_components="query;positive,negative", group_size=2, kappa=12,
        aux_infonce_coef=0.3, aux_infonce_temperature=0.2,
        gradient_estimator=estimator,
    ))
    batch = dict(
        query=tokens([0, 1]), positive_document=tokens([2, 5]),
        negative_document=tokens([3, 4, 6, 7]),
        relevance_labels=torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
        positive_mask=torch.tensor([[True, False, False], [True, False, False]]),
        candidate_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    before = backbone.embeddings.word_embeddings.weight.detach().clone()
    trainer = GRPOTrainer(
        model=wrapper,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, max_steps=1,
            per_device_train_batch_size=2, learning_rate=1e-3, logging_steps=1,
            save_steps=1, report_to=[], disable_tqdm=True, remove_unused_columns=False,
            gradient_checkpointing=True,
        ),
        train_dataset=[0, 1], data_collator=lambda _: batch,
    )
    trainer.train()
    assert (backbone.embeddings.word_embeddings.weight.detach() - before).norm() > 0
    step = next(row for row in trainer.state.log_history if "train/loss_infonce" in row)
    assert step["train/loss_infonce"] > 0
    assert step["train/loss_infonce_weighted"] == pytest.approx(0.3 * step["train/loss_infonce"], abs=1e-4)
    checkpoint = tmp_path / "checkpoint-1"
    payload = json.loads((checkpoint / "exploration_state.json").read_text())
    assert payload["aux_infonce"]["coefficient"] == 0.3
    assert payload["estimator"]["gradient_estimator"] == estimator
    restore_exploration_state(wrapper, checkpoint)


def negative_metadata(keys, ids=None, source='s', known_ids=(), positives=()):
    return dict(keys=keys, ids=ids or [None] * len(keys), source=source,
                known_ids=list(known_ids), known_positive_keys=list(positives))


def test_strong_negative_filter_and_document_gradient_scope():
    queries = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    documents = torch.randn(2, 5, 2, requires_grad=True)
    metadata = [
        negative_metadata(['own', 'own-negative', None, None, None], known_ids=['relevant-id'], positives=['known-positive']),
        negative_metadata(['other-positive', 'other-negative', 'known-positive', 'own-negative', 'id-match'],
                          ids=[None, None, None, None, 'relevant-id']),
    ]
    scores, valid, weight = cross_query_scores(
        queries, documents, metadata, include_negatives=True, cross_device=False, detach_documents=False,
    )
    assert weight == 1
    assert valid[0].tolist() == [False] * 5 + [True, True, False, False, False]
    (scores[0] * valid[0]).sum().backward()
    assert documents.grad[1, 0].norm() > 0 and documents.grad[1, 1].norm() > 0
    assert documents.grad[0].norm() == 0 and documents.grad[1, 2:].norm() == 0
    documents.grad = None
    detached, valid, _ = cross_query_scores(
        queries, documents, metadata, include_negatives=True, cross_device=False, detach_documents=True,
    )
    detached.masked_select(valid).sum().backward()
    assert documents.grad is None
    positive_scores, positive_valid, _ = cross_query_scores(
        queries, documents, metadata, include_negatives=False, cross_device=False, detach_documents=False,
    )
    assert positive_scores.shape == (2, 2)
    assert positive_valid.tolist() == [[False, True], [True, False]]


def _strong_batch(start, stop, width):
    rows = [[3, 6, 7], [4, 8, 9], [5, 10, 0]][start:stop]
    ids = torch.tensor(rows)[:, :width]
    candidate_mask = torch.ones_like(ids, dtype=torch.bool)
    if stop == 3 and width == 3:
        candidate_mask[-1, -1] = False
    def tokens(values):
        values = torch.as_tensor(values).reshape(-1, 1)
        return dict(input_ids=values, attention_mask=torch.ones_like(values))
    positives = torch.zeros_like(candidate_mask)
    positives[:, 0] = True
    if start <= 1 < stop:
        positives[1 - start, 1] = True  # Multi-positive query.
    return dict(query=tokens(range(start, stop)), positive_document=tokens(ids[:, 0]),
                negative_document=tokens(ids[:, 1:]), candidate_mask=candidate_mask,
                positive_mask=positives, relevance_labels=positives.float(),
                cross_batch_metadata=[negative_metadata(
                    [f'd{x}' if valid else None for x, valid in zip(row.tolist(), mask.tolist())]
                ) for row, mask in zip(ids, candidate_mask)])


def _distributed_strong_worker(rank, rendezvous, uneven, auxiliary):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(702)
        args = BaselineArguments(baseline_use_in_batch_negatives=True, baseline_temperature=.5,
                                 baseline_in_batch_include_negatives=True, baseline_cross_device_negatives=True,
                                 baseline_detach_in_batch_documents=False)
        if auxiliary:
            rl_args = RLArguments(action_components='query;positive,negative', group_size=2,
                                  kappa=12, reward_type='mrr', aux_infonce_coef=1.,
                                  aux_infonce_temperature=.5, aux_infonce_strong_negatives=True)
            model = GRPOModel(TinyEncoder(), rl_args)
        else:
            model = BaselineModel(TinyEncoder(), args)
        parallel = DistributedDataParallel(model)
        if uneven:
            start, stop, width = ((0, 2, 3) if rank == 0 else (2, 3, 2))
            total = 3
        else:
            start, stop, width = rank, rank + 1, 3
            total = 2
        batch = _strong_batch(start, stop, width)
        if auxiliary:
            # All candidates relevant to MRR => zero RL advantage. The auxiliary
            # binary labels stay independent, isolating the direct InfoNCE gradient.
            batch['relevance_labels'] = torch.ones_like(batch['relevance_labels'])
        loss = parallel(**batch).loss
        loss.backward()
        actual_gradient = model.model.table.weight.grad.clone()
        dist.all_reduce(loss.detach())  # Compare averaged rank losses below.
        global_loss = loss.detach() / 2

        # An independent single-process global objective with complete document gradients.
        torch.manual_seed(702)
        reference = TinyEncoder()
        global_batch = _strong_batch(0, total, 3)
        queries = F.normalize(reference.table(torch.arange(total)).float(), dim=-1)
        doc_ids = torch.tensor([[3, 6, 7], [4, 8, 9], [5, 10, 0]])[:total]
        documents = F.normalize(reference.table(doc_ids).float(), dim=-1)
        own_scores = torch.einsum('bd,bmd->bm', queries, documents)
        cross_scores = queries @ documents.reshape(-1, 5).T
        valid = global_batch['candidate_mask']
        cross_valid = valid.reshape(1, -1).expand(total, -1).clone()
        for i in range(total):
            cross_valid[i, i * 3:(i + 1) * 3] = False
        scores = torch.cat((own_scores, cross_scores), dim=-1)
        positives = torch.cat((global_batch['positive_mask'], torch.zeros_like(cross_valid)), dim=-1)
        expected = compute_infonce_loss(scores, positives, .5, torch.cat((valid, cross_valid), dim=-1)).mean()
        expected.backward()
        torch.testing.assert_close(global_loss, expected.detach(), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_gradient, reference.table.weight.grad, atol=1e-6, rtol=1e-5)
        assert actual_gradient[6:11].norm() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('uneven', [False, True])
@pytest.mark.parametrize('auxiliary', [False, True])
def test_cross_device_strong_cl_matches_global_loss_and_encoder_gradients(tmp_path, uneven, auxiliary):
    torch.multiprocessing.spawn(_distributed_strong_worker,
                               args=(str(tmp_path / 'rendezvous'), uneven, auxiliary), nprocs=2, join=True)
