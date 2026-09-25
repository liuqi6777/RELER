"""Hard candidate score gaps must survive mixed-precision encoder execution."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reler.config import BaselineArguments
from reler.data.protocol import pool_embeddings
from reler.objectives.contrastive import (
    compute_in_batch_positive_scores,
    compute_infonce_loss,
)
from reler.training.grpo_model import GRPO, _ActionComponent
from reler.training.supervised import (
    BaselineModel,
    compute_lambdaloss_loss,
    compute_ranknet_loss,
)


@pytest.mark.parametrize("pooling", ["last", "mean", "cls"])
def test_pooling_and_normalization_promote_before_reduction(pooling):
    torch.manual_seed(82)
    hidden = torch.randn(2, 4, 7).bfloat16().requires_grad_()
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = pool_embeddings(hidden, mask, pooling_method=pooling)
    expected = pool_embeddings(hidden.float(), mask, pooling_method=pooling)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad[0, 2:].norm() == 0


@pytest.mark.parametrize("device_name", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_reward_ranking_preserves_near_ties_under_autocast(device_name):
    device = torch.device(device_name)
    q = torch.tensor([[1., 0.]], device=device)
    docs = F.normalize(torch.tensor([[[.8, .6], [.8003, .5996]]], device=device), dim=-1)
    qc = _ActionComponent("query", q)
    dc = _ActionComponent("document", docs)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast(device_name, dtype=torch.bfloat16):
            rounded = torch.einsum("bd,bmd->bm", q, docs)
            actual = GRPO._compute_score_table(qc, dc)
        assert rounded[0, 0] == rounded[0, 1]
        assert actual.dtype == torch.float32
        assert actual[0, 1] > actual[0, 0]
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("loss_fn", [compute_infonce_loss, compute_ranknet_loss, compute_lambdaloss_loss])
def test_losses_promote_quantized_scores_and_ignore_infinite_padding(loss_fn):
    scores = torch.tensor([[.4, .2, -.3, float('-inf')]], dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[2., 1., 0., 0.]])
    valid = torch.tensor([[True, True, True, False]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss_fn(scores, labels, candidate_mask=valid)
    reference = loss_fn(scores.float(), labels, candidate_mask=valid)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.sum().backward()
    assert torch.isfinite(scores.grad).all() and scores.grad[0, -1] == 0


class EmbeddingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.values = torch.nn.Embedding(8, 5)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.values(input_ids).bfloat16())


def tokens(ids):
    ids = torch.tensor(ids).reshape(-1, 1)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


@pytest.mark.parametrize("objective", ["infonce", "ranknet", "lambdaloss"])
def test_baseline_complete_loss_and_gradients_match_float32_scoring(objective):
    torch.manual_seed(83)
    encoder = EmbeddingEncoder()
    args = BaselineArguments(baseline_loss=objective, baseline_use_in_batch_negatives=True)
    wrapper = BaselineModel(encoder, args)
    batch = dict(query=tokens([0, 1]), positive_document=tokens([2, 3]), negative_document=tokens([4, 5, 6, 7]),
                 relevance_labels=torch.tensor([[1., 0., 0.], [1., 0., 0.]]))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = wrapper(**batch).loss
    actual_grad = torch.autograd.grad(actual, encoder.values.weight)[0]
    reference = wrapper(**batch).loss
    reference_grad = torch.autograd.grad(reference, encoder.values.weight)[0]
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual_grad, reference_grad, atol=0, rtol=0)


def test_cross_scores_use_float32():
    torch.manual_seed(22)
    queries = torch.randn(2, 5).bfloat16().requires_grad_()
    documents = torch.randn(4, 5).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cross = compute_in_batch_positive_scores(queries, documents[:2])
    torch.testing.assert_close(cross, compute_in_batch_positive_scores(queries.float(), documents[:2].float()), atol=0, rtol=0)
