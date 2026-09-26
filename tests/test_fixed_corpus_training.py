"""Contracts for query-only GRPO against immutable document embeddings."""

from types import SimpleNamespace

import pytest
import torch
from candidate_helpers import candidate_record, document_key
from torch import nn

from reler.config import RLArguments
from reler.data.embedding import EmbeddingDataCollator
from reler.fixed_corpus.data import FixedCorpusDataCollator
from reler.fixed_corpus.training import (
    FixedCorpusGRPOModel,
    validate_fixed_corpus_grpo_arguments,
)


class ToyTokenizer:
    pad_token = "[PAD]"

    def __call__(self, texts, **_):
        ids = [[1 + sum(map(ord, text)) % 13] for text in texts]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones(len(ids), 1, dtype=torch.long),
        }


class TinyQueryEncoder(nn.Module):
    def __init__(self, dimension=4):
        super().__init__()
        self.config = SimpleNamespace()
        self.embedding = nn.Embedding(32, dimension)
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class MemoryIndex:
    def __init__(self):
        self.dimension = 4
        self.document_to_ordinal = {
            "positive": 0,
            "negative": 1,
            "other": 2,
            document_key("positive"): 0,
            document_key("negative"): 1,
            document_key("other"): 2,
            "p": 0,
            "n": 1,
            "o": 2,
        }
        self.values = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            requires_grad=True,
        )
        self.lookup_calls = 0

    def lookup_embeddings(self, ordinals, *, device=None):
        self.lookup_calls += 1
        return self.values[ordinals.cpu()].to(device=device)


def test_fixed_collator_preserves_joint_candidate_layout_without_document_tokens():
    index = MemoryIndex()
    collator = FixedCorpusDataCollator(ToyTokenizer(), index, append_token="none")
    first = candidate_record(
        ["positive", "negative"], query="q1", rank_labels=[1, 2],
        document_keys=["positive", "negative"], document_ids=["p", "n"],
    )
    second = candidate_record(["positive", "other", "negative"], query="q2",
                              relevance=[1, 1, 0])
    batch = collator([first, second])

    assert batch["candidate_ordinals"].tolist() == [[0, 1, -1], [0, 2, 1]]
    assert batch["candidate_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["positive_mask"].tolist() == [[True, False, False], [True, True, False]]
    joint = EmbeddingDataCollator(ToyTokenizer(), append_token="none")([first, second])
    for field in ("candidate_mask", "positive_mask", "relevance_labels", "rank_labels",
                  "in_batch_positive_mask", "in_batch_candidate_mask"):
        torch.testing.assert_close(batch[field], joint[field])
    assert "positive_document" not in batch and "negative_document" not in batch


def test_fixed_collator_prefers_explicit_document_ids_and_rejects_ambiguity():
    index = MemoryIndex()
    collator = FixedCorpusDataCollator(ToyTokenizer(), index, append_token="none")
    record = candidate_record(document_ids=["p", "n"],
                              document_keys=["unknown-key", "also-unknown"])
    assert collator([record])["candidate_ordinals"].tolist() == [[0, 1]]

    index.document_to_ordinal["p"] = 2
    ambiguous = FixedCorpusDataCollator(ToyTokenizer(), index, append_token="none")
    with pytest.raises(ValueError):
        ambiguous(
            [
                {
                    **record,
                    "document_keys": ["positive", "also-unknown"],
                }
            ]
        )


def test_fixed_collator_rejects_candidates_missing_from_the_index():
    collator = FixedCorpusDataCollator(
        ToyTokenizer(), MemoryIndex(), append_token="none"
    )
    record = candidate_record(["positive", "outside"])
    with pytest.raises(KeyError):
        collator([record])


def test_fixed_grpo_only_encodes_queries_and_keeps_document_vectors_out_of_autograd():
    torch.manual_seed(2)
    encoder = TinyQueryEncoder()
    index = MemoryIndex()
    model = FixedCorpusGRPOModel(
        encoder,
        index,
        RLArguments(
            action_components="query", group_size=3, kappa=6.0, reward_type="ndcg"
        ),
    )
    batch = {
        "query": {
            "input_ids": torch.tensor([[1], [2]]),
            "attention_mask": torch.ones(2, 1, dtype=torch.long),
        },
        "candidate_ordinals": torch.tensor([[0, 1], [0, 2]]),
        "candidate_mask": torch.ones(2, 2, dtype=torch.bool),
        "relevance_labels": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "positive_mask": torch.tensor([[True, False], [True, False]]),
        "rank_labels": torch.tensor([[2.0, 1.0], [2.0, 1.0]]),
    }
    output = model(**batch)
    output.loss.backward()

    assert encoder.calls == 1
    assert index.lookup_calls == 1
    assert encoder.embedding.weight.grad is not None
    assert index.values.grad is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_components": "query;positive,negative"},
        {"gradient_estimator": "conditional_projection"},
        {"reward_cross_device_negatives": True},
        {"cross_query_document_gradients": True},
    ],
)
def test_fixed_grpo_rejects_options_that_need_trainable_or_cross_rank_documents(kwargs):
    base = {"action_components": "query", "group_size": 3, "kappa": 4.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        validate_fixed_corpus_grpo_arguments(RLArguments(**base))
