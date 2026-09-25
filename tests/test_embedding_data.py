"""Behavioral contracts for listwise data, identity masks, and padded slates."""

import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from reler.config import BaselineArguments, RLArguments
from reler.data.embedding import (
    EmbeddingDataCollator,
    normalize_listwise_record,
    record_to_slate,
)
from reler.training.grpo_model import GRPOModel
from reler.training.supervised import BaselineModel, compute_ranknet_loss


class ToyTokenizer:
    pad_token = "[PAD]"

    def __call__(self, texts, **_):
        token_ids = [[1 + sum(map(ord, text)) % 29] if text else [0] for text in texts]
        return {
            "input_ids": torch.tensor(token_ids),
            "attention_mask": torch.ones(len(token_ids), 1, dtype=torch.long),
        }


class TrackingBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.config = SimpleNamespace()
        self.encoded_rows = []

    def forward(self, input_ids, attention_mask):
        self.encoded_rows.append(input_ids.size(0))
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def collator(relevance_scheme="binary"):
    return EmbeddingDataCollator(
        ToyTokenizer(), append_token="none", relevance_scheme=relevance_scheme
    )


def test_annotated_positive_and_teacher_order_remain_independent():
    # Document 3 is the annotated positive but document 2 is the teacher favorite.
    record = normalize_listwise_record(
        {
            "query": "query",
            "document": ["a", "b", "c", "d", "e", "f"],
            "ranking": [2, 1, 4, 5, 6, 3],
            "pos_index": 3,
        }
    )
    binary = collator("binary")([record])
    graded = collator("graded")([record])

    expected_positive = [[True, False, False, False, False, False]]
    expected_ranks = [[1, 5, 6, 4, 3, 2]]
    assert binary["positive_mask"].tolist() == expected_positive
    assert graded["positive_mask"].tolist() == expected_positive
    assert binary["rank_labels"].tolist() == expected_ranks
    assert graded["rank_labels"].tolist() == expected_ranks
    assert binary["relevance_labels"].tolist() == [[1, 0, 0, 0, 0, 0]]
    assert graded["relevance_labels"].tolist() == [[1, 2, 3, 2, 2, 2]]

    model = BaselineModel(TrackingBackbone(), BaselineArguments())
    torch.testing.assert_close(model(**binary).loss, model(**graded).loss)


@pytest.mark.parametrize("invalid", [0, 4, -1, True, 1.5, "2", None])
def test_listwise_positive_index_is_validated(invalid):
    raw = {
        "query": "query",
        "document": ["a", "b", "c"],
        "ranking": [2, 3, 1],
        "pos_index": invalid,
    }
    with pytest.raises(ValueError, match="pos_index"):
        normalize_listwise_record(raw)


def test_binary_listwise_data_requires_an_annotated_positive():
    record = normalize_listwise_record(
        {
            "query": "query",
            "document": ["a", "b", "c"],
            "ranking": [2, 3, 1],
        }
    )
    with pytest.raises(ValueError, match="pos_index"):
        collator("binary")([record])


def test_text_identity_filters_cross_query_false_negatives_without_document_ids():
    records = [
        normalize_listwise_record(
            {
                "query": "q",
                "document": ["Alpha", "Beta"],
                "ranking": [1, 2],
                "pos_index": 1,
            }
        ),
        normalize_listwise_record(
            {
                "query": "r",
                "document": [" alpha ", "Gamma", "Delta"],
                "ranking": [1, 2, 3],
                "pos_index": 1,
            }
        ),
    ]
    batch = collator()(records)
    assert batch["in_batch_positive_mask"].tolist() == [
        [False, False],
        [False, False],
    ]
    assert batch["in_batch_candidate_mask"][0, 1].tolist() == [
        False,
        True,
        True,
    ]
    assert batch["in_batch_candidate_mask"][1, 0].tolist() == [
        False,
        True,
        False,
    ]


def test_raw_document_metadata_does_not_change_joint_candidate_filtering():
    raw = [
        {
            "query": "q1",
            "document": ["alpha", "beta"],
            "ranking": [1, 2],
            "pos_index": 1,
            "source": "source",
        },
        {
            "query": "q2",
            "document": ["gamma", "delta"],
            "ranking": [1, 2],
            "pos_index": 1,
            "source": "source",
        },
    ]
    plain = [normalize_listwise_record(record) | {"source": "source"} for record in raw]
    with_ids = [
        normalize_listwise_record(
            {
                **record,
                "document_ids": ["shared", f"negative-{index}"],
                "document_keys": [f"positive-{index}", f"negative-{index}"],
                "known_document_ids": ["shared"],
                "known_positive_keys": [f"positive-{index}"],
            },
            preserve_document_metadata=True,
        )
        | {"source": "source"}
        for index, record in enumerate(raw)
    ]

    expected = collator()(plain)
    actual = collator()(with_ids)
    torch.testing.assert_close(
        actual["in_batch_positive_mask"], expected["in_batch_positive_mask"]
    )
    torch.testing.assert_close(
        actual["in_batch_candidate_mask"], expected["in_batch_candidate_mask"]
    )


def test_mined_records_keep_all_known_positives_and_do_not_rank_negatives():
    first = record_to_slate(
        {
            "query": "q",
            "pos": ["chosen", "Other Positive"],
            "pos_scores": [2.0, 1.0],
            "neg": ["n1", "n2"],
            "neg_scores": [0.9, 0.1],
        },
        slate_size=3,
        rng=random.Random(42),
    )
    second = record_to_slate(
        {
            "query": "r",
            "pos": [" other   positive "],
            "neg": ["n3", "n4"],
        },
        slate_size=3,
        rng=random.Random(42),
    )
    first["source"] = second["source"] = "nq"

    batch = collator()([first, second])
    assert batch["relevance_labels"].tolist() == [[1, 0, 0], [1, 0, 0]]
    assert batch["rank_labels"].tolist() == [[1, 0, 0], [1, 0, 0]]
    assert batch["in_batch_positive_mask"].tolist() == [
        [False, False],
        [True, False],
    ]

    scores = torch.tensor([[0.2, 0.7, -0.4]])
    expected = compute_ranknet_loss(scores, batch["rank_labels"][:1])
    actual = compute_ranknet_loss(scores[:, [0, 2, 1]], batch["rank_labels"][:1])
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="binary relevance"):
        collator("graded")([first, second])


def test_prepared_records_use_precomputed_document_identity():
    record = {
        "schema": "embedding_candidates_v2",
        "source": "source",
        "query": "query",
        "document": ["positive", "negative"],
        "document_ids": ["p", "n"],
        "document_keys": ["positive-key", "negative-key"],
        "known_document_ids": ["p"],
        "known_positive_keys": ["positive-key"],
        "relevance": [1, 0],
        "graded_relevance": [3, 0],
        "rank_labels": [2, 1],
    }
    with patch(
        "reler.data.embedding.document_key",
        side_effect=AssertionError(
            "prepared records must not hash text at collation time"
        ),
    ):
        batch = collator()([record])
    assert batch["candidate_mask"].tolist() == [[True, True]]
    assert batch["positive_mask"].tolist() == [[True, False]]


@pytest.mark.parametrize("objective", ["supervised", "grpo"])
def test_variable_slates_encode_only_real_candidates(objective):
    records = [
        normalize_listwise_record(
            {
                "query": "q1",
                "document": ["a", "b"],
                "ranking": [1, 2],
                "pos_index": 1,
            }
        ),
        normalize_listwise_record(
            {
                "query": "q2",
                "document": ["c", "d", "e", "f"],
                "ranking": [3, 1, 4, 2],
                "pos_index": 3,
            }
        ),
    ]
    batch = collator()(records)
    backbone = TrackingBackbone()
    if objective == "supervised":
        model = BaselineModel(backbone, BaselineArguments())
    else:
        model = GRPOModel(
            backbone,
            RLArguments(
                action_components="query;positive,negative",
                group_size=4,
                kappa=5.0,
                reward_type="ndcg",
            ),
        )
        torch.manual_seed(5)

    output = model(**batch)
    output.loss.backward()
    assert backbone.encoded_rows == [2, 6]
    assert torch.isfinite(output.loss)
    assert torch.isfinite(backbone.embedding.weight.grad).all()
