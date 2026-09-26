"""Prepared labels, document identity, and variable candidate training contracts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from candidate_helpers import candidate_record

from reler.config import BaselineArguments, RLArguments
from reler.data.candidates import validate_candidate_record
from reler.data.embedding import EmbeddingDataCollator, build_candidate_layout
from reler.training.grpo_model import GRPOModel
from reler.training.supervised import BaselineModel


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


def test_multiple_positives_and_teacher_order_remain_independent():
    record = candidate_record(
        ["a", "b", "c", "d", "e", "f"],
        relevance=[1, 1, 0, 0, 0, 0],
        rank_labels=[1, 5, 6, 4, 3, 2],
        graded_relevance=[1, 2, 3, 2, 2, 2],
    )
    original = deepcopy(record)
    binary = collator("binary")([record])
    graded = collator("graded")([record])

    expected_positive = [[True, True, False, False, False, False]]
    for batch in (binary, graded):
        assert batch["positive_mask"].tolist() == expected_positive
        assert batch["rank_labels"].tolist() == [[1, 5, 6, 4, 3, 2]]
    assert binary["relevance_labels"].tolist() == [[1, 1, 0, 0, 0, 0]]
    assert graded["relevance_labels"].tolist() == [[1, 2, 3, 2, 2, 2]]
    assert record == original
    model = BaselineModel(TrackingBackbone(), BaselineArguments())
    torch.testing.assert_close(model(**binary).loss, model(**graded).loss)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "embedding_candidates_v1"},
        {"schema": None},
        {"query": ""},
        {"query": None},
        {"query": 2},
        {"id": None},
        {"source": 1},
        {"document": ["only"]},
        {"document": ["a", ""]},
        {"document_ids": ["a"]},
        {"document_ids": ["a", None]},
        {"document_keys": []},
        {"known_document_ids": "a"},
        {"known_document_ids": [1]},
        {"relevance": [0, 1]},
        {"relevance": [1, 1]},
        {"relevance": [1, 2]},
        {"relevance": [True, False]},
        {"relevance": [1.0, 0.0]},
        {"graded_relevance": [3]},
        {"graded_relevance": [3, -1]},
        {"graded_relevance": [4, 0]},
        {"graded_relevance": [float("nan"), 0]},
        {"rank_labels": [2, 2]},
        {"rank_labels": [2, 0]},
        {"rank_labels": [2, True]},
        {"rank_labels": [[2], [1]]},
    ],
)
def test_invalid_prepared_records_fail_before_tensor_construction(overrides):
    record = candidate_record(**overrides)
    with pytest.raises(ValueError):
        validate_candidate_record(record)
    with pytest.raises(ValueError):
        collator()([record])


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "id",
        "query",
        "source",
        "document",
        "document_ids",
        "document_keys",
        "known_document_ids",
        "relevance",
        "graded_relevance",
        "rank_labels",
    ],
)
def test_required_fields_cannot_silently_fall_back(field):
    record = candidate_record()
    del record[field]
    with pytest.raises(ValueError):
        validate_candidate_record(record)


@pytest.mark.parametrize(
    "record",
    [
        {"query": "q", "document": ["a", "b"], "ranking": [1, 2]},
        {"query": "q", "pos": ["a"], "neg": ["b"]},
    ],
)
def test_unprepared_formats_are_rejected(record):
    with pytest.raises(ValueError):
        collator()([record])


def test_precomputed_text_keys_filter_duplicates_across_sources():
    records = [
        candidate_record(["Alpha", "Beta"], source="first"),
        candidate_record(
            ["different rendering", "Gamma", "Delta"],
            source="second",
            document_keys=[
                candidate_record(["Alpha", "Beta"])["document_keys"][0],
                "g",
                "d",
            ],
        ),
    ]
    batch = collator()(records)
    assert batch["in_batch_positive_mask"].tolist() == [[False, False], [False, False]]
    assert batch["in_batch_candidate_mask"][0, 1].tolist() == [False, True, True]
    assert batch["in_batch_candidate_mask"][1, 0].tolist() == [False, True, False]


def test_known_document_ids_filter_without_relabeling_and_are_source_scoped():
    first = candidate_record(
        ["a", "b"], document_ids=["a", "b"], known_document_ids=["a", "b", "c"]
    )
    second = candidate_record(["c", "d"], document_ids=["c", "d"])
    batch = collator()([first, second])
    assert batch["in_batch_candidate_mask"][0, 1].tolist() == [False, True]
    assert batch["positive_mask"].tolist() == [[True, False], [True, False]]
    second["source"] = "another-task"
    assert collator()([first, second])["in_batch_candidate_mask"][0, 1].tolist() == [
        True,
        True,
    ]


def test_candidate_order_and_full_length_are_preserved():
    record = candidate_record([f"document {i}" for i in range(23)])
    layout = build_candidate_layout([record], relevance_scheme="graded")
    assert layout.positive_documents + layout.negative_documents == record["document"]
    assert layout.ordered_document_keys == [record["document_keys"]]
    assert layout.ordered_document_ids == [record["document_ids"]]
    assert layout.batch["candidate_mask"].sum() == 23


@pytest.mark.parametrize("objective", ["supervised", "grpo"])
def test_variable_slates_encode_only_real_candidates(objective):
    records = [
        candidate_record(["a", "b"], query="q1"),
        candidate_record(
            ["c", "d", "e", "f"],
            query="q2",
            relevance=[1, 1, 0, 0],
            rank_labels=[4, 3, 1, 2],
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
