"""Behavioral contracts for the fixed-corpus directory."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from candidate_helpers import candidate_record, document_key

from reler.data.protocol import (
    EMBEDDING_PROTOCOL_FILENAME,
    POOLING_COMPUTE_DTYPE,
    TOKENIZATION_VERSION,
)
from reler.fixed_corpus.build import build_corpus, encode_corpus_shards
from reler.fixed_corpus.build import main as build_main
from reler.fixed_corpus.index import DOCUMENT_MAPPING_FILENAME, FrozenCorpusIndex


def _protocol() -> dict[str, object]:
    return {
        "tokenization_version": TOKENIZATION_VERSION,
        "pooling_compute_dtype": POOLING_COMPUTE_DTYPE,
        "model_name_or_path": "base",
        "resolved_model_revision": "abc",
        "pooling_method": "last",
        "padding_side": "left",
        "append_token": "pad",
        "document_prompt_template": "{document}",
        "document_max_length": 8,
        "normalize": True,
        "storage_dtype": "float16",
    }


def _write_corpus(directory: Path, *, rows: int = 5, dimension: int = 3) -> Path:
    directory.mkdir()
    vectors = np.arange(rows * dimension, dtype=np.float32).reshape(rows, dimension) + 1
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    (directory / DOCUMENT_MAPPING_FILENAME).write_text(
        json.dumps({f"doc-{ordinal}": ordinal for ordinal in range(rows)})
    )
    (directory / EMBEDDING_PROTOCOL_FILENAME).write_text(json.dumps(_protocol()))
    for shard_id, (start, end) in enumerate(((0, 2), (2, rows))):
        if start < end:
            np.save(
                directory / f"vectors-{shard_id:05d}.npy",
                vectors[start:end].astype(np.float16),
            )
    return directory


def test_lookup_maps_keys_and_returns_detached_fp32(tmp_path):
    corpus_dir = _write_corpus(tmp_path / "corpus")
    with FrozenCorpusIndex(corpus_dir, device="cpu") as index:
        assert index.lookup_ordinals(["doc-4", "doc-1"]) == [4, 1]
        assert index.document_to_ordinal["doc-4"] == 4
        vectors = index.lookup_embeddings(torch.tensor([[4, 1], [0, 2]]))
        assert vectors.shape == (2, 2, 3)
        assert vectors.dtype == torch.float32
        assert not vectors.requires_grad
        assert torch.allclose(vectors.norm(dim=-1), torch.ones(2, 2), atol=2e-3)
        index.validate_protocol(
            model_name_or_path="base",
            resolved_model_revision="abc",
            pooling_method="last",
            padding_side="left",
            append_token="pad",
            document_prompt_template="{document}",
            document_max_length=8,
        )
        with pytest.raises(KeyError):
            index.lookup_ordinals(["unknown"])
        with pytest.raises(IndexError):
            index.lookup_embeddings(torch.tensor([5]))
    with pytest.raises(RuntimeError):
        index.lookup_embeddings(torch.tensor([0]))


def test_layout_and_protocol_mismatches_are_rejected(tmp_path):
    corpus_dir = _write_corpus(tmp_path / "corpus")
    protocol_path = corpus_dir / EMBEDDING_PROTOCOL_FILENAME
    protocol = json.loads(protocol_path.read_text())
    protocol["pooling_method"] = "mean"
    protocol_path.write_text(json.dumps(protocol))

    with FrozenCorpusIndex(corpus_dir) as index:
        with pytest.raises(ValueError):
            index.validate_protocol(
                model_name_or_path="base",
                resolved_model_revision="abc",
                pooling_method="last",
                padding_side="left",
                append_token="pad",
            )

    protocol.pop("document_prompt_template")
    protocol_path.write_text(json.dumps(protocol))
    with FrozenCorpusIndex(corpus_dir) as index:
        with pytest.raises(ValueError):
            index.validate_protocol(
                model_name_or_path="base",
                resolved_model_revision="abc",
                pooling_method="mean",
                padding_side="left",
                append_token="pad",
            )

    np.save(corpus_dir / "vectors-00000.npy", np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        FrozenCorpusIndex(corpus_dir)

    other_dir = _write_corpus(tmp_path / "missing-shard")
    (other_dir / "vectors-00001.npy").rename(other_dir / "vectors-00002.npy")
    with pytest.raises(ValueError):
        FrozenCorpusIndex(other_dir)

    zero_dim_dir = _write_corpus(tmp_path / "zero-dimension")
    np.save(zero_dim_dir / "vectors-00000.npy", np.empty((2, 0), dtype=np.float16))
    np.save(zero_dim_dir / "vectors-00001.npy", np.empty((3, 0), dtype=np.float16))
    with pytest.raises(ValueError):
        FrozenCorpusIndex(zero_dim_dir)


class _Tokenizer:
    pad_token = "[PAD]"
    pad_token_id = 0

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [[len(text.split()) or 1] for text in texts]}

    def pad(self, rows, **_kwargs):
        width = max(len(row["input_ids"]) for row in rows)
        ids = [row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in rows]
        masks = [
            row["attention_mask"] + [0] * (width - len(row["attention_mask"]))
            for row in rows
        ]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class _Model(torch.nn.Module):
    config = SimpleNamespace(hidden_size=3, _commit_hash="local")

    def forward(self, input_ids, **_kwargs):
        hidden = torch.stack(
            (input_ids.float(), input_ids.float() + 1, input_ids.float() + 2), dim=-1
        )
        return SimpleNamespace(last_hidden_state=hidden)


def test_build_corpus_deduplicates_and_encodes_multiple_shards(tmp_path):
    source = tmp_path / "documents.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"id": "a", "content": "Alpha"},
                {"id": "b", "content": "Beta Gamma"},
                {"id": "a", "content": " alpha "},
            )
        )
        + "\n"
    )
    corpus_dir = tmp_path / "corpus"
    assert build_corpus(source, corpus_dir, "document_jsonl") == 2
    revision, protocol = encode_corpus_shards(
        corpus_path=corpus_dir / "corpus.jsonl",
        offsets_path=corpus_dir / "corpus_offsets.npy",
        output_dir=corpus_dir,
        model_name_or_path="unused",
        revision=None,
        num_shards=2,
        batch_size=1,
        max_length=4,
        pooling_method="last",
        padding_side="left",
        append_token="pad",
        document_prompt_template="{document}",
        tokenizer=_Tokenizer(),
        model=_Model(),
    )
    assert revision == "local"
    assert protocol["pooling_compute_dtype"] == "float32"
    shards = sorted(corpus_dir.glob("vectors-*.npy"))
    assert [np.load(path).shape for path in shards] == [(1, 3), (1, 3)]
    assert all(np.load(path).dtype == np.float16 for path in shards)


def test_build_corpus_uses_prepared_keys_across_source_local_ids(tmp_path):
    source = tmp_path / "training.jsonl"
    source.write_text("\n".join(json.dumps(value) for value in (
        candidate_record(["alpha", "beta"], source="first"),
        candidate_record(["gamma", "delta"], source="second"),
        candidate_record(["alpha", "delta"], source="third"),
    )) + "\n")
    corpus_dir = tmp_path / "corpus"
    assert build_corpus(source, corpus_dir) == 4
    mapping = json.loads((corpus_dir / DOCUMENT_MAPPING_FILENAME).read_text())
    assert {document_key(text) for text in ["alpha", "beta", "gamma", "delta"]} == set(mapping)


def test_builder_publishes_only_the_runtime_directory(tmp_path, monkeypatch):
    source = tmp_path / "documents.jsonl"
    source.write_text('{"id":"a","content":"alpha"}\n{"id":"b","content":"beta"}\n')
    output = tmp_path / "corpus"
    monkeypatch.setattr(
        "reler.fixed_corpus.build.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    monkeypatch.setattr(
        "reler.fixed_corpus.build.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: _Model(),
    )

    build_main(
        [
            "--input",
            str(source),
            "--input-format",
            "document_jsonl",
            "--output-dir",
            str(output),
            "--model",
            "base",
            "--max-length",
            "4",
            "--num-shards",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert {path.name for path in output.iterdir()} == {
        DOCUMENT_MAPPING_FILENAME,
        EMBEDDING_PROTOCOL_FILENAME,
        "vectors-00000.npy",
        "vectors-00001.npy",
    }
    with FrozenCorpusIndex(output) as index:
        assert index.count == 2
        assert index.dimension == 3
