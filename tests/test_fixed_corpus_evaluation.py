import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mteb.encoder_interface import PromptType

from reler.evaluation.fixed_corpus import (
    FixedCorpusMTEBModel,
    PerSubsetCorpusCache,
    document_encoder_protocol,
)
from reler.evaluation.run_mteb import EvalArguments, run_bright


class _Embedder(torch.nn.Module):
    def __init__(self, dimension=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.base_model = SimpleNamespace(config=SimpleNamespace(hidden_size=dimension))
        self.tokenizer = SimpleNamespace(
            pad_token_id=0,
            eos_token_id=2,
            padding_side="left",
        )
        self.append_token = "pad"
        self.pooler_type = "last"
        self.truncate_dim = 0
        self.do_norm = True


class _Model:
    def __init__(self, name, *, revision="e0-revision", dimension=4, value=1.0):
        self.model = _Embedder(dimension)
        self.mteb_model_meta = SimpleNamespace(name=name, revision=revision)
        self.document_prompt_template = "{text}"
        self.max_doc_length = 32
        self.amp_dtype = None
        self.world_size = 1
        self.value = value
        self.calls = []

    def encode(self, sentences, *, prompt_type=None, **kwargs):
        self.calls.append((tuple(sentences), prompt_type, kwargs))
        return np.full((len(sentences), 4), self.value, dtype=np.float32)


def _protocol():
    return {
        "model_name_or_path": "base",
        "resolved_revision": "e0-revision",
        "protocol": 1,
    }


def test_per_subset_cache_reuses_ordered_shards_without_reencoding(tmp_path):
    cache = PerSubsetCorpusCache(
        tmp_path,
        task_name="BrightRetrieval",
        encoder_protocol=_protocol(),
        dataset_revision="bright-revision",
    )
    cache.begin("biology")
    expected = cache.encode(
        ["document one", "document two"],
        lambda: np.arange(8, dtype=np.float32).reshape(2, 4),
    )
    cache.finish()

    metadata = json.loads((tmp_path / "biology" / "cache.json").read_text())
    assert metadata["dataset_revision"] == "bright-revision"
    assert (tmp_path / "biology" / "COMPLETE").is_file()
    assert (tmp_path / "biology" / "shards" / "00000.npy").is_file()

    cached = PerSubsetCorpusCache(
        tmp_path,
        task_name="BrightRetrieval",
        encoder_protocol=_protocol(),
        dataset_revision="bright-revision",
    )
    cached.begin("biology")
    actual = cached.encode(
        ["document one", "document two"],
        lambda: pytest.fail("a completed fixed corpus must not re-encode passages"),
    )
    cached.finish()

    np.testing.assert_array_equal(actual, expected)


def test_per_subset_cache_rejects_changed_dataset_revision(tmp_path):
    cache = PerSubsetCorpusCache(
        tmp_path,
        task_name="BrightRetrieval",
        encoder_protocol=_protocol(),
        dataset_revision="bright-revision-a",
    )
    cache.begin("biology")
    cache.encode(["document"], lambda: np.ones((1, 4), dtype=np.float32))
    cache.finish()

    changed = PerSubsetCorpusCache(
        tmp_path,
        task_name="BrightRetrieval",
        encoder_protocol=_protocol(),
        dataset_revision="bright-revision-b",
    )
    with pytest.raises(ValueError):
        changed.begin("biology")


def test_per_subset_cache_rejects_a_different_e0_revision(tmp_path):
    first_e0 = _Model("base", revision="e0-revision-a")
    first = PerSubsetCorpusCache(
        tmp_path,
        task_name="BrightRetrieval",
        encoder_protocol=document_encoder_protocol("base", first_e0),
        dataset_revision="bright-revision",
    )
    first.begin("biology")
    first.encode(["document"], lambda: np.ones((1, 4), dtype=np.float32))
    first.finish()

    changed_e0 = _Model("base", revision="e0-revision-b")
    changed = PerSubsetCorpusCache(
        tmp_path,
        task_name="BrightRetrieval",
        encoder_protocol=document_encoder_protocol("base", changed_e0),
        dataset_revision="bright-revision",
    )
    with pytest.raises(ValueError):
        changed.begin("biology")


def test_completed_cache_rejects_changed_shard_layout(tmp_path):
    cache = PerSubsetCorpusCache(
        tmp_path,
        task_name="BrightRetrieval",
        encoder_protocol=_protocol(),
        dataset_revision="bright-revision",
    )
    cache.begin("biology")
    cache.encode(["one", "two"], lambda: np.ones((2, 4), dtype=np.float32))
    cache.finish()

    cached = PerSubsetCorpusCache(
        tmp_path,
        task_name="BrightRetrieval",
        encoder_protocol=_protocol(),
        dataset_revision="bright-revision",
    )
    cached.begin("biology")
    with pytest.raises(ValueError):
        cached.encode(
            ["one"],
            lambda: pytest.fail("a completed fixed corpus must not re-encode passages"),
        )


def test_fixed_model_routes_passages_to_e0_and_queries_to_checkpoint(tmp_path):
    query_model = _Model("checkpoint", value=3.0)
    corpus_model = _Model("base", value=7.0)
    model = FixedCorpusMTEBModel(
        query_model,
        corpus_model,
        cache_dir=tmp_path,
        corpus_model_name_or_path="base",
        task_name="BrightRetrieval",
        dataset_revision="bright-revision",
    )

    model.begin_corpus_subset("biology")
    passages = model.encode(
        ["passage"], task_name="BrightRetrieval", prompt_type=PromptType.passage
    )
    queries = model.encode(
        ["query"], task_name="BrightRetrieval", prompt_type=PromptType.query
    )
    model.finish_corpus_subset()

    np.testing.assert_array_equal(passages, np.full((1, 4), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(queries, np.full((1, 4), 3.0, dtype=np.float32))
    assert corpus_model.calls[0][1] == PromptType.passage
    assert query_model.calls[0][1] == PromptType.query


def test_fixed_model_rejects_incompatible_embedding_dimensions(tmp_path):
    with pytest.raises(ValueError):
        FixedCorpusMTEBModel(
            _Model("checkpoint", dimension=4),
            _Model("base", dimension=8),
            cache_dir=tmp_path,
            corpus_model_name_or_path="base",
            task_name="BrightRetrieval",
            dataset_revision="bright-revision",
        )


def test_bright_runner_opens_and_completes_one_fixed_cache_per_subset(
    tmp_path, monkeypatch
):
    query_model = _Model("checkpoint", value=3.0)
    corpus_model = _Model("base", value=7.0)
    model = FixedCorpusMTEBModel(
        query_model,
        corpus_model,
        cache_dir=tmp_path / "cache",
        corpus_model_name_or_path="base",
        task_name="BrightRetrieval",
        dataset_revision="bright-revision",
    )
    task = SimpleNamespace(
        metadata=SimpleNamespace(eval_langs=["biology"]),
    )

    class _Evaluation:
        def run(self, wrapper, **kwargs):
            wrapper.encode(
                ["passage"],
                task_name="BrightRetrieval",
                prompt_type=PromptType.passage,
            )
            wrapper.encode(
                ["query"],
                task_name="BrightRetrieval",
                prompt_type=PromptType.query,
            )
            return [SimpleNamespace(scores={"standard": [{"hf_subset": "biology"}]})]

    monkeypatch.setattr(
        "reler.evaluation.run_mteb.load_official_bright", lambda *_: None
    )
    monkeypatch.setattr(
        "reler.evaluation.run_mteb.mteb.MTEB", lambda **_: _Evaluation()
    )

    results = run_bright(
        task,
        model,
        EvalArguments(output_dir=str(tmp_path / "results")),
        eval_subsets=["biology"],
    )

    assert len(results) == 1
    assert (tmp_path / "cache" / "biology" / "cache.json").is_file()
