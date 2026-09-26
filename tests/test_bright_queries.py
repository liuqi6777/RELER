"""Official generated queries, ID alignment, and real MTEB result isolation."""

from collections import Counter
from types import SimpleNamespace

import mteb
import numpy as np
import pytest
from transformers import HfArgumentParser

from reler.evaluation import run_mteb


@pytest.fixture
def bright_data(monkeypatch):
    originals = {"q1": "original", "q2": "generated"}
    # Reverse the file order and include gold annotations that must never be used.
    generated = [
        {"id": "q2", "query": "original", "reasoning": "gold gold"},
        {"id": "q1", "query": "\n generated  ", "reasoning": "gold gold"},
    ]
    corpus = {"d1": {"text": "generated"}, "d2": {"text": "original gold"}}
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    calls = {"base": [], "generated": []}
    task = mteb.get_tasks(tasks=["BrightRetrieval"])[0]

    def load_base(**kwargs):
        calls["base"].append(kwargs)
        subsets = kwargs["domains"]
        return (
            {s: {"standard": corpus} for s in subsets},
            {s: {"standard": dict(originals)} for s in subsets},
            {s: {"standard": qrels} for s in subsets},
        )

    def load_generated(path, config, **kwargs):
        calls["generated"].append((path, config, kwargs))
        return generated

    monkeypatch.setattr(task, "load_bright_data", load_base)
    monkeypatch.setattr(run_mteb, "load_dataset", load_generated)
    return SimpleNamespace(
        task=task, generated=generated, corpus=corpus, qrels=qrels, calls=calls
    )


class WordCountEncoder:
    def __init__(self):
        self.calls = []
        self.mteb_model_meta = mteb.ModelMeta(
            name="local/toy",
            revision="v1",
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=3,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            framework=["NumPy"],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )

    def encode(self, sentences, **kwargs):
        sentences = list(sentences)
        self.calls.append((sentences, kwargs))
        return np.array(
            [
                [
                    Counter(text.split())[word]
                    for word in ("original", "generated", "gold")
                ]
                for text in sentences
            ],
            dtype=np.float32,
        )


def test_generated_queries_change_retrieval_without_reusing_original_results(
    bright_data, tmp_path
):
    model = WordCountEncoder()
    original_args = run_mteb.EvalArguments(output_dir=str(tmp_path))
    original = run_mteb.run_bright(
        bright_data.task, model, original_args, eval_subsets=["biology"], verbosity=0
    )[0]
    assert original.scores["standard"][0]["ndcg_at_10"] == pytest.approx(0.63093)
    calls_before = len(model.calls)
    generated_args = run_mteb.EvalArguments(
        output_dir=str(tmp_path), bright_query_set="gpt4-reasoning"
    )
    generated = run_mteb.run_bright(
        bright_data.task,
        model,
        generated_args,
        eval_subsets=["biology", "pony"],
        verbosity=0,
        save_predictions=True,
    )[0]
    assert len(model.calls) > calls_before
    assert {s["hf_subset"] for s in generated.scores["standard"]} == {"biology", "pony"}
    assert all(s["ndcg_at_10"] == 1 for s in generated.scores["standard"])
    assert bright_data.task.corpus["biology"]["standard"] is bright_data.corpus
    assert bright_data.task.relevant_docs["biology"]["standard"] is bright_data.qrels
    assert list(bright_data.task.queries["biology"]["standard"]) == ["q1", "q2"]
    result_paths = list(tmp_path.rglob("BrightRetrieval.json"))
    assert len(result_paths) == 2
    assert (
        len(list((tmp_path / "query-gpt4-reasoning").rglob("BrightRetrieval.json")))
        == 1
    )
    assert (
        len(list((tmp_path / "query-gpt4-reasoning").glob("*_predictions.json"))) == 2
    )

    # Each mode may reuse its own completed result, without another encoder call.
    model.calls.clear()
    for args, subsets, expected in (
        (original_args, ["biology"], 0.63093),
        (generated_args, ["biology", "pony"], 1.0),
    ):
        cached = run_mteb.run_bright(
            bright_data.task, model, args, eval_subsets=subsets, verbosity=0
        )[0]
        assert cached.scores["standard"][0]["ndcg_at_10"] == pytest.approx(expected)
    assert not model.calls


def test_query_loading_is_cached_by_mode_subset_and_revision(bright_data):
    original_args = run_mteb.EvalArguments(bright_cache_dir="local-cache")
    run_mteb.load_official_bright(bright_data.task, ["biology"], original_args)
    run_mteb.load_official_bright(bright_data.task, ["biology"], original_args)
    assert len(bright_data.calls["base"]) == 1
    assert not bright_data.calls["generated"]
    generated_args = run_mteb.EvalArguments(
        bright_query_set="gpt4-reasoning",
        bright_dataset_revision="test-revision",
        bright_cache_dir="local-cache",
    )
    run_mteb.load_official_bright(bright_data.task, ["biology"], generated_args)
    run_mteb.load_official_bright(bright_data.task, ["biology"], generated_args)
    assert len(bright_data.calls["base"]) == 2
    assert bright_data.calls["generated"] == [
        (
            run_mteb.BRIGHT_DATASET_PATH,
            "gpt4_reason",
            {
                "split": "biology",
                "cache_dir": "local-cache",
                "revision": "test-revision",
            },
        )
    ]
    generated_args.bright_dataset_revision = "another-revision"
    run_mteb.load_official_bright(bright_data.task, ["biology", "pony"], generated_args)
    assert len(bright_data.calls["base"]) == 3
    assert len(bright_data.calls["generated"]) == 3


@pytest.mark.parametrize("text", [None, "", "  ", "N/A", "empty", 42])
def test_invalid_generated_query_fails_without_using_original(bright_data, text):
    bright_data.generated[0]["query"] = text
    with pytest.raises(ValueError):
        run_mteb.load_official_bright(
            bright_data.task,
            ["biology"],
            run_mteb.EvalArguments(bright_query_set="gpt4-reasoning"),
        )
    assert not bright_data.task.data_loaded


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "invalid_id"])
def test_generated_query_ids_must_match_exactly(bright_data, case):
    if case == "missing":
        bright_data.generated.pop()
    elif case == "extra":
        bright_data.generated.append({"id": "q3", "query": "unexpected"})
    elif case == "duplicate":
        bright_data.generated.append(dict(bright_data.generated[0]))
    else:
        bright_data.generated[0]["id"] = None
    with pytest.raises(ValueError):
        run_mteb.load_official_bright(
            bright_data.task,
            ["biology"],
            run_mteb.EvalArguments(bright_query_set="gpt4-reasoning"),
        )
    assert not bright_data.task.data_loaded


def test_cli_load_only_validates_generated_queries(bright_data, monkeypatch):
    monkeypatch.setattr(run_mteb, "get_tasks", lambda *_: [bright_data.task])
    monkeypatch.setattr(
        run_mteb.sys,
        "argv",
        [
            "reler-eval",
            "--tasks",
            "BrightRetrieval",
            "--bright_query_set",
            "gpt4-reasoning",
            "--only_load",
        ],
    )
    monkeypatch.setattr(
        run_mteb,
        "get_model",
        lambda *_a, **_kw: pytest.fail("preload must not load a model"),
    )
    run_mteb.main()
    assert bright_data.task.data_loaded
    bright_data.generated[0]["query"] = None
    del bright_data.task._reler_bright_identity
    with pytest.raises(ValueError):
        run_mteb.main()


def test_unknown_query_set_is_rejected_by_cli_and_dataclass():
    with pytest.raises(SystemExit):
        HfArgumentParser(run_mteb.EvalArguments).parse_args_into_dataclasses(
            ["--bright_query_set", "unknown"]
        )
    with pytest.raises(ValueError):
        run_mteb.EvalArguments(bright_query_set="unknown")
