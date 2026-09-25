import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import mteb
import torch
from transformers import HfArgumentParser

from reler.config.loader import load_raw_config_file
from reler.data.protocol import load_embedding_protocol, protocol_to_eval_kwargs
from reler.evaluation import get_tasks

from .fixed_corpus import FixedCorpusMTEBModel
from .qwen3_embedding_model import Qwen3Embedding

logging.basicConfig(
    format="%(levelname)s|%(asctime)s|%(name)s#%(lineno)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("run_mteb.py")


BRIGHT_TASK_NAME = "BrightRetrieval"
BRIGHT_SPLIT = "standard"
BRIGHT_DATASET_PATH = "xlangai/BRIGHT"
BRIGHT_DATASET_CONFIG = "documents"
BRIGHT_DATASET_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
BRIGHT_INSTRUCTIONS = {
    "biology": "Given a biology post, retrieve relevant passages that help answer the post.",
    "earth_science": "Given an earth science post, retrieve relevant passages that help answer the post.",
    "economics": "Given an economics post, retrieve relevant passages that help answer the post.",
    "psychology": "Given a psychology post, retrieve relevant passages that help answer the post.",
    "robotics": "Given a robotics post, retrieve relevant passages that help answer the post.",
    "stackoverflow": "Given a Stack Overflow post, retrieve relevant passages that help answer the post.",
    "sustainable_living": "Given a sustainable living post, retrieve relevant passages that help answer the post.",
    "pony": "Given a question about the Pony programming language, retrieve relevant passages that help answer the question.",
    "leetcode": "Given a coding problem, retrieve relevant examples that help answer the problem.",
    "aops": "Given a math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_theorems": "Given a math problem, retrieve relevant theorems that help answer the problem.",
    "theoremqa_questions": "Given a math problem, retrieve relevant examples that help answer the problem.",
}


@dataclass
class EvalArguments:
    """
    Arguments.
    """

    model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_name: Optional[str] = field(
        default=None, metadata={"help": "Model name for the save path"}
    )
    model_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional repository model YAML carrying the embedding protocol"
        },
    )
    model_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model kwargs, json string."},
    )
    encode_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific encode kwargs, json string."},
    )
    run_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific kwargs for `MTEB.run()`, json string."},
    )

    output_dir: Optional[str] = field(
        default="results", metadata={"help": "output dir of results"}
    )
    benchmark: Optional[str] = field(default=None, metadata={"help": "Benchmark name"})
    tasks: Optional[str] = field(default=None, metadata={"help": "',' seprated"})
    langs: Optional[str] = field(default=None, metadata={"help": "',' seprated"})
    only_load: bool = field(default=False, metadata={"help": ""})
    load_model: bool = field(default=False, metadata={"help": "when only_load"})

    batch_size: int = field(
        default=128, metadata={"help": "Will be set to `encode_kwargs`"}
    )
    precision: str = field(
        default="fp16", metadata={"help": "amp_fp16,amp_bf16,fp16,bf16,fp32"}
    )
    fail_on_task_error: bool = field(
        default=False,
        metadata={
            "help": "Exit non-zero if any selected task fails or returns no result"
        },
    )
    fixed_corpus_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Immutable E0 document encoder for fixed-corpus BRIGHT evaluation"
        },
    )
    fixed_corpus_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "Immutable E0 revision recorded in the fixed-corpus cache"},
    )
    fixed_corpus_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Root directory for cached E0 embeddings, isolated per BRIGHT subset"
        },
    )
    fixed_corpus_model_kwargs: Optional[str] = field(
        default=None,
        metadata={
            "help": "JSON Qwen3Embedding kwargs for the fixed E0 document encoder"
        },
    )
    bright_dataset_revision: str = field(
        default=BRIGHT_DATASET_REVISION,
        metadata={
            "help": "Immutable xlangai/BRIGHT revision used for documents/examples"
        },
    )
    bright_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional Hugging Face cache directory for official BRIGHT data"
        },
    )

    def __post_init__(self):
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(",")
        if isinstance(self.langs, str):
            self.langs = self.langs.split(",")
        for name in ("model", "encode", "run", "fixed_corpus_model"):
            name = name + "_kwargs"
            attr = getattr(self, name)
            if attr is None:
                setattr(self, name, dict())
            elif isinstance(attr, str):
                setattr(self, name, json.loads(attr))


def get_model(model_path: str, precision: str = "fp16", **kwargs):
    # Trained checkpoints carry this sidecar, so post-hoc evaluation cannot silently
    # fall back to the Qwen pooling/prompt protocol. Explicit CLI kwargs still win.
    protocol_kwargs = protocol_to_eval_kwargs(load_embedding_protocol(model_path))
    protocol_kwargs.update(kwargs)
    model = Qwen3Embedding(model_path, precision=precision, **protocol_kwargs)
    return model


def _model_config_eval_kwargs(path: str) -> dict[str, object]:
    raw_model_config = load_raw_config_file(path)
    protocol = {
        key: raw_model_config[key]
        for key in (
            "pooling_method",
            "padding_side",
            "append_token",
            "query_prompt_template",
            "document_prompt_template",
        )
        if key in raw_model_config
    }
    missing = {
        "pooling_method",
        "padding_side",
        "append_token",
        "query_prompt_template",
        "document_prompt_template",
    } - protocol.keys()
    if missing:
        raise ValueError(
            f"model_config {path!r} is missing embedding protocol fields: "
            f"{sorted(missing)}"
        )
    protocol["normalize"] = True
    protocol["max_length"] = raw_model_config.get("embedding_max_length", 8192)
    return protocol_to_eval_kwargs(protocol)


class _InstructionOverrideModel:
    """Add one query instruction while leaving MTEB's retrieval path unchanged."""

    def __init__(self, model, instruction: str):
        self.model = model
        self.instruction = instruction
        self.mteb_model_meta = model.mteb_model_meta

    def encode(self, sentences, **kwargs):
        return self.model.encode(
            sentences,
            task_instruction=self.instruction,
            **kwargs,
        )


def _bright_result_subsets(result) -> set[str]:
    scores = getattr(result, "scores", {})
    return {
        score["hf_subset"]
        for score in scores.get(BRIGHT_SPLIT, [])
        if score.get("hf_subset")
    }


def load_official_bright(task, subsets: list[str], args: EvalArguments) -> None:
    """Bypass MTEB's older pin and explicitly load BRIGHT's official documents config."""
    identity = {
        "path": BRIGHT_DATASET_PATH,
        "config": BRIGHT_DATASET_CONFIG,
        "revision": args.bright_dataset_revision,
        "subsets": sorted(subsets),
    }
    if getattr(task, "_reler_bright_identity", None) == identity:
        return
    corpus, queries, relevant_docs = task.load_bright_data(
        path=BRIGHT_DATASET_PATH,
        domains=subsets,
        eval_splits=[BRIGHT_SPLIT],
        cache_dir=args.bright_cache_dir,
        revision=args.bright_dataset_revision,
    )
    task.corpus = corpus
    task.queries = queries
    task.relevant_docs = relevant_docs
    task.data_loaded = True
    task._reler_bright_identity = identity


def run_bright(task, model, args, **kwargs):
    """Evaluate BRIGHT through MTEB, applying its instruction per domain.

    MTEB 1.38 evaluates all BRIGHT domains under one task name and therefore does
    not expose the current domain to ``Encoder.encode``. Run one MTEB subset at a
    time so the query instruction is correct, while retaining MTEB's data loader,
    exact retrieval, metrics, result merging, and on-disk schema.
    """
    if args.output_dir is None:
        raise ValueError(
            "BRIGHT evaluation requires output_dir to merge domain results"
        )

    run_kwargs = dict(kwargs)
    official_subsets = list(task.metadata.eval_langs)
    missing_instructions = sorted(set(official_subsets) - BRIGHT_INSTRUCTIONS.keys())
    if missing_instructions:
        raise RuntimeError(
            f"BRIGHT task contains subsets without query instructions: {missing_instructions}"
        )
    requested_subsets = run_kwargs.pop("eval_subsets", None)
    if requested_subsets is None:
        requested_subsets = official_subsets
    elif isinstance(requested_subsets, str):
        requested_subsets = [requested_subsets]
    else:
        requested_subsets = list(dict.fromkeys(requested_subsets))

    unknown_subsets = sorted(set(requested_subsets) - set(official_subsets))
    if unknown_subsets:
        raise ValueError(
            f"Unknown BRIGHT subsets: {unknown_subsets}; "
            f"expected one or more of {official_subsets}"
        )
    if not requested_subsets:
        raise ValueError("No BRIGHT subset selected")

    load_official_bright(task, requested_subsets, args)

    requested_splits = run_kwargs.pop("eval_splits", None) or [BRIGHT_SPLIT]
    if isinstance(requested_splits, str):
        requested_splits = [requested_splits]
    if list(requested_splits) != [BRIGHT_SPLIT]:
        raise ValueError(
            f"BrightRetrieval uses the official {BRIGHT_SPLIT!r} split; "
            f"got {list(requested_splits)!r}"
        )

    final_result = None
    for subset in requested_subsets:
        logger.info("Evaluating BRIGHT subset %s", subset)
        manages_fixed_corpus = hasattr(model, "begin_corpus_subset")
        if manages_fixed_corpus:
            model.begin_corpus_subset(subset)
        evaluation = mteb.MTEB(tasks=[task])
        try:
            results = evaluation.run(
                _InstructionOverrideModel(model, BRIGHT_INSTRUCTIONS[subset]),
                output_folder=args.output_dir,
                encode_kwargs=args.encode_kwargs or {},
                eval_splits=[BRIGHT_SPLIT],
                eval_subsets=[subset],
                **run_kwargs,
            )
            if manages_fixed_corpus:
                model.finish_corpus_subset()
        except Exception:
            if manages_fixed_corpus:
                model.cancel_corpus_subset()
            raise
        if not results:
            raise RuntimeError(f"MTEB returned no result for BRIGHT subset {subset!r}")
        final_result = results[0]

    completed_subsets = _bright_result_subsets(final_result)
    missing_subsets = sorted(set(requested_subsets) - completed_subsets)
    if missing_subsets:
        raise RuntimeError(
            f"BRIGHT result is missing evaluated subsets: {missing_subsets}"
        )
    return [final_result]


def run_eval(
    model,
    tasks: list,
    args: EvalArguments,
    *,
    fail_on_task_error: bool = False,
    **kwargs,
):
    if not tasks:
        raise RuntimeError("No task selected")

    encode_kwargs = args.encode_kwargs or dict()
    all_results = []
    failed_tasks = []

    _num_gpus, _started = torch.cuda.device_count(), False
    if _num_gpus > 1 and not _started and hasattr(model, "start"):
        model.start()
        _started = True

    try:
        for t in tasks:
            if t.metadata.name == BRIGHT_TASK_NAME:
                all_results.extend(run_bright(t, model, args, **kwargs))
                continue
            evaluation = mteb.MTEB(tasks=[t])

            try:
                results = evaluation.run(
                    model,
                    output_folder=args.output_dir,
                    encode_kwargs=encode_kwargs,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(
                    f"meet error when running task: {t.metadata.name}. {str(e)}"
                )
                failed_tasks.append((t.metadata.name, str(e)))
                continue
            if not results:
                failed_tasks.append((t.metadata.name, "MTEB returned no result"))
            all_results.extend(results or [])
    finally:
        if model is not None and _started and hasattr(model, "stop"):
            model.stop()
    if fail_on_task_error and failed_tasks:
        details = "; ".join(f"{name}: {error}" for name, error in failed_tasks)
        raise RuntimeError(
            f"Incomplete MTEB evaluation ({len(failed_tasks)} failed tasks): {details}"
        )
    return all_results


def _validate_fixed_corpus_args(args: EvalArguments, tasks: list) -> None:
    """Keep fixed-E0 evaluation explicit and leave normal MTEB unchanged."""
    fixed_options = (
        args.fixed_corpus_cache_dir,
        args.fixed_corpus_model_revision,
        args.fixed_corpus_model_kwargs,
    )
    if not args.fixed_corpus_model:
        if any(fixed_options):
            raise ValueError(
                "--fixed_corpus_model is required when fixed-corpus options are supplied"
            )
        return
    if not args.fixed_corpus_cache_dir:
        raise ValueError(
            "--fixed_corpus_cache_dir is required with --fixed_corpus_model"
        )
    non_bright = [
        task.metadata.name for task in tasks if task.metadata.name != BRIGHT_TASK_NAME
    ]
    if non_bright:
        raise ValueError(
            "Fixed-corpus evaluation is currently defined only for BrightRetrieval; "
            f"got additional tasks: {non_bright}"
        )


def main():
    parser = HfArgumentParser(EvalArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        with open(os.path.abspath(sys.argv[1])) as f:
            config = json.load(f)
        logger.warning(f"Json config {f.name} : \n{json.dumps(config, indent=2)}")
        args, *_ = parser.parse_dict(config)
        del config, f
    else:
        args, *_ = parser.parse_args_into_dataclasses()
        logger.warning(f"Args {args}")
    del parser

    config_kwargs = None
    if args.model_config:
        config_kwargs = _model_config_eval_kwargs(args.model_config)
        config_kwargs.update(args.model_kwargs)
        args.model_kwargs = config_kwargs

    tasks = get_tasks(args.tasks, args.langs, args.benchmark)
    _validate_fixed_corpus_args(args, tasks)
    # print('args.model_kwargs', args.model_kwargs)
    logger.warning(f"Selected {len(tasks)} tasks:\n" + "\n".join(str(t) for t in tasks))
    if args.only_load:
        for t in tasks:
            logger.warning(f"Loading {t}")
            try:
                if t.metadata.name == BRIGHT_TASK_NAME:
                    load_official_bright(t, list(t.metadata.eval_langs), args)
                else:
                    t.load_data()
            except Exception as e:
                logger.warning(
                    f"meet error when loading task: {t.metadata.name}. {str(e)}"
                )
                # t.load_data(force_download=True)
            else:
                continue

        if not args.load_model:
            return
    model = get_model(args.model, precision=args.precision, **args.model_kwargs)
    if args.fixed_corpus_model:
        # The query checkpoint's embedding protocol is also the document-space
        # contract unless the caller explicitly overrides a field for E0.
        corpus_kwargs = protocol_to_eval_kwargs(load_embedding_protocol(args.model))
        corpus_kwargs.update(
            {
                key: value
                for key, value in args.model_kwargs.items()
                if key not in {"model_name", "revision"}
            }
        )
        corpus_kwargs.update(args.fixed_corpus_model_kwargs)
        if args.fixed_corpus_model_revision:
            corpus_kwargs["revision"] = args.fixed_corpus_model_revision
        corpus_model = get_model(
            args.fixed_corpus_model,
            precision=args.precision,
            **corpus_kwargs,
        )
        model = FixedCorpusMTEBModel(
            model,
            corpus_model,
            cache_dir=args.fixed_corpus_cache_dir,
            corpus_model_name_or_path=args.fixed_corpus_model,
            task_name=BRIGHT_TASK_NAME,
            dataset_revision=args.bright_dataset_revision,
        )
    if args.only_load:
        return

    args.encode_kwargs.update(batch_size=args.batch_size)
    run_eval(
        model,
        tasks,
        args,
        fail_on_task_error=args.fail_on_task_error,
        **args.run_kwargs,
    )
    logger.warning(f"Done {len(tasks)} tasks.")
    return


if __name__ == "__main__":
    main()
