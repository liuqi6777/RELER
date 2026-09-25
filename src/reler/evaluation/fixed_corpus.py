"""Fixed-E0 document embeddings for asymmetric BRIGHT evaluation."""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mteb.encoder_interface import PromptType

from reler.data.protocol import POOLING_COMPUTE_DTYPE, tokenization_metadata

CACHE_FORMAT_VERSION = 1
_SAFE_SUBSET = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHARD_NAME = re.compile(r"^(\d{5})\.npy$")


def document_encoder_protocol(model_name_or_path: str, model: Any) -> dict[str, object]:
    """Describe the E0 document embedding space used by a cache."""
    embedder = getattr(model, "model", None)
    backbone = getattr(embedder, "base_model", None)
    config = getattr(backbone, "config", None)
    revision = getattr(config, "_commit_hash", None) or getattr(
        getattr(model, "mteb_model_meta", None), "revision", None
    )
    if not revision:
        raise ValueError(
            "Fixed-corpus evaluation requires an immutable E0 revision. "
            "Pass --fixed_corpus_model_revision."
        )
    try:
        parameter_dtype = str(next(embedder.parameters()).dtype)
    except (AttributeError, StopIteration):
        parameter_dtype = None
    return {
        **tokenization_metadata(embedder.tokenizer, embedder.append_token),
        "pooling_compute_dtype": POOLING_COMPUTE_DTYPE,
        "model_name_or_path": model_name_or_path,
        "resolved_revision": revision,
        "pooling_method": getattr(embedder, "pooler_type", None),
        "truncate_dim": getattr(embedder, "truncate_dim", None),
        "normalize": getattr(embedder, "do_norm", None),
        "parameter_dtype": parameter_dtype,
        "autocast_dtype": str(getattr(model, "amp_dtype", None)),
        "padding_side": getattr(embedder.tokenizer, "padding_side", None),
        "append_token": getattr(embedder, "append_token", None),
        "document_prompt_template": getattr(model, "document_prompt_template", None),
        "document_max_length": getattr(model, "max_doc_length", None),
    }


class PerSubsetCorpusCache:
    """Sequential E0 embedding shards for exactly one active BRIGHT subset."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        task_name: str,
        encoder_protocol: dict[str, object],
        dataset_revision: str,
    ) -> None:
        if not dataset_revision:
            raise ValueError("Fixed-corpus evaluation requires a dataset revision")
        self.root = Path(root).resolve()
        self.metadata = {
            "format_version": CACHE_FORMAT_VERSION,
            "task_name": task_name,
            "encoder": copy.deepcopy(encoder_protocol),
            "dataset_revision": dataset_revision,
        }
        self.subset: str | None = None
        self.subset_dir: Path | None = None
        self._complete = False
        self._shards: list[Path] = []
        self._observed = 0
        self._dimension: int | None = None
        self._dtype: str | None = None

    def begin(self, subset: str) -> None:
        if self.subset is not None:
            raise RuntimeError(f"Corpus subset {self.subset!r} is still active")
        if not _SAFE_SUBSET.fullmatch(subset):
            raise ValueError(f"Unsafe corpus subset name: {subset!r}")

        subset_dir = self.root / subset
        shard_dir = subset_dir / "shards"
        metadata_path = subset_dir / "cache.json"
        complete_path = subset_dir / "COMPLETE"
        subset_dir.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(exist_ok=True)

        expected = {**self.metadata, "subset": subset}
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as handle:
                saved = json.load(handle)
            if saved != expected:
                raise ValueError(
                    f"Fixed-corpus cache protocol changed: {metadata_path}. "
                    "Use a new --fixed_corpus_cache_dir."
                )
        else:
            if complete_path.exists() or any(shard_dir.iterdir()):
                raise ValueError(
                    f"Fixed-corpus cache metadata is missing: {metadata_path}. "
                    "Use a new --fixed_corpus_cache_dir."
                )
            self._write_json(metadata_path, expected)

        self.subset = subset
        self.subset_dir = subset_dir
        self._complete = complete_path.is_file()
        self._shards = self._list_shards(shard_dir)
        if self._complete and not self._shards:
            self.cancel()
            raise ValueError(
                f"Completed fixed-corpus cache has no shards: {subset_dir}"
            )
        if not self._complete:
            for path in shard_dir.iterdir():
                if path.is_file() and (
                    path.suffix == ".npy" or path.name.endswith(".npy.partial")
                ):
                    path.unlink()
            self._shards = []
        self._observed = 0
        self._dimension = None
        self._dtype = None

    def encode(
        self,
        texts: Sequence[str],
        encode_fn: Callable[[], np.ndarray | torch.Tensor],
    ) -> np.ndarray:
        if self.subset_dir is None:
            raise RuntimeError("Select a corpus subset before encoding passages")
        if not texts:
            raise ValueError("A fixed-corpus shard cannot be empty")

        position = self._observed
        if self._complete:
            if position >= len(self._shards):
                raise ValueError(
                    f"Corpus subset {self.subset!r} has more chunks than its cache; "
                    "use a new --fixed_corpus_cache_dir"
                )
            values = self._load_shard(self._shards[position])
        else:
            values = self._as_numpy(encode_fn())
            self._write_shard(position, values)
        if values.shape[0] != len(texts):
            raise ValueError(
                f"Corpus subset {self.subset!r} changed chunk layout; "
                "use a new --fixed_corpus_cache_dir"
            )
        self._accept_layout(values)
        self._observed += 1
        return values

    def finish(self) -> None:
        if self.subset_dir is None or self.subset is None:
            raise RuntimeError("No corpus subset is active")
        if self._observed == 0:
            if self._complete:
                self.cancel()
                return
            raise RuntimeError(f"Corpus subset {self.subset!r} did not encode passages")
        if self._complete:
            if self._observed != len(self._shards):
                raise ValueError(
                    f"Corpus subset {self.subset!r} changed chunk layout; "
                    "use a new --fixed_corpus_cache_dir"
                )
        else:
            temporary = self.subset_dir / "COMPLETE.partial"
            temporary.write_text("", encoding="utf-8")
            os.replace(temporary, self.subset_dir / "COMPLETE")
        self.cancel()

    def cancel(self) -> None:
        self.subset = None
        self.subset_dir = None
        self._complete = False
        self._shards = []
        self._observed = 0
        self._dimension = None
        self._dtype = None

    def _write_shard(self, position: int, values: np.ndarray) -> None:
        assert self.subset_dir is not None
        shard_path = self.subset_dir / "shards" / f"{position:05d}.npy"
        temporary = shard_path.with_suffix(".npy.partial")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        os.replace(temporary, shard_path)

    def _accept_layout(self, values: np.ndarray) -> None:
        dimension, dtype = int(values.shape[1]), str(values.dtype)
        if dimension <= 0:
            raise ValueError("Corpus embeddings must have a positive dimension")
        if self._dimension is None:
            self._dimension, self._dtype = dimension, dtype
        elif (dimension, dtype) != (self._dimension, self._dtype):
            raise ValueError(
                f"Inconsistent embedding shards for subset {self.subset!r}"
            )

    @staticmethod
    def _list_shards(directory: Path) -> list[Path]:
        paths = sorted(directory.glob("*.npy"))
        ids = [
            int(match.group(1)) if (match := _SHARD_NAME.fullmatch(path.name)) else None
            for path in paths
        ]
        if ids != list(range(len(paths))):
            raise ValueError(
                f"Fixed-corpus shards must be consecutively numbered in {directory}"
            )
        return paths

    @classmethod
    def _load_shard(cls, path: Path) -> np.ndarray:
        try:
            return cls._as_numpy(np.load(path, allow_pickle=False))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot load fixed-corpus shard: {path}") from exc

    @staticmethod
    def _as_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            values = values.detach().float().cpu().numpy()
        values = np.asarray(values)
        if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(
                f"Corpus embeddings must be a floating matrix, got "
                f"shape={values.shape} dtype={values.dtype}"
            )
        return values

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)


def _embedding_dimension(model: Any) -> int | None:
    embedder = getattr(model, "model", None)
    truncate_dim = getattr(embedder, "truncate_dim", 0) or 0
    if truncate_dim > 0:
        return int(truncate_dim)
    return getattr(
        getattr(getattr(embedder, "base_model", None), "config", None),
        "hidden_size",
        None,
    )


class FixedCorpusMTEBModel:
    """Encode queries with a checkpoint and passages with cached E0 embeddings."""

    def __init__(
        self,
        query_model: Any,
        corpus_model: Any,
        *,
        cache_dir: str | os.PathLike[str],
        corpus_model_name_or_path: str,
        task_name: str,
        dataset_revision: str,
    ) -> None:
        query_dimension = _embedding_dimension(query_model)
        corpus_dimension = _embedding_dimension(corpus_model)
        if (
            query_dimension is not None
            and corpus_dimension is not None
            and query_dimension != corpus_dimension
        ):
            raise ValueError(
                "Query checkpoint and fixed E0 corpus encoder dimensions differ: "
                f"{query_dimension} != {corpus_dimension}"
            )
        self.query_model = query_model
        self.corpus_model = corpus_model
        protocol = document_encoder_protocol(corpus_model_name_or_path, corpus_model)
        self.cache = PerSubsetCorpusCache(
            cache_dir,
            task_name=task_name,
            encoder_protocol=protocol,
            dataset_revision=dataset_revision,
        )
        self.mteb_model_meta = copy.copy(query_model.mteb_model_meta)
        revision_label = re.sub(
            r"[^A-Za-z0-9_.-]+", "-", str(protocol["resolved_revision"])
        ).strip("-")
        self.mteb_model_meta.name = (
            f"{self.mteb_model_meta.name}__fixed-corpus-{revision_label}"
        )
        self.world_size = getattr(query_model, "world_size", 1)

    def begin_corpus_subset(self, subset: str) -> None:
        self.cache.begin(subset)

    def finish_corpus_subset(self) -> None:
        self.cache.finish()

    def cancel_corpus_subset(self) -> None:
        self.cache.cancel()

    def encode(self, sentences, *, prompt_type=None, **kwargs):
        if prompt_type == PromptType.passage:
            return self.cache.encode(
                sentences,
                lambda: self.corpus_model.encode(
                    sentences, prompt_type=prompt_type, **kwargs
                ),
            )
        return self.query_model.encode(sentences, prompt_type=prompt_type, **kwargs)

    def start(self) -> None:
        for model in (self.query_model, self.corpus_model):
            if hasattr(model, "start"):
                model.start()

    def stop(self) -> None:
        errors = []
        for model in (self.query_model, self.corpus_model):
            if hasattr(model, "stop"):
                try:
                    model.stop()
                except Exception as exc:  # pragma: no cover - cleanup best effort
                    errors.append(exc)
        if errors:
            raise errors[0]
