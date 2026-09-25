"""Read-only document-vector lookup for query-only RELER training."""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reler.data.protocol import (
    EMBEDDING_PROTOCOL_FILENAME,
    POOLING_COMPUTE_DTYPE,
    TOKENIZATION_VERSION,
    load_embedding_protocol,
)

DOCUMENT_MAPPING_FILENAME = "document_to_ordinal.json"
VECTOR_GLOB = "vectors-*.npy"
_VECTOR_NAME = re.compile(r"^vectors-(\d{5})\.npy$")


def validate_frozen_protocol(
    protocol: dict[str, Any],
    *,
    model_name_or_path: str,
    resolved_model_revision: str | None,
    pooling_method: str,
    padding_side: str,
    append_token: str,
    document_prompt_template: str | None = None,
    document_max_length: int | None = None,
) -> None:
    """Reject document vectors produced by a different embedding protocol."""
    required = (
        "model_name_or_path",
        "pooling_method",
        "padding_side",
        "append_token",
        "document_prompt_template",
        "document_max_length",
    )
    missing = [key for key in required if protocol.get(key) is None]
    if missing:
        raise ValueError(
            f"Frozen document protocol is missing required fields: {', '.join(missing)}"
        )
    if protocol.get("tokenization_version") != TOKENIZATION_VERSION:
        raise ValueError(
            f"Frozen document tokenization_version must be {TOKENIZATION_VERSION}; "
            "rebuild the index with the current embedding protocol."
        )
    if protocol.get("pooling_compute_dtype") != POOLING_COMPUTE_DTYPE:
        raise ValueError(
            "Frozen document pooling_compute_dtype must be float32; rebuild the index"
        )
    if protocol.get("normalize") is not True:
        raise ValueError(
            "Frozen document vectors must be normalized; rebuild the index"
        )
    if protocol.get("storage_dtype") != "float16":
        raise ValueError(
            "Frozen document vectors must use float16 storage; rebuild the index"
        )

    expected_model = protocol.get("model_name_or_path")
    if expected_model and expected_model != model_name_or_path:
        raise ValueError(
            f"Model {model_name_or_path!r} differs from frozen document model "
            f"{expected_model!r}"
        )
    expected_revision = protocol.get("resolved_model_revision")
    if expected_revision and expected_revision != resolved_model_revision:
        raise ValueError(
            "Model revision differs from the frozen document encoder revision"
        )

    for key, actual in (
        ("pooling_method", pooling_method),
        ("padding_side", padding_side),
        ("append_token", append_token),
        ("document_prompt_template", document_prompt_template),
        ("document_max_length", document_max_length),
    ):
        expected = protocol.get(key)
        if actual is not None and expected is not None and expected != actual:
            raise ValueError(
                f"Frozen document protocol mismatch for {key}: "
                f"{actual!r} != {expected!r}"
            )


class FrozenCorpusIndex:
    """Lookup-only vectors stored in a small, fixed directory layout.

    The directory contains ``embedding_protocol.json``,
    ``document_to_ordinal.json`` and one or more ``vectors-*.npy`` shards.
    Shards are ordered by filename and rows are addressed by global ordinal.
    """

    def __init__(
        self,
        index_dir: str | Path,
        *,
        backend: str = "lookup",
        device: str | torch.device = "cpu",
    ) -> None:
        if backend.lower() != "lookup":
            raise ValueError("FrozenCorpusIndex only supports backend='lookup'")
        self.root = Path(index_dir).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.backend = "lookup"
        self.device = torch.device(device)
        self.embedding_protocol = load_embedding_protocol(self.root)
        if not self.embedding_protocol:
            raise FileNotFoundError(self.root / EMBEDDING_PROTOCOL_FILENAME)

        self._mapping = self._load_mapping()
        self._shard_paths = sorted(self.root.glob(VECTOR_GLOB))
        if not self._shard_paths:
            raise FileNotFoundError(f"No {VECTOR_GLOB} files found in {self.root}")
        shard_ids = [
            int(match.group(1))
            if (match := _VECTOR_NAME.fullmatch(path.name))
            else None
            for path in self._shard_paths
        ]
        if shard_ids != list(range(len(self._shard_paths))):
            raise ValueError(
                "Frozen vector shards must be consecutively named "
                "vectors-00000.npy, vectors-00001.npy, ..."
            )
        self._shard_starts: list[int] = []
        self._memmaps: list[np.ndarray] | None = None
        self._closed = False
        self.count, self.dimension = self._validate_layout()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("FrozenCorpusIndex is closed")

    def _load_mapping(self) -> dict[str, int]:
        path = self.root / DOCUMENT_MAPPING_FILENAME
        with path.open(encoding="utf-8") as handle:
            raw_mapping = json.load(handle)
        if not isinstance(raw_mapping, dict) or not raw_mapping:
            raise ValueError("Document mapping must be a non-empty JSON object")
        try:
            mapping = {str(key): int(value) for key, value in raw_mapping.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Document mapping ordinals must be integers") from exc
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError(
                "Document mapping ordinals must be a complete 0-based permutation"
            )
        return mapping

    def _validate_layout(self) -> tuple[int, int]:
        count = 0
        dimension: int | None = None
        for path in self._shard_paths:
            vectors = np.load(path, mmap_mode="r")
            try:
                if vectors.ndim != 2 or min(vectors.shape) <= 0:
                    raise ValueError(
                        f"Frozen vector shard must have shape [rows, dimension]: {path}"
                    )
                if vectors.dtype != np.float16:
                    raise ValueError(
                        f"Shard dtype mismatch for {path}: {vectors.dtype} != float16"
                    )
                shard_dimension = int(vectors.shape[1])
                if dimension is None:
                    dimension = shard_dimension
                elif shard_dimension != dimension:
                    raise ValueError(
                        f"Shard dimension mismatch for {path}: "
                        f"{shard_dimension} != {dimension}"
                    )
                self._shard_starts.append(count)
                count += int(vectors.shape[0])
            finally:
                mmap = getattr(vectors, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        if count != len(self._mapping):
            raise ValueError(
                f"Vector row count {count} does not match document mapping "
                f"count {len(self._mapping)}"
            )
        assert dimension is not None
        return count, dimension

    def validate_protocol(self, **kwargs: Any) -> None:
        self._require_open()
        validate_frozen_protocol(self.embedding_protocol, **kwargs)

    @property
    def document_to_ordinal(self) -> Mapping[str, int]:
        """Read-only by contract; shared with the collator to avoid a large copy."""
        self._require_open()
        return self._mapping

    def lookup_ordinals(self, document_keys: Iterable[str]) -> list[int]:
        self._require_open()
        keys = [str(key) for key in document_keys]
        missing = [key for key in keys if key not in self._mapping]
        if missing:
            raise KeyError(f"Unknown document_key(s): {', '.join(missing[:3])}")
        return [self._mapping[key] for key in keys]

    def _ensure_memmaps(self) -> list[np.ndarray]:
        self._require_open()
        if self._memmaps is None:
            self._memmaps = [np.load(path, mmap_mode="r") for path in self._shard_paths]
        return self._memmaps

    def lookup_embeddings(
        self,
        ordinals: torch.Tensor,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Fetch shard rows as detached fp32 embeddings on ``device``."""
        self._require_open()
        if not isinstance(ordinals, torch.Tensor):
            raise TypeError("ordinals must be a torch.Tensor")
        values = (
            ordinals.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
        )
        if values and (min(values) < 0 or max(values) >= self.count):
            invalid = min(values) if min(values) < 0 else max(values)
            raise IndexError(f"Corpus ordinal out of range: {invalid}")
        target = self.device if device is None else torch.device(device)
        if not values:
            return torch.empty(
                (*ordinals.shape, self.dimension), dtype=torch.float32, device=target
            )

        memmaps = self._ensure_memmaps()
        rows = []
        for ordinal in values:
            shard_index = bisect_right(self._shard_starts, ordinal) - 1
            rows.append(
                np.asarray(
                    memmaps[shard_index][ordinal - self._shard_starts[shard_index]],
                    dtype=np.float32,
                ).copy()
            )
        result = torch.from_numpy(np.stack(rows)).reshape(
            *ordinals.shape, self.dimension
        )
        return result.to(device=target, dtype=torch.float32).detach()

    def close(self) -> None:
        if self._closed:
            return
        if self._memmaps is not None:
            for memmap in self._memmaps:
                mmap = getattr(memmap, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        self._memmaps = None
        self._closed = True

    def __enter__(self) -> "FrozenCorpusIndex":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
