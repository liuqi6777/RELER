import glob
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Sequence

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset, Sampler

from .candidates import validate_candidate_record
from .protocol import format_embedding_text, tokenize_embedding_texts

DEFAULT_TASK_PROMPT = (
    "Given a query, retrieve the documents that are relevant to the query"
)


def _length_bucket_from_path(path: str) -> str:
    """Extract the ``len-<lo>-<hi>`` length-bucket tag from a data filename.

    Files can be named like ``train_len-0-500.jsonl``; the tag is what distinguishes
    one length bucket from another within the same source. Returns the substring from
    ``len-`` to the extension, or an empty string when the filename carries no tag
    (so untagged files all share one bucket and behave exactly as before).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    marker = stem.find("len-")
    return stem[marker:] if marker != -1 else ""


@dataclass(frozen=True)
class _RetainedBatchLayout:
    """One fixed retained-data view consumed by every epoch's sampler."""

    entries: tuple[int, ...]
    groups: tuple[tuple[str, tuple[int, ...]], ...]
    dropped_batch_keys: tuple[str, ...]
    num_batches: int


def _plan_retained_batches(
    locations_by_source: dict[str, list[int]],
    batch_key_by_location: dict[int, str],
    *,
    batch_size: int,
    per_source_limit: int | None,
    rng=random,
) -> _RetainedBatchLayout:
    """Freeze the capped rows and incomplete-tail policy before epoch shuffling.

    The dataset length and retained sample identities intentionally stay fixed for the
    whole run. ``SingleSourceBatchSampler`` only recombines these positions later; it
    never decides which rows belong to the training set.
    """
    ordered_batches: list[list[int]] = []
    dropped_batch_keys: list[str] = []
    for location_ids in locations_by_source.values():
        # Copy before shuffling so index discovery and retention remain separate phases.
        shuffled_ids = list(location_ids)
        rng.shuffle(shuffled_ids)
        limited_ids = (
            shuffled_ids
            if per_source_limit is None
            else shuffled_ids[:per_source_limit]
        )
        ids_by_batch_key: dict[str, list[int]] = defaultdict(list)
        for location_id in limited_ids:
            ids_by_batch_key[batch_key_by_location[location_id]].append(location_id)
        for batch_key, bucket_ids in ids_by_batch_key.items():
            for start in range(0, len(bucket_ids), batch_size):
                batch = bucket_ids[start : start + batch_size]
                if len(batch) == batch_size:
                    ordered_batches.append(batch)
                else:
                    dropped_batch_keys.append(batch_key)

    # Keep construction-time batch order byte-for-byte compatible. Epoch samplers use
    # their own seeded torch generator and never mutate this retained layout.
    rng.shuffle(ordered_batches)
    entries = tuple(location_id for batch in ordered_batches for location_id in batch)
    positions_by_key: dict[str, list[int]] = defaultdict(list)
    for position, location_id in enumerate(entries):
        positions_by_key[batch_key_by_location[location_id]].append(position)
    return _RetainedBatchLayout(
        entries=entries,
        groups=tuple(
            (key, tuple(positions)) for key, positions in positions_by_key.items()
        ),
        dropped_batch_keys=tuple(dropped_batch_keys),
        num_batches=len(ordered_batches),
    )


class EmbeddingDataset(Dataset):
    query_prompt_template = "Instruct: {task_description}\nQuery:{query}"

    def __init__(
        self,
        data_args: Any,
        batch_size: int | None = None,
        query_prompt_template: str | None = None,
    ):
        self.batch_size = batch_size or 32
        if query_prompt_template is not None:
            self.query_prompt_template = query_prompt_template
        self.per_dataset_max_samples = data_args.per_dataset_max_samples
        raw_file_glob = getattr(data_args, "file_glob", "*.jsonl")
        self.file_glob = raw_file_glob
        self.file_globs = [g.strip() for g in raw_file_glob.split(",") if g.strip()]
        if not self.file_globs:
            raise ValueError(f"file_glob resolved to no patterns: {raw_file_glob!r}")
        self.batch_per_length_bucket = getattr(
            data_args, "batch_per_length_bucket", False
        )
        include_sources = getattr(data_args, "include_sources", None)
        self.include_sources = (
            {s.strip() for s in include_sources.split(",") if s.strip()}
            if include_sources
            else None
        )
        self.index_cache_dir = getattr(data_args, "index_cache_dir", None)

        # Keep row offsets and compact metadata in memory instead of parsed records.
        # ``_files`` holds file paths. ``entries`` is the global
        # sample order after per-source batching + shuffling; each entry is an index into
        # ``_locations`` which stores (file_id, byte_offset). ``__getitem__`` seeks +
        # parses + validates on demand. Per-worker file handles are opened lazily in
        # ``_handle`` so DataLoader workers each get their own fd after fork.
        self._files: list[dict[str, Any]] = []
        self._locations: list[tuple[int, int]] = []
        self.entries: list[int] = []
        self.batch_groups: dict[str, tuple[int, ...]] = {}
        self._file_handles: dict[int, Any] = {}
        self._record_sources: dict[str, list[str]] = {}
        self._candidate_counts: dict[str, list[int]] = {}

        self._discover_files(data_args.data_path)
        self._build_index()

    # ------------------------------------------------------------------ discovery
    def _discover_files(self, data_path: str) -> None:
        """Discover JSONL files; source identity always comes from each record."""
        if os.path.isfile(data_path):
            paths = [data_path]
        elif os.path.isdir(data_path):
            paths = sorted(
                {
                    path
                    for pattern in self.file_globs
                    for path in glob.glob(
                        os.path.join(data_path, "**", pattern), recursive=True
                    )
                    if os.path.isfile(path)
                }
            )
        else:
            raise FileNotFoundError(f"data_path does not exist: {data_path}")
        if not paths:
            raise FileNotFoundError(
                f"No data files found under {data_path!r} (glob={self.file_glob!r})"
            )
        self._files = [{"path": path} for path in paths]

    def _batch_key(self, source: str, path: str) -> str:
        """Batching key: ``source`` plus its length bucket when batching per length.

        With ``batch_per_length_bucket`` off (or a filename that carries no length tag)
        this collapses back to ``source``, so single-bucket runs are unchanged.
        """
        if not self.batch_per_length_bucket:
            return source
        bucket = _length_bucket_from_path(path)
        return f"{source}::{bucket}" if bucket else source

    # ------------------------------------------------------------------ indexing
    def _index_cache_path(self, path: str) -> str:
        if self.index_cache_dir:
            os.makedirs(self.index_cache_dir, exist_ok=True)
            safe = path.replace(os.sep, "__").lstrip("_")
            return os.path.join(self.index_cache_dir, safe + ".reler_idx.json")
        return path + ".reler_idx.json"

    def _scan_offsets(self, path: str) -> list[int]:
        """Cache validated row offsets, sources and candidate counts for lazy reads."""
        cache_path = self._index_cache_path(path)
        stat = os.stat(path)
        signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "schema": 2}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cached = json.load(f)
                offsets = cached["offsets"]
                sources = cached["sources"]
                counts = cached["candidate_counts"]
                if cached["signature"] == signature and len(offsets) == len(
                    sources
                ) == len(counts):
                    self._record_sources[path] = sources
                    self._candidate_counts[path] = counts
                    return offsets
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                pass

        offsets, sources, counts = [], [], []
        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = validate_candidate_record(json.loads(line))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"Invalid training record in {path} at byte offset {offset}: {exc}"
                    ) from exc
                offsets.append(offset)
                sources.append(record["source"])
                counts.append(len(record["document"]))
        self._record_sources[path] = sources
        self._candidate_counts[path] = counts
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "signature": signature,
                        "offsets": offsets,
                        "sources": sources,
                        "candidate_counts": counts,
                    },
                    f,
                )
        except OSError as exc:
            print(f"Warning: could not write offset cache {cache_path}: {exc}")
        return offsets

    def _build_index(self) -> None:
        """Build the per-source-capped, per-batch-key batched, shuffled sample order.

        The training cap applies per **source** so ``per_dataset_max_samples`` keeps
        its "total per task" meaning regardless of how many length buckets a source
        spans. Batching then happens per **batch_key** (source + length bucket when
        ``batch_per_length_bucket``) so each micro-batch stays single-source and
        single-length.
        """
        locations_by_source: dict[str, list[int]] = defaultdict(list)
        batch_key_by_location: dict[int, str] = {}
        candidate_count_by_location: list[int] = []
        for file_id, meta in enumerate(self._files):
            path = meta["path"]
            offsets = self._scan_offsets(path)
            for offset, source, count in zip(
                offsets, self._record_sources[path], self._candidate_counts[path]
            ):
                if (
                    self.include_sources is not None
                    and source not in self.include_sources
                ):
                    continue
                location_id = len(self._locations)
                self._locations.append((file_id, offset))
                candidate_count_by_location.append(count)
                locations_by_source[source].append(location_id)
                batch_key_by_location[location_id] = self._batch_key(source, path)

        self._batch_layout = _plan_retained_batches(
            locations_by_source,
            batch_key_by_location,
            batch_size=self.batch_size,
            per_source_limit=self.per_dataset_max_samples,
        )
        for batch_key in self._batch_layout.dropped_batch_keys:
            print(f"Skip 1 incomplete batch for dataset {batch_key}.")

        # Preserve the existing public containers while keeping the planning result as
        # one explicit contract shared by indexing diagnostics and future refactors.
        self.entries = list(self._batch_layout.entries)
        self.batch_groups = dict(self._batch_layout.groups)
        self.num_batches = self._batch_layout.num_batches
        self.max_slate_size = max(
            (candidate_count_by_location[index] for index in self.entries), default=0
        )
        print(
            f"Indexed {len(self.entries)} samples in "
            f"{self.num_batches} single-source batches "
            f"across {len(self._files)} file(s)."
        )

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.entries)

    def _format_query(self, query: str) -> str:
        return format_embedding_text(
            self.query_prompt_template,
            query,
            task_description=DEFAULT_TASK_PROMPT,
        )

    def _handle(self, file_id: int):
        handle = self._file_handles.get(file_id)
        if handle is None:
            handle = open(self._files[file_id]["path"], "rb")
            self._file_handles[file_id] = handle
        return handle

    def close(self) -> None:
        for handle in self._file_handles.values():
            handle.close()
        self._file_handles.clear()

    def __del__(self):
        # Dataset workers own independent lazy handles after fork. Closing whichever
        # handles belong to this instance avoids leaking descriptors in short probes and
        # tests while leaving normal DataLoader lifetime unchanged.
        try:
            self.close()
        except Exception:
            pass

    def _read_record(self, location_id: int) -> dict[str, Any]:
        file_id, offset = self._locations[location_id]
        handle = self._handle(file_id)
        handle.seek(offset)
        return json.loads(handle.readline())

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = validate_candidate_record(self._read_record(self.entries[index]))
        return {**record, "query": self._format_query(record["query"])}


class SingleSourceBatchSampler(Sampler[int]):
    """Recombine retained samples into single-source micro-batches each epoch.

    ``EmbeddingDataset`` applies the source cap and per-group
    tail dropping once. This sampler shuffles the retained indices within each
    source (and length bucket when enabled), then shuffles the resulting full
    micro-batches. In-batch companions can change without mixing sources or changing
    the retained dataset. Dataset indices stay stable for persistent workers.

    Both permutations use a local generator seeded by ``seed + epoch``, so every
    rank walks the same global batch order; accelerate hands whole batches to ranks
    round-robin (``split_batches=False``), and each rank still sees single-source
    micro-batches.
    """

    def __init__(
        self,
        dataset: EmbeddingDataset,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size != dataset.batch_size:
            raise ValueError(
                f"Sampler batch_size={batch_size} must match EmbeddingDataset "
                f"batch_size={dataset.batch_size} to preserve source/length groups."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            yield from range(len(self.dataset))
            return

        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        for key in sorted(self.dataset.batch_groups):
            positions = self.dataset.batch_groups[key]
            permutation = torch.randperm(len(positions), generator=generator).tolist()
            shuffled = [positions[i] for i in permutation]
            batches.extend(
                shuffled[start : start + self.batch_size]
                for start in range(0, len(shuffled), self.batch_size)
            )

        for batch_index in torch.randperm(len(batches), generator=generator).tolist():
            yield from batches[batch_index]


def build_slate_inputs(
    positive_document: Dict[str, torch.Tensor],
    negative_document: Dict[str, torch.Tensor],
    batch_size: int,
    slate_length: int,
) -> Dict[str, torch.Tensor]:
    """Re-interleave the collator's split tensors into one flat ``[batch * slate, ...]`` batch.

    ``EmbeddingDataCollator`` emits every sample's representative positive in ``positive_document``
    (``[batch, ...]``) and all remaining candidates in ``negative_document``, ordered sample-major
    (``[batch * (slate - 1), ...]``). Encoding them needs a single flat batch whose rows read
    ``(sample 0 positive, sample 0 negatives..., sample 1 positive, ...)`` so that the
    ``reshape(batch, slate, -1)`` on the far side puts each sample's own candidates into its
    own slate -- lining up with the ``relevance_labels`` the collator built in that same order.

    A plain ``cat((positive, negative), dim=0)`` does *not* satisfy this: it lays out all
    positives first, so ``reshape`` would deal other samples' positives into sample 0's slate.
    """
    num_negatives = slate_length - 1
    slate_inputs: Dict[str, torch.Tensor] = {}
    for key, positive_value in positive_document.items():
        negative_value = negative_document[key].reshape(batch_size, num_negatives, -1)
        slate_value = torch.cat((positive_value.unsqueeze(1), negative_value), dim=1)
        slate_inputs[key] = slate_value.reshape(batch_size * slate_length, -1)
    return slate_inputs


@dataclass
class CandidateLayout:
    """Candidate tensors plus the text/key order they were derived from.

    Joint and frozen-corpus training must agree on this layout exactly: position zero
    is the representative positive, remaining positions are candidates, and padding
    is appended only on the right.  Keeping that policy independent from document
    tokenization lets a frozen corpus skip needless document batches.
    """

    batch: Dict[str, Any]
    positive_documents: list[str]
    negative_documents: list[str]
    ordered_document_keys: list[list[str | None]]
    ordered_document_ids: list[list[Any | None]]


def build_candidate_layout(
    instances: Sequence[Dict[str, Any]],
    *,
    relevance_scheme: str,
    include_cross_batch_metadata: bool = False,
) -> CandidateLayout:
    """Build labels and identity masks without changing prepared candidate order."""
    if relevance_scheme not in {"binary", "graded"}:
        raise ValueError(f"Unsupported relevance_scheme: {relevance_scheme}")
    if not instances:
        raise ValueError("At least one training record is required")
    for instance in instances:
        validate_candidate_record(instance)
    width = max(len(instance["document"]) for instance in instances)

    positive_documents, negative_documents = [], []
    ordered_keys, known_id_sets, known_positive_key_sets = [], [], []
    relevance_labels, rank_labels, masks, ordered_ids, positive_masks = (
        [],
        [],
        [],
        [],
        [],
    )
    for instance in instances:
        docs = instance["document"]
        n = len(docs)
        binary_positive = torch.tensor(instance["relevance"], dtype=torch.bool)
        label_field = (
            "relevance" if relevance_scheme == "binary" else "graded_relevance"
        )
        labels = torch.tensor(instance[label_field], dtype=torch.float32)
        ranks = torch.tensor(instance["rank_labels"], dtype=torch.float32)
        positive_documents.append(docs[0])
        negative_documents.extend(docs[1:] + [""] * (width - n))
        relevance_labels.append(F.pad(labels, (0, width - n)))
        positive_masks.append(F.pad(binary_positive, (0, width - n)))
        rank_labels.append(F.pad(ranks, (0, width - n)))
        masks.append([True] * n + [False] * (width - n))
        ordered_ids.append(instance["document_ids"] + [None] * (width - n))
        ordered_keys.append(instance["document_keys"] + [None] * (width - n))
        known_id_sets.append(
            set(instance["known_document_ids"]) | set(instance["document_ids"])
        )
        known_positive_key_sets.append(
            {
                key
                for key, positive in zip(
                    instance["document_keys"], instance["relevance"]
                )
                if positive
            }
        )

    batch_size = len(instances)
    cross_masks = []
    for positive_only in (True, False):
        cross = torch.zeros(batch_size, batch_size, width, dtype=torch.bool)
        for i, instance in enumerate(instances):
            seen = (set(ordered_keys[i]) - {None}) | known_positive_key_sets[i]
            known_ids = known_id_sets[i]
            for j, other in enumerate(instances):
                if i == j:
                    continue
                for k in range(1 if positive_only else width):
                    if not masks[j][k]:
                        continue
                    key = ordered_keys[j][k]
                    candidate_id = ordered_ids[j][k]
                    known = (
                        candidate_id not in (None, "")
                        and instance.get("source") == other.get("source")
                        and candidate_id in known_ids
                    )
                    if key not in seen and not known:
                        cross[i, j, k] = True
                        seen.add(key)
        cross_masks.append(cross)

    result: Dict[str, Any] = {
        "relevance_labels": torch.stack(relevance_labels),
        "positive_mask": torch.stack(positive_masks),
        "rank_labels": torch.stack(rank_labels),
        "candidate_mask": torch.tensor(masks, dtype=torch.bool),
        "in_batch_positive_mask": cross_masks[0][..., 0],
        "in_batch_candidate_mask": cross_masks[1],
    }
    if include_cross_batch_metadata:
        result["cross_batch_metadata"] = [
            dict(
                keys=keys,
                ids=ids,
                source=instance.get("source"),
                known_ids=list(known_ids),
                known_positive_keys=list(positive_keys),
            )
            for instance, keys, ids, known_ids, positive_keys in zip(
                instances,
                ordered_keys,
                ordered_ids,
                known_id_sets,
                known_positive_key_sets,
            )
        ]
    return CandidateLayout(
        batch=result,
        positive_documents=positive_documents,
        negative_documents=negative_documents,
        ordered_document_keys=ordered_keys,
        ordered_document_ids=ordered_ids,
    )


class EmbeddingDataCollator:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        query_max_length: int = 512,
        doc_max_length: int = 1024,
        relevance_scheme: str = "binary",
        document_prompt_template: str = "{document}",
        append_token: str = "pad",
        include_cross_batch_metadata: bool = False,
        **_: Any,
    ):
        if relevance_scheme not in {"binary", "graded"}:
            raise ValueError(f"Unsupported relevance_scheme: {relevance_scheme}")
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.relevance_scheme = relevance_scheme
        self.document_prompt_template = document_prompt_template
        self.append_token = append_token
        self.include_cross_batch_metadata = include_cross_batch_metadata
        if not self.tokenizer.pad_token:
            if getattr(self.tokenizer, "eot_token", None):
                self.tokenizer.pad_token = self.tokenizer.eot_token
            elif self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.bos_token
        if not self.tokenizer.pad_token:
            raise ValueError(
                "Tokenizer has no pad/eot/eos/bos token available for batched embedding inputs"
            )
        print(f"use ``{self.tokenizer.pad_token}`` as pad token for llm")

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        layout = build_candidate_layout(
            instances,
            relevance_scheme=self.relevance_scheme,
            include_cross_batch_metadata=self.include_cross_batch_metadata,
        )
        query_inputs = tokenize_embedding_texts(
            [instance["query"] for instance in instances],
            self.tokenizer,
            self.append_token,
            max_length=self.query_max_length,
        )

        result = {"query": query_inputs, **layout.batch}
        batch_size = len(instances)
        documents = [
            format_embedding_text(self.document_prompt_template, document)
            for document in [
                *layout.positive_documents,
                *layout.negative_documents,
            ]
        ]
        document_inputs = tokenize_embedding_texts(
            documents,
            self.tokenizer,
            self.append_token,
            max_length=self.doc_max_length,
        )
        result["positive_document"] = {
            key: value[:batch_size] for key, value in document_inputs.items()
        }
        result["negative_document"] = {
            key: value[batch_size:] for key, value in document_inputs.items()
        }
        return result
