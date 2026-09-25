import glob
import hashlib
import json
import os
import random
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Sequence

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset, Sampler

from .protocol import format_embedding_text, tokenize_embedding_texts


def document_key(text):
    """Identify duplicate documents in uncompiled teacher-ranking data."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


TASK_PROMPTS = {
    "msmarco": "Given a web search query, retrieve the documents that answer the query",
    "nq": "Given a question, retrieve Wikipedia documents that answer the question",
    "hotpotqa": "Given a multi-hop question, retrieve the documents that can help answer the question",
    "trivia": "Retrieve Wikipedia documents that answer the question",
    "t2ranking": "Given a Chinese search query, retrieve the documents that answer the query",
    "dureader": "Given a Chinese search query, retrieve the documents that answer the query",
    "mmarco_chinese": "Given a Chinese web search query, retrieve the documents that answer the query",
    "cMedQAv2": "Given a Chinese medical question, retrieve the documents that answer the question",
    "miracl": "Given a question, retrieve Wikipedia documents that answer the question",
    "allnli": "Given a premise, retrieve a hypothesis that is entailed by the premise",
    "fever": "Given a claim, retrieve documents that support or refute the claim",
    "eli5_question_answer": "Given a question, retrieve the answer that explains it",
    "squad": "Given a question, retrieve a Wikipedia passage that answers the question",
    "quora_duplicates": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "mrtydi": "Given a question, retrieve Wikipedia documents that answer the question",
    "mldr": "Given a query, retrieve the long documents that are relevant to the query",
    "law_medical": "Given a Chinese question, retrieve legal or medical documents that answer the question",
    "zh_nli": "Given a premise, retrieve a hypothesis that is entailed by the premise",
}

DEFAULT_TASK_PROMPTS = (
    "Given a query, retrieve the documents that are relevant to the query"
)


# Maps a BGE-M3 source subdirectory name to a TASK_PROMPTS key. Directory names whose
# lowercased form already matches a TASK_PROMPTS key (e.g. "cMedQAv2") need no entry.
SOURCE_DIR_TO_TASK = {
    "msmarco": "msmarco",
    "nq": "nq",
    "hotpotqa": "hotpotqa",
    "trivia": "trivia",
    "t2ranking": "t2ranking",
    "dureader": "dureader",
    "mmarco-zh": "mmarco_chinese",
    "cmedqav2": "cMedQAv2",
    "miracl": "miracl",
    "en_nli_data": "allnli",
    "zh_nli_data": "zh_nli",
    "squad": "squad",
    "mr.tydi": "mrtydi",
    "mrtydi": "mrtydi",
    "mldr": "mldr",
    "law-medical_data": "law_medical",
}


def _source_name_from_dir(dir_name: str) -> str:
    """Resolve a source subdirectory name to a TASK_PROMPTS key.

    Falls back to the lowercased directory name (which then hits DEFAULT_TASK_PROMPTS
    in ``_format_query`` if it is not a registered task).
    """
    key = dir_name.strip().lower()
    return SOURCE_DIR_TO_TASK.get(key, key)


def _length_bucket_from_path(path: str) -> str:
    """Extract the ``len-<lo>-<hi>`` length-bucket tag from a data filename.

    Files are named like ``dureader_len-0-500.jsonl``; the tag is what distinguishes
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


def record_to_slate(
    record: dict[str, Any],
    slate_size: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Convert one BGE-M3 mining record into a project slate record.

    BGE-M3 records look like ``{query, pos, neg, [pos_scores], [neg_scores]}``. We take
    one positive (highest ``pos_scores`` when present, else random) and up to
    ``slate_size - 1`` negatives (top ``neg_scores`` when present, else shuffled).
    The positive is at index 0. ``ranking`` is only a layout placeholder, not teacher
    supervision. All positive text keys survive for cross-query filtering.
    Returns ``None`` when the record has no positive or no negative.
    """
    positives = record.get("pos") or []
    negatives = record.get("neg") or []
    if not positives or not negatives:
        return None
    num_negatives = min(slate_size - 1, len(negatives))

    pos_scores = record.get("pos_scores")
    if pos_scores and len(pos_scores) == len(positives):
        positive = positives[max(range(len(positives)), key=lambda i: pos_scores[i])]
    else:
        positive = rng.choice(positives)

    neg_scores = record.get("neg_scores")
    if neg_scores and len(neg_scores) == len(negatives):
        order = sorted(range(len(negatives)), key=lambda i: neg_scores[i], reverse=True)
        chosen = [negatives[i] for i in order[:num_negatives]]
    else:
        chosen = list(negatives)
        rng.shuffle(chosen)
        chosen = chosen[:num_negatives]

    documents = [positive, *chosen]
    ranking = list(range(1, len(documents) + 1))
    return {
        "query": record["query"],
        "document": documents,
        "ranking": ranking,
        "pos_index": 1,
        "ranking_source": "pos_neg_layout",
        "known_positive_keys": sorted({document_key(text) for text in positives}),
    }


def listwise_positive_index(record: dict[str, Any], *, required: bool = False) -> int:
    """Resolve the annotated 1-based document position, separately from teacher order."""
    if "pos_index" not in record:
        if required:
            raise ValueError(
                "Binary listwise relevance requires a 1-indexed 'pos_index'"
            )
        # Legacy teacher-only graded records remain readable.
        return record["ranking"][0] - 1
    index = record["pos_index"]
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 1 <= index <= len(record["document"])
    ):
        raise ValueError("'pos_index' must be an integer in [1, len(document)]")
    return index - 1


def normalize_listwise_record(
    record: dict[str, Any], *, preserve_document_metadata: bool = False
) -> dict[str, Any]:
    """Validate and normalize an E2Rank ``{query, document, ranking}`` record.

    ``ranking`` is a 1-indexed permutation of document positions, ordered from most
    to least relevant. Unlike BGE-M3 conversion, the complete candidate list and its
    teacher ordering are preserved. Optional ``pos_index`` is the annotated 1-based
    document position, independent of teacher order; binary collation requires it.
    """
    if record.get("schema") is not None or "relevance" in record:
        raise ValueError(
            "Use the prepared candidate format for explicit relevance records"
        )
    query = record.get("query")
    documents = record.get("document")
    ranking = record.get("ranking")
    if not isinstance(query, str) or not query:
        raise ValueError("Listwise record requires a non-empty string field 'query'")
    if (
        not isinstance(documents, list)
        or not documents
        or not all(isinstance(document, str) for document in documents)
    ):
        raise ValueError(
            "Listwise record requires a non-empty string list field 'document'"
        )
    if (
        not isinstance(ranking, list)
        or len(ranking) != len(documents)
        or not all(
            isinstance(rank, int) and not isinstance(rank, bool) for rank in ranking
        )
    ):
        raise ValueError(
            "Listwise record requires an integer 'ranking' with the same length as 'document'"
        )
    expected = list(range(1, len(documents) + 1))
    if sorted(ranking) != expected:
        raise ValueError(
            f"Listwise ranking must be a 1-indexed permutation of {expected}, got {ranking}"
        )
    normalized = {"query": query, "document": list(documents), "ranking": list(ranking)}
    if "pos_index" in record:
        normalized["pos_index"] = listwise_positive_index(record) + 1
    if preserve_document_metadata:
        for field_name in (
            "document_keys",
            "document_ids",
            "known_document_ids",
            "known_positive_keys",
        ):
            value = record.get(field_name)
            if value is not None:
                normalized[field_name] = (
                    list(value) if isinstance(value, list) else value
                )
    return normalized


class EmbeddingDataset(Dataset):
    query_prompt_template = "Instruct: {task_description}\nQuery:{query}"

    def __init__(
        self,
        data_args: Any,
        batch_size: int | None = None,
        query_prompt_template: str | None = None,
        preserve_document_metadata: bool = False,
    ):
        self.batch_size = batch_size or 32
        if query_prompt_template is not None:
            self.query_prompt_template = query_prompt_template
        self.preserve_document_metadata = preserve_document_metadata
        self.per_dataset_max_samples = data_args.per_dataset_max_samples
        self.slate_size = getattr(data_args, "slate_size", 8)
        # ``file_glob`` accepts a comma-separated list of patterns so several length
        # buckets can be mixed in one run, e.g. "*_len-0-500.jsonl,*_len-500-1000.jsonl".
        # A single-pattern string stays byte-identical to the legacy behaviour.
        raw_file_glob = getattr(data_args, "file_glob", "*_len-0-500.jsonl")
        self.file_glob = raw_file_glob
        self.file_globs = [g.strip() for g in raw_file_glob.split(",") if g.strip()]
        if not self.file_globs:
            raise ValueError(f"file_glob resolved to no patterns: {raw_file_glob!r}")
        # When multiple length buckets are read, batch each bucket separately so every
        # micro-batch holds documents of one length range (less padding waste). Off by
        # default: single-bucket runs are then bit-for-bit unchanged.
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

        # Lazy-loading state (L1): we keep only byte offsets in memory, not parsed rows.
        # ``_files`` holds per-file metadata (path, source). ``entries`` is the global
        # sample order after per-source batching + shuffling; each entry is an index into
        # ``_locations`` which stores (file_id, byte_offset). ``__getitem__`` seeks +
        # parses + converts on demand. Per-worker file handles are opened lazily in
        # ``_handle`` so DataLoader workers each get their own fd after fork.
        self._files: list[dict[str, Any]] = []
        self._locations: list[tuple[int, int]] = []
        self.entries: list[int] = []
        self.batch_groups: dict[str, tuple[int, ...]] = {}
        self._file_handles: dict[int, Any] = {}
        self._record_sources: dict[str, list[str]] = {}
        self._rng = random.Random()

        self._discover_files(data_args.data_path)
        self._build_index()

    # ------------------------------------------------------------------ discovery
    def _discover_files(self, data_path: str) -> None:
        """Populate ``self._files`` with (path, source, batch_key) for every data file.

        ``source`` drives the task prompt (length-agnostic); ``batch_key`` drives
        per-source batching and additionally splits by length bucket so every
        micro-batch holds documents of one length range, minimising padding waste.
        """
        if os.path.isfile(data_path):
            source = _source_name_from_dir(os.path.basename(os.path.dirname(data_path)))
            self._files.append(
                {
                    "path": data_path,
                    "source": source,
                    "batch_key": self._batch_key(source, data_path),
                    # A standalone E2Rank file may mix tasks and carries ``source`` per
                    # record. BGE-M3 direct-file inputs simply fall back to the parent.
                    "source_from_record": True,
                }
            )
            return

        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"data_path does not exist: {data_path}")

        # Directory input: treat each immediate subdirectory as one source, reading the
        # files that match ``file_globs`` inside it. Files sitting directly under
        # data_path are also picked up (source inferred from the parent directory name).
        for entry in sorted(os.listdir(data_path)):
            full = os.path.join(data_path, entry)
            if os.path.isdir(full):
                if (
                    self.include_sources is not None
                    and entry not in self.include_sources
                ):
                    continue
                source = _source_name_from_dir(entry)
                # Union the matches across every glob, de-duplicating so overlapping
                # patterns never load the same file twice, then sort for stable order.
                matched: set[str] = set()
                for pattern in self.file_globs:
                    matched.update(glob.glob(os.path.join(full, pattern)))
                for path in sorted(matched):
                    self._files.append(
                        {
                            "path": path,
                            "source": source,
                            "batch_key": self._batch_key(source, path),
                            "source_from_record": False,
                        }
                    )
            elif entry.endswith(".jsonl") or entry.endswith(".json"):
                source = _source_name_from_dir(os.path.basename(data_path))
                self._files.append(
                    {
                        "path": full,
                        "source": source,
                        "batch_key": self._batch_key(source, full),
                        "source_from_record": True,
                    }
                )

        if not self._files:
            raise FileNotFoundError(
                f"No data files found under {data_path!r} (glob={self.file_glob!r})"
            )

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

    def _scan_offsets(
        self,
        path: str,
        *,
        source_from_record: bool = False,
        fallback_source: str = "unknown",
    ) -> list[int]:
        """Return the byte offset of every line in ``path``, caching to disk.

        BGE-M3 source-directory files only need byte offsets. Standalone/mixed
        listwise files are parsed once while indexing so their per-record ``source``
        values can preserve single-source batches; those sources are cached beside the
        offsets. The cache is invalidated when file size or mtime changes.
        """
        cache_path = self._index_cache_path(path)
        stat = os.stat(path)
        signature = {"size": stat.st_size, "mtime": int(stat.st_mtime)}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                cached_sources = cached.get("sources")
                sources_are_usable = not source_from_record or (
                    isinstance(cached_sources, list)
                    and len(cached_sources) == len(cached.get("offsets", []))
                )
                if cached.get("signature") == signature and sources_are_usable:
                    if source_from_record:
                        self._record_sources[path] = cached_sources
                    return cached["offsets"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        offsets: list[int] = []
        sources: list[str] = []
        with open(path, "rb") as f:
            offset = f.tell()
            line = f.readline()
            while line:
                if line.strip():
                    offsets.append(offset)
                    if source_from_record:
                        try:
                            record = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                            raise ValueError(
                                f"Invalid JSON record in {path} at byte offset {offset}"
                            ) from exc
                        raw_source = record.get("source") or fallback_source
                        sources.append(_source_name_from_dir(str(raw_source)))
                offset = f.tell()
                line = f.readline()
        if source_from_record:
            self._record_sources[path] = sources
        try:
            with open(cache_path, "w") as f:
                cached_index = {"signature": signature, "offsets": offsets}
                if source_from_record:
                    cached_index["sources"] = sources
                json.dump(cached_index, f)
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
        for file_id, meta in enumerate(self._files):
            offsets = self._scan_offsets(
                meta["path"],
                source_from_record=meta["source_from_record"],
                fallback_source=meta["source"],
            )
            record_sources = self._record_sources.get(meta["path"])
            for record_index, offset in enumerate(offsets):
                source = (
                    record_sources[record_index]
                    if record_sources is not None
                    else meta["source"]
                )
                batch_key = (
                    self._batch_key(source, meta["path"])
                    if meta["source_from_record"]
                    else meta["batch_key"]
                )
                location_id = len(self._locations)
                self._locations.append((file_id, offset))
                locations_by_source[source].append(location_id)
                batch_key_by_location[location_id] = batch_key

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
        print(
            f"Indexed {len(self.entries)} samples in "
            f"{self.num_batches} single-source batches "
            f"across {len(self._files)} file(s)."
        )

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.entries)

    def _format_query(self, task_name: str, query: str) -> str:
        retrieval_prompt = TASK_PROMPTS.get(task_name, DEFAULT_TASK_PROMPTS)
        return format_embedding_text(
            self.query_prompt_template,
            query,
            task_description=retrieval_prompt,
        )

    def _handle(self, file_id: int):
        handle = self._file_handles.get(file_id)
        if handle is None:
            handle = open(self._files[file_id]["path"], "r")
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

    def _convert(self, file_id: int, record: dict[str, Any]) -> dict[str, Any] | None:
        if record.get("schema") in {
            "embedding_candidates_v1",
            "embedding_candidates_v2",
        }:
            converted = dict(record)
        elif "positive" in record or "negatives" in record:
            raise ValueError(
                "Compile public records during preprocessing and load train.ready.jsonl"
            )
        elif "document" in record or "ranking" in record:
            if "document" not in record or "ranking" not in record:
                raise ValueError(
                    "Listwise records must contain both 'document' and 'ranking'"
                )
            converted = normalize_listwise_record(
                record,
                preserve_document_metadata=self.preserve_document_metadata,
            )
        elif "pos" in record or "neg" in record:
            converted = record_to_slate(record, self.slate_size, self._rng)
        else:
            raise ValueError(
                "Unsupported training record schema: expected either "
                "{query, document, ranking} or {query, pos, neg}"
            )
        if converted is None:
            return None
        raw_source = record.get("source") or self._files[file_id]["source"]
        source = _source_name_from_dir(str(raw_source))
        converted["source"] = source
        converted["query"] = self._format_query(source, converted["query"])
        return converted

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Resolve within the original storage block so an unusable BGE-M3 record
        # (no positive or negative) is replaced within the same source/length bucket.
        # This block need not be the current epoch's micro-batch. Variable slate
        # lengths are padded and masked by the collator.
        batch_start = (index // self.batch_size) * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.entries))
        order = [index] + [i for i in range(batch_start, batch_end) if i != index]
        for candidate in order:
            location_id = self.entries[candidate]
            file_id, _ = self._locations[location_id]
            record = self._read_record(location_id)
            converted = self._convert(file_id, record)
            if converted is not None:
                return converted
        raise RuntimeError(
            f"No convertible sample in batch starting at {batch_start}; "
            "records may be missing positives or negatives."
        )


class SingleSourceBatchSampler(Sampler[int]):
    """Recombine retained samples into single-source micro-batches each epoch.

    ``EmbeddingDataset`` applies the source cap, optional dev split and per-group
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


def build_relevance_labels(
    ranking: torch.Tensor,
    scheme: str = "graded",
) -> torch.Tensor:
    if ranking is None:
        raise ValueError("ranking is required to build relevance labels")
    if ranking.dim() != 2:
        raise ValueError(
            f"ranking must be a 2D tensor, got shape {tuple(ranking.shape)}"
        )
    if scheme not in {"graded", "binary"}:
        raise ValueError(f"Unsupported relevance scheme: {scheme}")

    batch_size, slate_length = ranking.shape
    relevance = torch.zeros(
        batch_size, slate_length, device=ranking.device, dtype=torch.float32
    )

    rank_scores = torch.zeros(slate_length, device=ranking.device, dtype=torch.float32)
    if slate_length > 0:
        rank_scores[0] = 3.0 if scheme == "graded" else 1.0
    if scheme == "graded":
        if slate_length > 1:
            rank_scores[1 : min(5, slate_length)] = 2.0
        if slate_length > 5:
            rank_scores[5 : min(10, slate_length)] = 1.0

    relevance.scatter_(
        dim=1,
        index=ranking,
        src=rank_scores.unsqueeze(0).expand(batch_size, -1),
    )
    return relevance


def build_rank_labels(ranking: torch.Tensor) -> torch.Tensor:
    """Invert a 0-indexed teacher permutation into dense higher-is-better labels."""
    if ranking.dim() != 2:
        raise ValueError(
            f"ranking must be a 2D tensor, got shape {tuple(ranking.shape)}"
        )
    batch_size, slate_length = ranking.shape
    labels = torch.zeros_like(ranking, dtype=torch.float32)
    rank_values = torch.arange(
        slate_length,
        0,
        -1,
        device=ranking.device,
        dtype=torch.float32,
    )
    labels.scatter_(
        dim=1,
        index=ranking,
        src=rank_values.unsqueeze(0).expand(batch_size, -1),
    )
    return labels


def build_slate_inputs(
    positive_document: Dict[str, torch.Tensor],
    negative_document: Dict[str, torch.Tensor],
    batch_size: int,
    slate_length: int,
) -> Dict[str, torch.Tensor]:
    """Re-interleave the collator's split tensors into one flat ``[batch * slate, ...]`` batch.

    ``EmbeddingDataCollator`` emits every sample's gold positive in ``positive_document``
    (``[batch, ...]``) and all negatives in ``negative_document``, ordered sample-major
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
    use_unprepared_document_metadata: bool = False,
) -> CandidateLayout:
    """Build labels, masks and candidate order shared by both training modes.

    Joint training retains its historical text-hash identity for raw listwise rows.
    Fixed-corpus callers opt into explicit keys on those rows so the key used for
    ordinal lookup remains stable even when its text representation changes.
    """
    explicit = ["relevance" in instance for instance in instances]
    if any(
        "relevance" in item
        and item.get("schema")
        not in {"embedding_candidates_v1", "embedding_candidates_v2"}
        for item in instances
    ):
        raise ValueError(
            "Explicit relevance must use preprocessed embedding_candidates_v1/v2 records"
        )
    if any(explicit) and not all(explicit):
        raise ValueError(
            "Do not mix explicit relevance and teacher-grade records in one batch"
        )
    width = max(len(instance["document"]) for instance in instances)
    if width < 2:
        raise ValueError("At least two candidates are required")

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
        ready = instance.get("schema") in {
            "embedding_candidates_v1",
            "embedding_candidates_v2",
        }
        if ready:
            labels = torch.tensor(instance["relevance"], dtype=torch.float32)
            ranks = torch.tensor(instance["rank_labels"], dtype=torch.float32)
            if (
                len(labels) != n
                or len(ranks) != n
                or labels[0] != 1
                or not ((labels == 0) | (labels == 1)).all()
            ):
                raise ValueError(
                    "Prepared binary labels must match candidates with a representative positive first"
                )
            binary_positive = labels.bool()
            if relevance_scheme == "graded":
                labels = torch.tensor(instance["graded_relevance"], dtype=torch.float32)
                if len(labels) != n:
                    raise ValueError("Prepared grades must match candidates")
            positive_index = 0
        else:
            mined = instance.get("ranking_source") == "pos_neg_layout"
            if mined and relevance_scheme != "binary":
                raise ValueError(
                    "BGE-M3 pos/neg data requires binary relevance; candidate order "
                    "is not a teacher ranking for graded supervision"
                )
            ranking = torch.tensor([instance["ranking"]], dtype=torch.long) - 1
            ranks = build_rank_labels(ranking)[0]
            positive_index = listwise_positive_index(
                instance, required=relevance_scheme == "binary"
            )
            binary_positive = torch.zeros(n, dtype=torch.bool)
            binary_positive[positive_index] = True
            labels = (
                binary_positive.float()
                if relevance_scheme == "binary"
                else build_relevance_labels(ranking, "graded")[0]
            )
            if mined:
                # No preference is annotated between mined negatives.
                ranks = binary_positive.float()
        order = (
            list(range(n))
            if ready
            else [positive_index] + [i for i in range(n) if i != positive_index]
        )
        reordered = [docs[i] for i in order]
        positive_documents.append(reordered[0])
        negative_documents.extend(reordered[1:] + [""] * (width - n))
        relevance_labels.append(F.pad(labels[order], (0, width - n)))
        positive_masks.append(F.pad(binary_positive[order], (0, width - n)))
        rank_labels.append(F.pad(ranks[order], (0, width - n)))
        masks.append([True] * n + [False] * (width - n))
        use_metadata = ready or use_unprepared_document_metadata
        ids = instance.get("document_ids") if use_metadata else None
        if ids is None:
            ids = [None] * n
        elif not isinstance(ids, list):
            raise ValueError("document_ids must be a list")
        if len(ids) != n:
            raise ValueError("document_ids must match document count")
        ordered_ids.append([ids[i] for i in order] + [None] * (width - n))
        keys = instance.get("document_keys") if use_metadata else None
        if keys is None:
            keys = [document_key(document) for document in docs]
        elif not isinstance(keys, list):
            raise ValueError("document_keys must be a list")
        if len(keys) != n:
            raise ValueError("document_keys must match document count")
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError(
                "document_keys must contain one non-empty string per candidate"
            )
        ordered_keys.append([keys[i] for i in order] + [None] * (width - n))
        if ready:
            known_ids = set(instance["known_document_ids"])
        elif use_unprepared_document_metadata:
            known_ids = set(instance.get("known_document_ids", [])) | set(ids)
        else:
            known_ids = set()
        known_id_sets.append(known_ids - {None, ""})
        known_positive_key_sets.append(
            set(instance.get("known_positive_keys", []))
            if ready or use_unprepared_document_metadata or mined
            else set()
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
        query_inputs = tokenize_embedding_texts(
            [instance["query"] for instance in instances],
            self.tokenizer,
            self.append_token,
            max_length=self.query_max_length,
        )

        layout = build_candidate_layout(
            instances,
            relevance_scheme=self.relevance_scheme,
            include_cross_batch_metadata=self.include_cross_batch_metadata,
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
