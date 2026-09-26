"""Build a frozen document-vector directory without training-code coupling."""

from __future__ import annotations

import argparse
import json
import math
import os
import unicodedata
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from reler.config.loader import load_raw_config_file
from reler.data.candidates import validate_candidate_record
from reler.data.protocol import (
    EMBEDDING_PROTOCOL_FILENAME,
    POOLING_COMPUTE_DTYPE,
    format_embedding_text,
    pool_embeddings,
    tokenization_metadata,
    tokenize_embedding_texts,
)
from reler.fixed_corpus.index import DOCUMENT_MAPPING_FILENAME


def normalize_document(text: str) -> str:
    """Canonicalize text only for duplicate-key consistency checks."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _iter_source_documents(
    source_path: Path, input_format: str
) -> Iterator[tuple[str, str, int | None]]:
    if input_format == "document_jsonl":
        with source_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    yield str(record["id"]), record["content"], line_number
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid document JSON at {source_path}:{line_number}"
                    ) from exc
        return

    if input_format != "training_candidates":
        raise ValueError(f"Unsupported input_format: {input_format!r}")
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = validate_candidate_record(json.loads(line))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid training record at {source_path}:{line_number}: {exc}"
                ) from exc
            # IDs are local to a source; prepared text keys are shared across sources.
            for text, key in zip(record["document"], record["document_keys"]):
                yield key, text, line_number


def build_corpus(
    source_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    input_format: str = "training_candidates",
) -> int:
    """Write unique documents, row offsets and a stable key-to-ordinal mapping."""
    source_path, output_dir = Path(source_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    offsets: list[int] = []
    key_to_ordinal: dict[str, int] = {}
    normalized_by_key: dict[str, str] = {}
    with corpus_path.open("wb") as corpus:
        for key, text, line_number in _iter_source_documents(source_path, input_format):
            location = (
                f"{source_path}:{line_number}" if line_number else str(source_path)
            )
            if not isinstance(text, str) or not text.strip() or not key:
                raise ValueError(
                    f"Non-empty string document/key required at {location}"
                )
            normalized = normalize_document(text)
            if key in normalized_by_key:
                if normalized_by_key[key] != normalized:
                    raise ValueError(
                        f"Conflicting normalized text for document_key={key}"
                    )
                continue
            key_to_ordinal[key] = len(key_to_ordinal)
            normalized_by_key[key] = normalized
            offsets.append(corpus.tell())
            corpus.write(
                json.dumps(
                    {"document_key": key, "contents": text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        corpus.flush()
        os.fsync(corpus.fileno())
    if not key_to_ordinal:
        raise ValueError(f"No documents found in {source_path}")
    np.save(output_dir / "corpus_offsets.npy", np.asarray(offsets, dtype=np.int64))
    with (output_dir / DOCUMENT_MAPPING_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(key_to_ordinal, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    return len(key_to_ordinal)


def _read_texts(handle: Any, offsets: np.ndarray, start: int, end: int) -> list[str]:
    texts = []
    for ordinal in range(start, end):
        handle.seek(int(offsets[ordinal]))
        texts.append(json.loads(handle.readline())["contents"])
    return texts


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _validate_vectors(vectors: np.ndarray, path: Path) -> None:
    """Establish the normalization invariant before the shard becomes visible."""
    block_size = 1 << 17
    for start in range(0, len(vectors), block_size):
        block = np.asarray(vectors[start : start + block_size], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"Encoded shard contains non-finite vectors: {path}")
        norms = np.linalg.norm(block, axis=-1)
        if not np.allclose(norms, 1.0, rtol=2e-3, atol=2e-3):
            raise ValueError(f"Encoded shard contains non-normalized vectors: {path}")


def encode_corpus_shards(
    *,
    corpus_path: str | os.PathLike[str],
    offsets_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    model_name_or_path: str,
    revision: str | None,
    num_shards: int,
    batch_size: int,
    max_length: int,
    pooling_method: str,
    padding_side: str,
    append_token: str,
    document_prompt_template: str,
    device: torch.device | str = "cpu",
    overwrite: bool = False,
    tokenizer: Any | None = None,
    model: Any | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Encode contiguous fp16 shards of an existing corpus."""
    if any(value <= 0 for value in (num_shards, batch_size, max_length)):
        raise ValueError("shard, batch and maximum-length settings must be positive")
    device = torch.device(device)
    corpus_path, offsets_path, output_dir = (
        Path(corpus_path),
        Path(offsets_path),
        Path(output_dir),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        padding_side=padding_side,
        trust_remote_code=True,
    )
    if not getattr(tokenizer, "pad_token", None):
        tokenizer.pad_token = (
            getattr(tokenizer, "eot_token", None)
            or getattr(tokenizer, "eos_token", None)
            or getattr(tokenizer, "bos_token", None)
        )
    if not getattr(tokenizer, "pad_token", None):
        raise ValueError(
            "Frozen corpus tokenizer must define a padding-compatible token"
        )
    model = model or AutoModel.from_pretrained(
        model_name_or_path,
        revision=revision,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device).eval()
    dimension = int(model.config.hidden_size)
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    offsets = np.load(offsets_path, mmap_mode="r")
    count = len(offsets)
    if count <= 0:
        raise ValueError("Corpus must contain at least one document")
    per_shard = math.ceil(count / num_shards)
    token_protocol = {
        **tokenization_metadata(tokenizer, append_token),
        "pooling_compute_dtype": POOLING_COMPUTE_DTYPE,
    }

    with corpus_path.open("rb") as corpus:
        for shard_id in range(num_shards):
            start, end = shard_id * per_shard, min((shard_id + 1) * per_shard, count)
            if start >= end:
                continue
            output_path = output_dir / f"vectors-{shard_id:05d}.npy"
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Index shard already exists: {output_path}")
            partial_path = output_path.with_suffix(".npy.partial")
            vectors = np.lib.format.open_memmap(
                partial_path,
                mode="w+",
                dtype=np.float16,
                shape=(end - start, dimension),
            )
            for batch_start in range(start, end, batch_size):
                batch_end = min(batch_start + batch_size, end)
                texts = [
                    format_embedding_text(document_prompt_template, text)
                    for text in _read_texts(corpus, offsets, batch_start, batch_end)
                ]
                inputs = _move_to_device(
                    tokenize_embedding_texts(
                        texts, tokenizer, append_token, max_length=max_length
                    ),
                    device,
                )
                with torch.inference_mode():
                    embeddings = pool_embeddings(
                        model(**inputs).last_hidden_state,
                        inputs["attention_mask"],
                        pooling_method=pooling_method,
                    )
                vectors[batch_start - start : batch_end - start] = (
                    embeddings.float().cpu().numpy()
                )
            vectors.flush()
            _validate_vectors(vectors, output_path)
            del vectors
            os.replace(partial_path, output_path)

    return resolved_revision, token_protocol


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lookup-only RELER frozen corpus"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--input-format",
        choices=("training_candidates", "document_jsonl"),
        default="training_candidates",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--pooling-method", default=None)
    parser.add_argument("--padding-side", default=None)
    parser.add_argument("--append-token", default=None)
    parser.add_argument("--document-prompt-template", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    model_config = load_raw_config_file(args.model_config) if args.model_config else {}
    defaults = {
        "pooling_method": "last",
        "padding_side": "left",
        "append_token": "pad",
        "document_prompt_template": "{document}",
    }
    args.model = args.model or model_config.get("model_name_or_path")
    args.revision = args.revision or model_config.get("model_revision")
    for name, default in defaults.items():
        setattr(args, name, getattr(args, name) or model_config.get(name, default))
    if not args.model:
        parser.error("one of --model or --model-config is required")
    return args


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for reler-index but is unavailable")
    return device


def main(argv: list[str] | None = None) -> None:
    """Build an atomic fixed-corpus directory from a document source."""
    args = _parse_args(argv)
    if any(value <= 0 for value in (args.num_shards, args.batch_size, args.max_length)):
        raise ValueError("shard, batch and maximum-length settings must be positive")
    source_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing frozen index directory: {output_dir}"
        )
    building = output_dir.with_name(f"{output_dir.name}.building")
    if building.exists():
        raise FileExistsError(f"Remove or inspect stale build directory: {building}")
    building.mkdir(parents=True)
    try:
        count = build_corpus(source_path, building, args.input_format)
        resolved_revision, token_protocol = encode_corpus_shards(
            corpus_path=building / "corpus.jsonl",
            offsets_path=building / "corpus_offsets.npy",
            output_dir=building,
            model_name_or_path=args.model,
            revision=args.revision,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling_method=args.pooling_method,
            padding_side=args.padding_side,
            append_token=args.append_token,
            document_prompt_template=args.document_prompt_template,
            device=_resolve_device(args.device),
        )
        protocol = {
            **token_protocol,
            "model_name_or_path": args.model,
            "resolved_model_revision": resolved_revision,
            "pooling_method": args.pooling_method,
            "padding_side": args.padding_side,
            "append_token": args.append_token,
            "document_prompt_template": args.document_prompt_template,
            "document_max_length": args.max_length,
            "normalize": True,
            "storage_dtype": "float16",
        }
        with (building / EMBEDDING_PROTOCOL_FILENAME).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(protocol, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        (building / "corpus.jsonl").unlink()
        (building / "corpus_offsets.npy").unlink()
        os.replace(building, output_dir)
    except Exception:
        # Keep the incomplete directory for inspection; it cannot be mistaken for an index.
        raise
    print(f"Built {count} frozen documents: {output_dir}")


if __name__ == "__main__":
    main()
