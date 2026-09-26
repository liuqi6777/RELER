"""Candidate collation for query-only training against immutable documents."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import transformers

from reler.data.embedding import build_candidate_layout
from reler.data.protocol import tokenize_embedding_texts


class FixedCorpusDataCollator:
    """Build the normal candidate layout without tokenizing document text.

    The index owns an immutable ``document identity -> ordinal`` mapping. Candidate
    text is retained in the dataset for compatibility and auditability, but only
    its stable key reaches the training model; document embeddings are fetched by
    ordinal from the fixed index.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        index,
        *,
        query_max_length: int = 512,
        relevance_scheme: str = "binary",
        append_token: str = "pad",
        include_cross_batch_metadata: bool = False,
    ) -> None:
        if relevance_scheme not in {"binary", "graded"}:
            raise ValueError(f"Unsupported relevance_scheme: {relevance_scheme}")
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.relevance_scheme = relevance_scheme
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
        self.document_to_ordinal = index.document_to_ordinal

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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

        candidate_mask = layout.batch["candidate_mask"]
        ordinals = torch.full_like(candidate_mask, -1, dtype=torch.long)
        missing: list[tuple[int, int, object, str]] = []
        for row, keys in enumerate(layout.ordered_document_keys):
            for column, key in enumerate(keys):
                if not candidate_mask[row, column]:
                    continue
                assert key is not None  # The layout guarantees a key for valid rows.
                document_id = layout.ordered_document_ids[row][column]
                id_ordinal = (
                    self.document_to_ordinal.get(str(document_id))
                    if document_id not in (None, "")
                    else None
                )
                key_ordinal = self.document_to_ordinal.get(key)
                if (
                    id_ordinal is not None
                    and key_ordinal is not None
                    and id_ordinal != key_ordinal
                ):
                    raise ValueError(
                        "Frozen corpus candidate identity is ambiguous at "
                        f"row={row}, candidate={column}: document_id={document_id!r} "
                        f"maps to ordinal {id_ordinal}, while document_key={key!r} "
                        f"maps to ordinal {key_ordinal}."
                    )
                ordinal = id_ordinal if id_ordinal is not None else key_ordinal
                if ordinal is None:
                    missing.append((row, column, document_id, key))
                else:
                    ordinals[row, column] = ordinal
        if missing:
            preview = ", ".join(
                f"row={row}, candidate={column}, document_id={document_id!r}, key={key!r}"
                for row, column, document_id, key in missing[:3]
            )
            raise KeyError(
                "Frozen corpus does not contain candidate document ID or key: "
                f"{preview}. Rebuild the index from this corpus or provide matching "
                "document_ids/document_keys."
            )
        return {
            "query": query_inputs,
            "candidate_ordinals": ordinals,
            **layout.batch,
        }
