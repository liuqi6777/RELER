"""Small synthetic records in the sole training input format."""

import hashlib
import unicodedata


def document_key(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def candidate_record(documents=("positive", "negative"), **overrides):
    count = len(documents)
    ids = [str(i) for i in range(count)]
    record = {
        "schema": "embedding_candidates_v2",
        "id": "sample",
        "source": "task",
        "query": "query",
        "document": list(documents),
        "document_ids": ids,
        "document_keys": [document_key(text) for text in documents],
        "known_document_ids": overrides.get("document_ids", ids),
        "relevance": [1] + [0] * (count - 1),
        "graded_relevance": [3] + [0] * (count - 1),
        "rank_labels": list(range(count, 0, -1)),
    }
    return record | overrides
