"""The prepared candidate record shared by training and corpus construction."""

from typing import Any


def validate_candidate_record(record: Any) -> dict[str, Any]:
    """Validate a v2 record without sampling, reordering, or deriving labels.

    Candidate-aligned fields retain their input order. Position zero is the
    preselected positive representative; binary positives and teacher supervision
    are independent. Document IDs are scoped by source, while document keys
    identify duplicate text across sources. ``known_document_ids`` excludes known
    documents from cross-query pools; it is not a list of positive labels.
    """
    if (
        not isinstance(record, dict)
        or record.get("schema") != "embedding_candidates_v2"
    ):
        raise ValueError("Training records must use schema='embedding_candidates_v2'")
    for field in ("id", "source", "query"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"'{field}' must be a non-empty string")

    documents = record.get("document")
    if not isinstance(documents, list) or len(documents) < 2:
        raise ValueError("'document' must contain at least two candidates")
    count = len(documents)
    for field in ("document", "document_ids", "document_keys", "known_document_ids"):
        values = record.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"'{field}' must be a list of non-empty strings")
        if field != "known_document_ids" and len(values) != count:
            raise ValueError(f"'{field}' must match the document count")

    for field in ("relevance", "graded_relevance", "rank_labels"):
        values = record.get(field)
        if (
            not isinstance(values, list)
            or len(values) != count
            or any(type(value) is not int for value in values)
        ):
            raise ValueError(f"'{field}' must contain one integer per document")
    binary = record["relevance"]
    if set(binary) - {0, 1} or binary[0] != 1 or 0 not in binary:
        raise ValueError(
            "'relevance' must be binary, with a representative positive first "
            "and at least one negative"
        )
    if set(record["graded_relevance"]) - {0, 1, 2, 3}:
        raise ValueError("'graded_relevance' must contain grades in [0, 3]")
    if sorted(record["rank_labels"]) != list(range(1, count + 1)):
        raise ValueError("'rank_labels' must be a permutation of 1..document count")
    return record
