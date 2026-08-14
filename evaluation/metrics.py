"""Evaluation metrics for ranked information-retrieval results."""

from math import log2


def _validate_k(k: int) -> None:
    """Raise an error when a ranking cutoff is not positive."""
    if k <= 0:
        raise ValueError("k must be greater than 0")


def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Return the proportion of retrieved documents in the top k that are relevant."""
    _validate_k(k)
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = sum(document_id in relevant for document_id in top_k)
    return relevant_retrieved / len(top_k)


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Return the proportion of relevant documents retrieved in the top k."""
    _validate_k(k)
    if not relevant:
        return 0.0
    relevant_retrieved = sum(
        document_id in relevant for document_id in retrieved[:k]
    )
    return relevant_retrieved / len(relevant)


def f1_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Return the harmonic mean of precision and recall at k."""
    _validate_k(k)
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Return average precision for one ranked result list."""
    if not relevant:
        return 0.0

    relevant_retrieved = 0
    precision_sum = 0.0
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            relevant_retrieved += 1
            precision_sum += relevant_retrieved / rank
    return precision_sum / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Return the reciprocal rank of the first relevant document."""
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Return normalized discounted cumulative gain at k for binary relevance."""
    _validate_k(k)
    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, document_id in enumerate(retrieved[:k], start=1)
        if document_id in relevant
    )
    ideal_relevant_count = min(len(relevant), k)
    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
