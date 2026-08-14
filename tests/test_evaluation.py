"""Tests for ranked retrieval evaluation metrics."""

import pytest

from evaluation.metrics import (
    average_precision,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_perfect_ranking() -> None:
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d2", "d3"}

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1.0)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)
    assert f1_at_k(retrieved, relevant, 3) == pytest.approx(1.0)
    assert average_precision(retrieved, relevant) == pytest.approx(1.0)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_partial_ranking() -> None:
    retrieved = ["d1", "x1", "d2", "x2"]
    relevant = {"d1", "d2", "d3"}

    assert precision_at_k(retrieved, relevant, 4) == pytest.approx(0.5)
    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
    assert f1_at_k(retrieved, relevant, 4) == pytest.approx(4 / 7)
    assert average_precision(retrieved, relevant) == pytest.approx(5 / 9)
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)
    assert 0.0 < ndcg_at_k(retrieved, relevant, 4) < 1.0


def test_no_relevant_documents_retrieved() -> None:
    retrieved = ["x1", "x2"]
    relevant = {"d1", "d2"}

    assert precision_at_k(retrieved, relevant, 2) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert f1_at_k(retrieved, relevant, 2) == 0.0
    assert average_precision(retrieved, relevant) == 0.0
    assert reciprocal_rank(retrieved, relevant) == 0.0
    assert ndcg_at_k(retrieved, relevant, 2) == 0.0


def test_empty_retrieved_list() -> None:
    relevant = {"d1"}

    assert precision_at_k([], relevant, 3) == 0.0
    assert recall_at_k([], relevant, 3) == 0.0
    assert f1_at_k([], relevant, 3) == 0.0
    assert average_precision([], relevant) == 0.0
    assert reciprocal_rank([], relevant) == 0.0
    assert ndcg_at_k([], relevant, 3) == 0.0


def test_empty_relevant_set() -> None:
    retrieved = ["d1", "d2"]

    assert precision_at_k(retrieved, set(), 2) == 0.0
    assert recall_at_k(retrieved, set(), 2) == 0.0
    assert f1_at_k(retrieved, set(), 2) == 0.0
    assert average_precision(retrieved, set()) == 0.0
    assert reciprocal_rank(retrieved, set()) == 0.0
    assert ndcg_at_k(retrieved, set(), 2) == 0.0


def test_precision_at_k_uses_number_retrieved_when_fewer_than_k() -> None:
    assert precision_at_k(["d1", "x1"], {"d1"}, 5) == pytest.approx(0.5)


def test_reciprocal_rank_with_later_first_relevant_document() -> None:
    assert reciprocal_rank(["x1", "x2", "d1"], {"d1"}) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "metric",
    [precision_at_k, recall_at_k, f1_at_k, ndcg_at_k],
)
@pytest.mark.parametrize("k", [0, -1])
def test_metrics_reject_non_positive_k(metric, k: int) -> None:
    with pytest.raises(ValueError):
        metric(["d1"], {"d1"}, k)
