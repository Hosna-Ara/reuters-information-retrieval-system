"""Tests for TF-IDF and BM25 ranking."""

import pytest

from src.bm25 import BM25Ranker
from src.indexing import InvertedIndex
from src.tfidf import TfidfRanker


@pytest.fixture
def index() -> InvertedIndex:
    documents = [
        {
            "document_id": "doc1",
            "text": "oil oil market",
            "categories": ["energy"],
            "split": "training",
        },
        {
            "document_id": "doc2",
            "text": "oil market market",
            "categories": ["energy"],
            "split": "training",
        },
        {
            "document_id": "doc3",
            "text": "bank finance",
            "categories": ["finance"],
            "split": "test",
        },
    ]
    inverted_index = InvertedIndex()
    inverted_index.build(
        documents,
        normalization="none",
        remove_stopwords=True,
    )
    return inverted_index


@pytest.fixture(params=[TfidfRanker, BM25Ranker], ids=["tfidf", "bm25"])
def ranker(request: pytest.FixtureRequest, index: InvertedIndex):
    return request.param(index)


def test_oil_ranking(ranker) -> None:
    results = ranker.rank(["oil"])

    assert [document_id for document_id, _ in results] == ["doc1", "doc2"]


@pytest.mark.parametrize("query", [["unknown"], []], ids=["out-of-vocabulary", "empty"])
def test_no_results_for_unmatched_query(ranker, query: list[str]) -> None:
    assert ranker.rank(query) == []


def test_top_k_limits_results(ranker) -> None:
    assert [document_id for document_id, _ in ranker.rank(["oil"], top_k=1)] == [
        "doc1"
    ]


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_raises(ranker, top_k: int) -> None:
    with pytest.raises(ValueError):
        ranker.rank(["oil"], top_k=top_k)


@pytest.mark.parametrize(
    ("parameters"),
    [
        {"k1": 0},
        {"k1": -1},
        {"b": -0.1},
        {"b": 1.1},
    ],
)
def test_bm25_rejects_invalid_parameters(
    index: InvertedIndex,
    parameters: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        BM25Ranker(index, **parameters)


def test_ties_are_broken_by_document_id(ranker) -> None:
    results = ranker.rank(["oil", "market"])

    assert results[0][1] == pytest.approx(results[1][1])
    assert [document_id for document_id, _ in results] == ["doc1", "doc2"]
