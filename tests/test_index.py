import pytest

from src.indexing import InvertedIndex


@pytest.fixture
def documents() -> list[dict[str, object]]:
    return [
        {
            "document_id": "doc1",
            "text": "Oil prices rise oil",
            "categories": ["energy"],
            "split": "training",
        },
        {
            "document_id": "doc2",
            "text": "Bank trade prices",
            "categories": ["finance"],
            "split": "test",
        },
    ]


@pytest.fixture
def index(documents: list[dict[str, object]]) -> InvertedIndex:
    inverted_index = InvertedIndex()
    inverted_index.build(
        documents, normalization="none", remove_stopwords=True
    )
    return inverted_index


def test_document_count(index: InvertedIndex) -> None:
    assert index.document_count == 2


def test_document_lengths(index: InvertedIndex) -> None:
    assert index.document_lengths == {"doc1": 4, "doc2": 3}


def test_term_frequencies(index: InvertedIndex) -> None:
    assert index.get_postings("oil") == {"doc1": 2}
    assert index.get_postings("prices") == {"doc1": 1, "doc2": 1}
    assert index.get_postings("price") == {}


def test_document_frequencies(index: InvertedIndex) -> None:
    assert index.document_frequencies["oil"] == 1
    assert index.document_frequencies["prices"] == 2


def test_average_document_length(index: InvertedIndex) -> None:
    assert index.average_document_length == pytest.approx(3.5)


def test_get_postings_returns_empty_dict_for_unknown_term(
    index: InvertedIndex,
) -> None:
    assert index.get_postings("unknown") == {}


def test_rebuild_clears_previous_index(index: InvertedIndex) -> None:
    replacement_documents: list[dict[str, object]] = [
        {
            "document_id": "doc3",
            "text": "Wheat exports",
            "categories": ["trade"],
            "split": "training",
        }
    ]

    index.build(
        replacement_documents,
        normalization="none",
        remove_stopwords=True,
    )

    assert index.document_count == 1
    assert index.document_lengths == {"doc3": 2}
    assert index.get_postings("wheat") == {"doc3": 1}
    assert index.get_postings("oil") == {}
    assert "oil" not in index.document_frequencies
