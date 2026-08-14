"""Inverted-index construction for the Reuters IR system."""

from collections import Counter, defaultdict
from typing import cast

from src.preprocessing import preprocess_text


class InvertedIndex:
    """Store term postings and document-level index statistics."""

    def __init__(self) -> None:
        """Initialize an empty inverted index."""
        self.postings: defaultdict[str, dict[str, int]] = defaultdict(dict)
        self.document_lengths: dict[str, int] = {}
        self.document_frequencies: defaultdict[str, int] = defaultdict(int)
        self.document_count: int = 0
        self.average_document_length: float = 0.0

    def build(
        self,
        documents: list[dict[str, object]],
        normalization: str = "lemmatize",
        remove_stopwords: bool = True,
    ) -> None:
        """Build the index from documents, replacing any existing index data."""
        self.postings.clear()
        self.document_lengths.clear()
        self.document_frequencies.clear()
        self.document_count = 0
        self.average_document_length = 0.0

        total_document_length = 0

        for document in documents:
            document_id = cast(str, document["document_id"])
            text = cast(str, document["text"])
            tokens = preprocess_text(text, normalization, remove_stopwords)
            term_frequencies = Counter(tokens)

            document_length = len(tokens)
            self.document_lengths[document_id] = document_length
            total_document_length += document_length

            for term, frequency in term_frequencies.items():
                self.postings[term][document_id] = frequency
                self.document_frequencies[term] += 1

        self.document_count = len(documents)
        if self.document_count:
            self.average_document_length = (
                total_document_length / self.document_count
            )

    def get_postings(self, term: str) -> dict[str, int]:
        """Return postings for an exact normalized term, or an empty mapping."""
        return self.postings.get(term, {})
