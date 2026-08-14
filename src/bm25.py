"""BM25 ranking for the Reuters IR system."""

from collections import Counter
from math import log

from src.indexing import InvertedIndex


class BM25Ranker:
    """Rank preprocessed queries against an inverted index using BM25."""

    def __init__(
        self,
        index: InvertedIndex,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize the ranker with an index and BM25 parameters."""
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1 inclusive")

        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term: str) -> float:
        """Return the BM25 inverse document frequency for a term."""
        document_frequency = self.index.document_frequencies.get(term, 0)
        return log(
            1
            + (
                (self.index.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
        )

    def score(self, query_tokens: list[str]) -> dict[str, float]:
        """Return BM25 scores for documents matching a tokenized query."""
        if not query_tokens or self.index.document_count == 0:
            return {}

        query_frequencies = Counter(
            term for term in query_tokens if term in self.index.postings
        )
        if not query_frequencies or self.index.average_document_length == 0:
            return {}

        scores: dict[str, float] = {}
        average_length = self.index.average_document_length

        for term, query_frequency in query_frequencies.items():
            inverse_document_frequency = self.idf(term)
            for document_id, term_frequency in self.index.get_postings(term).items():
                document_length = self.index.document_lengths[document_id]
                normalization = self.k1 * (
                    1 - self.b + self.b * document_length / average_length
                )
                contribution = query_frequency * inverse_document_frequency * (
                    term_frequency * (self.k1 + 1)
                    / (term_frequency + normalization)
                )
                scores[document_id] = scores.get(document_id, 0.0) + contribution

        return {document_id: score for document_id, score in scores.items() if score}

    def rank(
        self,
        query_tokens: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the highest-scoring documents in deterministic order."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        scores = self.score(query_tokens)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
