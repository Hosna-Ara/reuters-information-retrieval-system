"""TF-IDF cosine-similarity ranking for the Reuters IR system."""

import math
from collections import Counter, defaultdict

from src.indexing import InvertedIndex


class TfidfRanker:
    """Rank indexed documents using TF-IDF cosine similarity."""

    def __init__(self, index: InvertedIndex) -> None:
        """Store the index and precompute each document's TF-IDF norm."""
        self.index = index
        squared_norms: defaultdict[str, float] = defaultdict(float)

        for term, postings in self.index.postings.items():
            term_idf = self.idf(term)
            for document_id, term_frequency in postings.items():
                weight = term_frequency * term_idf
                squared_norms[document_id] += weight * weight

        self.document_norms: dict[str, float] = {
            document_id: math.sqrt(squared_norms[document_id])
            for document_id in self.index.document_lengths
        }

    def idf(self, term: str) -> float:
        """Return the smoothed inverse document frequency of a term."""
        document_frequency = self.index.document_frequencies.get(term, 0)
        return math.log(
            (self.index.document_count + 1) / (document_frequency + 1)
        ) + 1.0

    def score(self, query_tokens: list[str]) -> dict[str, float]:
        """Return non-zero cosine-similarity scores for a tokenized query."""
        if not query_tokens:
            return {}

        query_frequencies = Counter(
            term for term in query_tokens if term in self.index.postings
        )
        if not query_frequencies:
            return {}

        query_weights = {
            term: frequency * self.idf(term)
            for term, frequency in query_frequencies.items()
        }
        query_norm = math.sqrt(
            sum(weight * weight for weight in query_weights.values())
        )

        dot_products: defaultdict[str, float] = defaultdict(float)
        for term, query_weight in query_weights.items():
            term_idf = self.idf(term)
            for document_id, term_frequency in self.index.get_postings(term).items():
                document_weight = term_frequency * term_idf
                dot_products[document_id] += query_weight * document_weight

        scores: dict[str, float] = {}
        for document_id, dot_product in dot_products.items():
            document_norm = self.document_norms[document_id]
            if dot_product and document_norm:
                scores[document_id] = dot_product / (query_norm * document_norm)

        return scores

    def rank(
        self,
        query_tokens: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the highest-scoring documents for a tokenized query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        scores = self.score(query_tokens)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
