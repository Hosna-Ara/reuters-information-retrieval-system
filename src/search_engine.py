"""Search orchestration for the Reuters information retrieval system."""

from typing import cast

from src.bm25 import BM25Ranker
from src.indexing import InvertedIndex
from src.query_processor import QueryProcessor
from src.tfidf import TfidfRanker


class SearchEngine:
    """Process queries, rank indexed documents, and format search results."""

    def __init__(
        self,
        documents: list[dict[str, object]],
        index: InvertedIndex,
        query_processor: QueryProcessor,
    ) -> None:
        """Initialize the engine with documents and prebuilt retrieval components."""
        self.documents = documents
        self.index = index
        self.query_processor = query_processor
        self.document_lookup: dict[str, dict[str, object]] = {
            cast(str, document["document_id"]): document for document in documents
        }
        self.tfidf_ranker = TfidfRanker(index)
        self.bm25_ranker = BM25Ranker(index)

    def search(
        self,
        query: str,
        method: str = "bm25",
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        """Return ranked Reuters documents for a raw query."""
        rankers = {
            "tfidf": self.tfidf_ranker,
            "bm25": self.bm25_ranker,
        }
        if method not in rankers:
            raise ValueError(
                f"Unsupported ranking method {method!r}; expected 'tfidf' or 'bm25'"
            )

        query_tokens = self.query_processor.process(query)
        if not query_tokens:
            return []

        ranked_documents = rankers[method].rank(query_tokens, top_k)
        results: list[dict[str, object]] = []
        for rank, (document_id, score) in enumerate(ranked_documents, start=1):
            document = self.document_lookup[document_id]
            results.append(
                {
                    "rank": rank,
                    "document_id": document_id,
                    "score": float(score),
                    "categories": document["categories"],
                    "split": document["split"],
                    "text": document["text"],
                }
            )

        return results
