"""Query processing for the Reuters information retrieval system."""

from src.preprocessing import preprocess_text


class QueryProcessor:
    """Convert raw user queries into normalized tokens."""

    def __init__(
        self,
        normalization: str = "lemmatize",
        remove_stopwords: bool = True,
    ) -> None:
        """Configure query normalization and stopword removal."""
        self.normalization = normalization
        self.remove_stopwords = remove_stopwords

    def process(self, query: str) -> list[str]:
        """Process a raw query into tokens suitable for retrieval."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()
        if not query:
            return []

        return preprocess_text(
            query,
            normalization=self.normalization,
            remove_stopwords=self.remove_stopwords,
        )
