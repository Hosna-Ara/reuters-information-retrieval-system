"""Query processing for the Reuters information retrieval system."""

from src.preprocessing import preprocess_text
from src.query_expansion import QueryExpander
from src.spell_correction import SpellCorrector


class QueryProcessor:
    """Convert raw user queries into normalized tokens."""

    def __init__(
        self,
        normalization: str = "lemmatize",
        remove_stopwords: bool = True,
        spell_corrector: SpellCorrector | None = None,
        enable_spelling_correction: bool = False,
        query_expander: QueryExpander | None = None,
        enable_query_expansion: bool = False,
    ) -> None:
        """Configure preprocessing, spelling correction, and query expansion."""
        if enable_spelling_correction and spell_corrector is None:
            raise ValueError(
                "spell_corrector is required when spelling correction is enabled"
            )
        if enable_query_expansion and query_expander is None:
            raise ValueError(
                "query_expander is required when query expansion is enabled"
            )

        self.normalization = normalization
        self.remove_stopwords = remove_stopwords
        self.spell_corrector = spell_corrector
        self.enable_spelling_correction = enable_spelling_correction
        self.query_expander = query_expander
        self.enable_query_expansion = enable_query_expansion

    def process(self, query: str) -> list[str]:
        """Process a raw query into tokens suitable for retrieval."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()
        if not query:
            return []

        tokens = preprocess_text(
            query,
            normalization=self.normalization,
            remove_stopwords=self.remove_stopwords,
        )

        if self.enable_spelling_correction:
            assert self.spell_corrector is not None
            tokens = self.spell_corrector.correct_tokens(tokens)

        if self.enable_query_expansion:
            assert self.query_expander is not None
            tokens = self.query_expander.expand_tokens(tokens)

        return tokens
