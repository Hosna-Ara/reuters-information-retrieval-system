"""Corpus-aware spelling correction for preprocessed query tokens."""

from nltk.metrics.distance import edit_distance

from src.indexing import InvertedIndex


class SpellCorrector:
    """Correct tokens using indexed vocabulary and corpus frequencies."""

    def __init__(
        self,
        index: InvertedIndex,
        max_distance: int = 2,
        min_token_length: int = 3,
        min_corpus_frequency: int = 2,
    ) -> None:
        """Initialize correction settings and cache corpus term frequencies."""
        if max_distance < 1:
            raise ValueError("max_distance must be at least 1")
        if min_token_length < 1:
            raise ValueError("min_token_length must be at least 1")
        if min_corpus_frequency < 1:
            raise ValueError("min_corpus_frequency must be at least 1")

        self.index = index
        self.max_distance = max_distance
        self.min_token_length = min_token_length
        self.min_corpus_frequency = min_corpus_frequency
        self.term_frequencies: dict[str, int] = {
            term: sum(postings.values())
            for term, postings in index.postings.items()
        }

    def correct_token(self, token: str) -> str:
        """Return the best indexed spelling candidate for a token."""
        if len(token) < self.min_token_length:
            return token
        if self.term_frequencies.get(token, 0) >= self.min_corpus_frequency:
            return token

        candidates: list[tuple[int, int, str]] = []
        for candidate in self.index.postings:
            if self.term_frequencies[candidate] < self.min_corpus_frequency:
                continue
            if abs(len(candidate) - len(token)) > self.max_distance:
                continue

            distance = edit_distance(token, candidate)
            if distance <= self.max_distance:
                candidates.append(
                    (distance, -self.term_frequencies[candidate], candidate)
                )

        if not candidates:
            return token

        return min(candidates)[2]

    def correct_tokens(self, tokens: list[str]) -> list[str]:
        """Return a new list containing corrections for all supplied tokens."""
        return [self.correct_token(token) for token in tokens]
