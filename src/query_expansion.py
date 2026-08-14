"""Conservative corpus-aware WordNet query expansion."""

from dataclasses import dataclass

from nltk.corpus import wordnet

from src.indexing import InvertedIndex


@dataclass(frozen=True)
class _Candidate:
    """Store a synonym candidate and its ranking evidence."""

    term: str
    jaccard: float
    shared_documents: int
    matched_context_terms: int
    context_shared_documents: int
    corpus_frequency: int


class QueryExpander:
    """Expand preprocessed query tokens with corpus-supported synonyms."""

    def __init__(
        self,
        index: InvertedIndex,
        min_corpus_frequency: int = 2,
        min_shared_documents: int = 5,
        min_jaccard: float = 0.05,
        min_context_terms: int = 1,
        max_expansions: int = 1,
    ) -> None:
        """Configure expansion thresholds and cache corpus term frequencies."""
        if min_corpus_frequency < 1:
            raise ValueError("min_corpus_frequency must be at least 1")
        if min_shared_documents < 1:
            raise ValueError("min_shared_documents must be at least 1")
        if not 0 <= min_jaccard <= 1:
            raise ValueError("min_jaccard must be between 0 and 1 inclusive")
        if min_context_terms < 0:
            raise ValueError("min_context_terms must be at least 0")
        if max_expansions < 1:
            raise ValueError("max_expansions must be at least 1")

        self.index = index
        self.min_corpus_frequency = min_corpus_frequency
        self.min_shared_documents = min_shared_documents
        self.min_jaccard = min_jaccard
        self.min_context_terms = min_context_terms
        self.max_expansions = max_expansions
        self.corpus_term_frequencies: dict[str, int] = {
            term: sum(postings.values())
            for term, postings in index.postings.items()
        }

    def _eligible_candidates(
        self, target: str, original_tokens: set[str]
    ) -> list[_Candidate]:
        """Collect corpus-supported WordNet synonyms for one query token."""
        target_documents = set(self.index.postings.get(target, {}))
        candidates = {
            lemma.name().lower()
            for synset in wordnet.synsets(target)
            for lemma in synset.lemmas()
        }
        eligible: list[_Candidate] = []

        for candidate in candidates:
            if (
                candidate == target
                or "_" in candidate
                or not candidate.isalpha()
                or candidate in original_tokens
                or candidate not in self.index.postings
                or self.corpus_term_frequencies[candidate]
                < self.min_corpus_frequency
            ):
                continue

            candidate_documents = set(self.index.postings[candidate])
            shared_documents = len(target_documents & candidate_documents)
            union_size = len(target_documents | candidate_documents)
            jaccard = shared_documents / union_size if union_size else 0.0

            matched_context_terms = 0
            context_shared_documents = 0
            for context_term in original_tokens - {target}:
                context_documents = set(self.index.postings.get(context_term, {}))
                shared_with_context = len(candidate_documents & context_documents)
                if shared_with_context:
                    matched_context_terms += 1
                    context_shared_documents += shared_with_context

            if (
                shared_documents >= self.min_shared_documents
                and jaccard >= self.min_jaccard
                and matched_context_terms >= self.min_context_terms
            ):
                eligible.append(
                    _Candidate(
                        term=candidate,
                        jaccard=jaccard,
                        shared_documents=shared_documents,
                        matched_context_terms=matched_context_terms,
                        context_shared_documents=context_shared_documents,
                        corpus_frequency=self.corpus_term_frequencies[candidate],
                    )
                )

        return eligible

    def expand_tokens(self, tokens: list[str]) -> list[str]:
        """Return original tokens followed by the best eligible synonyms."""
        expanded = list(tokens)
        if not tokens:
            return expanded

        original_tokens = set(tokens)
        candidates = [
            candidate
            for target in tokens
            for candidate in self._eligible_candidates(target, original_tokens)
        ]
        candidates.sort(
            key=lambda candidate: (
                -candidate.jaccard,
                -candidate.shared_documents,
                -candidate.matched_context_terms,
                -candidate.context_shared_documents,
                -candidate.corpus_frequency,
                candidate.term,
            )
        )

        selected: set[str] = set()
        for candidate in candidates:
            if candidate.term in selected:
                continue
            expanded.append(candidate.term)
            selected.add(candidate.term)
            if len(selected) == self.max_expansions:
                break

        return expanded
