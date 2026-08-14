"""Tests for query processing and search orchestration."""

import pytest

from src.indexing import InvertedIndex
from src.query_expansion import QueryExpander
from src.query_processor import QueryProcessor
from src.search_engine import SearchEngine
from src.spell_correction import SpellCorrector


@pytest.fixture
def documents() -> list[dict[str, object]]:
    return [
        {
            "document_id": "doc1",
            "text": "oil prices rise market",
            "categories": ["energy"],
            "split": "training",
        },
        {
            "document_id": "doc2",
            "text": "bank trade finance",
            "categories": ["finance"],
            "split": "test",
        },
        {
            "document_id": "doc3",
            "text": "oil production industry",
            "categories": ["energy"],
            "split": "training",
        },
    ]


@pytest.fixture
def query_processor() -> QueryProcessor:
    return QueryProcessor(normalization="lemmatize", remove_stopwords=True)


@pytest.fixture
def search_engine(
    documents: list[dict[str, object]], query_processor: QueryProcessor
) -> SearchEngine:
    index = InvertedIndex()
    index.build(
        documents,
        normalization="lemmatize",
        remove_stopwords=True,
    )
    return SearchEngine(documents, index, query_processor)


@pytest.fixture
def spelling_index() -> InvertedIndex:
    documents = [
        {
            "document_id": "spell1",
            "text": "interest interest production coffee trade interst zzzz",
        },
        {
            "document_id": "spell2",
            "text": "interest production coffee trade",
        },
    ]
    index = InvertedIndex()
    index.build(
        documents,
        normalization="none",
        remove_stopwords=True,
    )
    return index


@pytest.fixture
def spell_corrector(spelling_index: InvertedIndex) -> SpellCorrector:
    return SpellCorrector(spelling_index)


@pytest.fixture
def query_expansion_index() -> InvertedIndex:
    documents = [
        {"document_id": "exp1", "text": "car auto export freight"},
        {"document_id": "exp2", "text": "car auto export shipping"},
        {"document_id": "exp3", "text": "car automobile market"},
        {"document_id": "exp4", "text": "agriculture wheat harvest"},
    ]
    index = InvertedIndex()
    index.build(
        documents,
        normalization="none",
        remove_stopwords=True,
    )
    return index


@pytest.fixture
def query_expander(query_expansion_index: InvertedIndex) -> QueryExpander:
    return QueryExpander(
        query_expansion_index,
        min_corpus_frequency=1,
        min_shared_documents=1,
        min_jaccard=0.05,
        min_context_terms=1,
        max_expansions=1,
    )


def test_query_processor_normalizes_query(query_processor: QueryProcessor) -> None:
    assert query_processor.process("Oil Prices") == ["oil", "price"]


def test_query_processor_handles_surrounding_whitespace(
    query_processor: QueryProcessor,
) -> None:
    assert query_processor.process("  Oil Prices  ") == ["oil", "price"]


@pytest.mark.parametrize("query", ["", "   "])
def test_query_processor_returns_empty_list_for_empty_query(
    query_processor: QueryProcessor, query: str
) -> None:
    assert query_processor.process(query) == []


def test_query_processor_rejects_non_string(
    query_processor: QueryProcessor,
) -> None:
    with pytest.raises(TypeError):
        query_processor.process(123)  # type: ignore[arg-type]


def test_query_processor_does_not_correct_spelling_when_disabled(
    spell_corrector: SpellCorrector,
) -> None:
    processor = QueryProcessor(
        normalization="none",
        remove_stopwords=True,
        spell_corrector=spell_corrector,
        enable_spelling_correction=False,
    )

    assert processor.process("cofee trade") == ["cofee", "trade"]


def test_query_processor_corrects_spelling_when_enabled(
    spell_corrector: SpellCorrector,
) -> None:
    processor = QueryProcessor(
        normalization="none",
        remove_stopwords=True,
        spell_corrector=spell_corrector,
        enable_spelling_correction=True,
    )

    assert processor.process("cofee trade producion") == [
        "coffee",
        "trade",
        "production",
    ]


def test_query_processor_corrects_rare_reuters_style_typo(
    spell_corrector: SpellCorrector,
) -> None:
    processor = QueryProcessor(
        normalization="none",
        remove_stopwords=True,
        spell_corrector=spell_corrector,
        enable_spelling_correction=True,
    )

    assert processor.process("interst") == ["interest"]


def test_query_processor_spelling_correction_preserves_noise(
    spell_corrector: SpellCorrector,
) -> None:
    processor = QueryProcessor(
        normalization="none",
        remove_stopwords=True,
        spell_corrector=spell_corrector,
        enable_spelling_correction=True,
    )

    assert processor.process("xyzzzz") == ["xyzzzz"]


def test_query_processor_requires_spell_corrector_when_enabled() -> None:
    with pytest.raises(ValueError):
        QueryProcessor(enable_spelling_correction=True)


def test_query_processor_default_does_not_correct_spelling() -> None:
    processor = QueryProcessor(normalization="none", remove_stopwords=True)

    assert processor.process("cofee trade") == ["cofee", "trade"]


def test_query_processor_does_not_expand_query_when_disabled(
    query_expander: QueryExpander,
) -> None:
    processor = QueryProcessor(
        normalization="none",
        remove_stopwords=True,
        query_expander=query_expander,
        enable_query_expansion=False,
    )

    assert processor.process("car export") == ["car", "export"]


def test_query_processor_expands_query_when_enabled(
    query_expander: QueryExpander,
) -> None:
    processor = QueryProcessor(
        normalization="none",
        remove_stopwords=True,
        query_expander=query_expander,
        enable_query_expansion=True,
    )

    assert processor.process("car export") == ["car", "export", "auto"]


def test_query_processor_requires_query_expander_when_enabled() -> None:
    with pytest.raises(ValueError):
        QueryProcessor(enable_query_expansion=True)


def test_query_processor_baseline_is_unchanged_when_advanced_features_disabled(
    spell_corrector: SpellCorrector,
    query_expander: QueryExpander,
) -> None:
    processor = QueryProcessor(
        normalization="none",
        remove_stopwords=True,
        spell_corrector=spell_corrector,
        enable_spelling_correction=False,
        query_expander=query_expander,
        enable_query_expansion=False,
    )

    assert processor.process("cofee car export") == ["cofee", "car", "export"]


def test_query_processor_applies_preprocessing_spelling_then_expansion() -> None:
    calls: list[tuple[str, list[str]]] = []

    class RecordingSpellCorrector:
        def correct_tokens(self, tokens: list[str]) -> list[str]:
            calls.append(("spelling", list(tokens)))
            return ["coffee" if token == "cofee" else token for token in tokens]

    class RecordingQueryExpander:
        def expand_tokens(self, tokens: list[str]) -> list[str]:
            calls.append(("expansion", list(tokens)))
            return [*tokens, "java"]

    processor = QueryProcessor(
        normalization="none",
        remove_stopwords=True,
        spell_corrector=RecordingSpellCorrector(),  # type: ignore[arg-type]
        enable_spelling_correction=True,
        query_expander=RecordingQueryExpander(),  # type: ignore[arg-type]
        enable_query_expansion=True,
    )

    assert processor.process("The COFEE!") == ["coffee", "java"]
    assert calls == [
        ("spelling", ["cofee"]),
        ("expansion", ["coffee"]),
    ]


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_search_ranks_doc1_first(
    search_engine: SearchEngine, method: str
) -> None:
    results = search_engine.search("oil prices", method=method)

    assert results
    assert results[0]["document_id"] == "doc1"


def test_search_result_has_expected_fields_and_raw_text(
    search_engine: SearchEngine,
) -> None:
    result = search_engine.search("oil prices", method="tfidf")[0]

    assert set(result) == {
        "rank",
        "document_id",
        "score",
        "categories",
        "split",
        "text",
    }
    assert result["text"] == "oil prices rise market"


def test_search_respects_top_k(search_engine: SearchEngine) -> None:
    assert len(search_engine.search("oil", top_k=1)) == 1


def test_search_returns_empty_list_for_empty_query(
    search_engine: SearchEngine,
) -> None:
    assert search_engine.search("   ") == []


def test_search_rejects_unsupported_method(search_engine: SearchEngine) -> None:
    with pytest.raises(ValueError):
        search_engine.search("oil", method="unsupported")


def test_search_returns_empty_list_for_out_of_vocabulary_query(
    search_engine: SearchEngine,
) -> None:
    assert search_engine.search("quasar") == []


def test_spell_corrector_corrects_rare_in_vocabulary_typo(
    spell_corrector: SpellCorrector,
) -> None:
    assert spell_corrector.correct_token("interst") == "interest"


def test_spell_corrector_corrects_standard_misspelling(
    spell_corrector: SpellCorrector,
) -> None:
    assert spell_corrector.correct_token("producion") == "production"


def test_spell_corrector_corrects_another_misspelling(
    spell_corrector: SpellCorrector,
) -> None:
    assert spell_corrector.correct_token("cofee") == "coffee"


def test_spell_corrector_preserves_trusted_known_word(
    spell_corrector: SpellCorrector,
) -> None:
    assert spell_corrector.correct_token("trade") == "trade"


def test_spell_corrector_does_not_correct_noise_to_rare_word(
    spell_corrector: SpellCorrector,
) -> None:
    corrected = spell_corrector.correct_token("xyzzzz")

    assert corrected == "xyzzzz"
    assert corrected != "zzzz"


def test_spell_corrector_correct_tokens_preserves_order_and_input(
    spell_corrector: SpellCorrector,
) -> None:
    tokens = ["cofee", "trade", "producion", "interst"]
    original_tokens = tokens.copy()

    corrected = spell_corrector.correct_tokens(tokens)

    assert corrected == ["coffee", "trade", "production", "interest"]
    assert tokens == original_tokens


def test_query_expander_selects_context_supported_expansion(
    query_expander: QueryExpander,
) -> None:
    assert query_expander.expand_tokens(["car", "export"]) == [
        "car",
        "export",
        "auto",
    ]


def test_query_expander_preserves_original_token_order(
    query_expander: QueryExpander,
) -> None:
    tokens = ["export", "car"]

    assert query_expander.expand_tokens(tokens)[: len(tokens)] == tokens


def test_query_expander_does_not_modify_input(
    query_expander: QueryExpander,
) -> None:
    tokens = ["car", "export"]
    original_tokens = tokens.copy()

    query_expander.expand_tokens(tokens)

    assert tokens == original_tokens


def test_query_expander_returns_empty_list_for_empty_input(
    query_expander: QueryExpander,
) -> None:
    assert query_expander.expand_tokens([]) == []


def test_query_expander_returns_copy_when_no_synonym_is_eligible(
    query_expander: QueryExpander,
) -> None:
    tokens = ["quasar"]

    expanded = query_expander.expand_tokens(tokens)

    assert expanded == tokens
    assert expanded is not tokens


def test_query_expander_respects_maximum_expansion_limit(
    query_expander: QueryExpander,
) -> None:
    tokens = ["car", "export"]

    expanded = query_expander.expand_tokens(tokens)

    assert len(expanded) - len(tokens) <= 1


def test_query_expander_does_not_append_duplicate_expansions(
    query_expander: QueryExpander,
) -> None:
    expanded = query_expander.expand_tokens(["car", "car", "export"])

    appended_terms = expanded[3:]
    assert appended_terms == ["auto"]
    assert len(appended_terms) == len(set(appended_terms))


@pytest.mark.parametrize(
    "invalid_parameter",
    [
        {"min_corpus_frequency": 0},
        {"min_shared_documents": 0},
        {"min_jaccard": -0.01},
        {"min_jaccard": 1.01},
        {"min_context_terms": -1},
        {"max_expansions": 0},
    ],
)
def test_query_expander_rejects_invalid_parameters(
    query_expansion_index: InvertedIndex,
    invalid_parameter: dict[str, int | float],
) -> None:
    with pytest.raises(ValueError):
        QueryExpander(query_expansion_index, **invalid_parameter)


def test_spell_corrector_preserves_short_token(
    spell_corrector: SpellCorrector,
) -> None:
    assert spell_corrector.correct_token("an") == "an"


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_distance", 0),
        ("min_token_length", 0),
        ("min_corpus_frequency", 0),
    ],
)
def test_spell_corrector_rejects_invalid_parameters(
    spelling_index: InvertedIndex, parameter: str, value: int
) -> None:
    with pytest.raises(ValueError):
        SpellCorrector(spelling_index, **{parameter: value})
