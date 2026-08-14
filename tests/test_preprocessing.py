import pytest

from src.preprocessing import preprocess_text


def test_no_normalization_removes_stopwords():
    assert preprocess_text(
        "The companies were producing cars.", normalization="none"
    ) == ["companies", "producing", "cars"]


def test_stemming_normalizes_words():
    assert preprocess_text("companies producing cars", normalization="stem") == [
        "compani",
        "produc",
        "car",
    ]


def test_lemmatization_normalizes_plural_nouns():
    assert preprocess_text("companies cars", normalization="lemmatize") == [
        "company",
        "car",
    ]


def test_stopwords_can_be_retained():
    tokens = preprocess_text(
        "The companies were producing cars.",
        normalization="none",
        remove_stopwords=False,
    )

    assert "the" in tokens
    assert "were" in tokens


def test_non_alphabetic_tokens_are_removed():
    assert preprocess_text(
        "Cars, 123 and trucks!", normalization="none", remove_stopwords=False
    ) == ["cars", "and", "trucks"]


def test_invalid_normalization_raises_value_error():
    with pytest.raises(ValueError):
        preprocess_text("cars", normalization="unsupported")
