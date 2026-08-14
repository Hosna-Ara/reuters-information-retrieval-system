"""Text preprocessing utilities for the Reuters IR system."""

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


ENGLISH_STOPWORDS: set[str] = set(stopwords.words("english"))
PORTER_STEMMER = PorterStemmer()
WORDNET_LEMMATIZER = WordNetLemmatizer()


def preprocess_text(
    text: str,
    normalization: str = "lemmatize",
    remove_stopwords: bool = True,
) -> list[str]:
    """Tokenize, filter, and normalize text into alphabetic terms."""
    if normalization not in {"none", "stem", "lemmatize"}:
        raise ValueError(
            "Unsupported normalization mode "
            f"{normalization!r}; expected 'none', 'stem', or 'lemmatize'."
        )

    tokens = [token.lower() for token in word_tokenize(text)]
    tokens = [token for token in tokens if token.isalpha()]

    if remove_stopwords:
        tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]

    if normalization == "stem":
        return [PORTER_STEMMER.stem(token) for token in tokens]
    if normalization == "lemmatize":
        return [WORDNET_LEMMATIZER.lemmatize(token) for token in tokens]
    return tokens
