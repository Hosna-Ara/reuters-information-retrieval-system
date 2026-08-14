"""Streamlit interface for the Reuters information retrieval system."""

import time

import streamlit as st

from src.dataset import load_reuters_documents
from src.indexing import InvertedIndex
from src.query_expansion import QueryExpander
from src.query_processor import QueryProcessor
from src.search_engine import SearchEngine
from src.spell_correction import SpellCorrector


st.set_page_config(page_title="Reuters Intelligent Search", layout="wide")


@st.cache_resource
def load_search_resources() -> tuple[
    list[dict[str, object]], InvertedIndex, SpellCorrector, QueryExpander
]:
    """Load the corpus and build reusable retrieval resources once."""
    documents = load_reuters_documents()
    index = InvertedIndex()
    index.build(
        documents,
        normalization="lemmatize",
        remove_stopwords=True,
    )
    spell_corrector = SpellCorrector(index)
    query_expander = QueryExpander(index)
    return documents, index, spell_corrector, query_expander


def show_sidebar(document_count: int) -> None:
    """Display a concise overview of the retrieval system."""
    with st.sidebar:
        st.subheader("System Information")
        st.markdown("**Dataset:** Reuters-21578 / NLTK Reuters Corpus")
        st.markdown(f"**Documents:** {document_count:,}")
        st.markdown("**Ranking:** TF-IDF + BM25")
        st.markdown(
            "**Advanced retrieval:** spelling correction + WordNet query expansion"
        )


def show_results(results: list[dict[str, object]]) -> None:
    """Render ranked search results without altering their stored raw text."""
    if not results:
        st.info("No matching Reuters documents were found. Try different search terms.")
        return

    st.subheader(f"Search Results ({len(results)})")
    for result in results:
        raw_text = str(result["text"])
        preview_text = " ".join(raw_text.split())
        if len(preview_text) > 350:
            preview_text = f"{preview_text[:350].rstrip()}…"

        categories = result["categories"]
        category_text = ", ".join(str(category) for category in categories)
        if not category_text:
            category_text = "None"

        with st.container(border=True):
            st.markdown(
                f"### {result['rank']}. Document `{result['document_id']}`"
            )
            score_column, categories_column, split_column = st.columns([1, 3, 1])
            score_column.metric("Relevance score", f"{float(result['score']):.4f}")
            categories_column.markdown(f"**Categories:** {category_text}")
            split_column.markdown(f"**Dataset split:** {result['split']}")
            st.write(preview_text)
            with st.expander("View full document"):
                st.text(raw_text)


def main() -> None:
    """Run the Streamlit application."""
    st.title("Reuters Intelligent Information Retrieval System")
    st.caption(
        "Search the Reuters corpus using TF-IDF or BM25, with optional spelling "
        "correction and query expansion."
    )

    try:
        documents, index, spell_corrector, query_expander = load_search_resources()
    except LookupError:
        st.error(
            "Required NLTK resources are missing. Run the following command, then "
            "restart the application:"
        )
        st.code("python scripts/setup_nltk.py", language="bash")
        return
    except Exception as error:
        st.error(f"Unable to initialize the search system: {error}")
        return

    show_sidebar(len(documents))

    with st.form("reuters_search_form"):
        query = st.text_input("Search Reuters")
        controls = st.columns(4)
        with controls[0]:
            ranking_method = st.selectbox("Ranking method", ["BM25", "TF-IDF"])
        with controls[1]:
            top_k = st.selectbox("Top results", [5, 10, 20], index=1)
        with controls[2]:
            spelling_enabled = st.checkbox("Spelling correction", value=True)
        with controls[3]:
            expansion_enabled = st.checkbox("Query expansion", value=False)
        submitted = st.form_submit_button("Search", type="primary")

    if not submitted:
        return
    if not query.strip():
        st.warning("Enter a search query before searching.")
        return

    try:
        query_processor = QueryProcessor(
            normalization="lemmatize",
            remove_stopwords=True,
            spell_corrector=spell_corrector,
            enable_spelling_correction=spelling_enabled,
            query_expander=query_expander,
            enable_query_expansion=expansion_enabled,
        )
        search_engine = SearchEngine(documents, index, query_processor)
        ranking_key = {"BM25": "bm25", "TF-IDF": "tfidf"}[ranking_method]

        search_started = time.perf_counter()
        results = search_engine.search(query, method=ranking_key, top_k=top_k)
        search_time_ms = (time.perf_counter() - search_started) * 1_000

        processed_tokens = query_processor.process(query)
    except LookupError:
        st.error(
            "Required NLTK resources are missing. Run the following command, then "
            "try again:"
        )
        st.code("python scripts/setup_nltk.py", language="bash")
        return
    except Exception as error:
        st.error(f"The search could not be completed: {error}")
        return

    st.subheader("Processed Query")
    query_column, tokens_column, timing_column = st.columns([2, 3, 1])
    query_column.markdown(f"**Original query:** {query}")
    tokens_column.markdown(
        f"**Final processed query tokens:** `{processed_tokens}`"
    )
    timing_column.metric("Search time", f"{search_time_ms:.2f} ms")

    show_results(results)


if __name__ == "__main__":
    main()
