"""Run reproducible category-based evaluation of the Reuters IR system."""

import csv
import time
from pathlib import Path
from statistics import fmean
from typing import cast

from evaluation.metrics import (
    average_precision,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.dataset import load_reuters_documents
from src.indexing import InvertedIndex
from src.query_expansion import QueryExpander
from src.query_processor import QueryProcessor
from src.search_engine import SearchEngine
from src.spell_correction import SpellCorrector


EVALUATION_CONFIGURATIONS: dict[str, dict[str, object]] = {
    "tfidf": {"method": "tfidf", "spelling": False, "expansion": False},
    "bm25": {"method": "bm25", "spelling": False, "expansion": False},
    "bm25_spelling": {
        "method": "bm25",
        "spelling": True,
        "expansion": False,
    },
    "bm25_expansion": {
        "method": "bm25",
        "spelling": False,
        "expansion": True,
    },
    "bm25_spelling_expansion": {
        "method": "bm25",
        "spelling": True,
        "expansion": True,
    },
}

_REQUIRED_QUERY_COLUMNS = {"query_id", "query_text", "target_category"}
_METRIC_COLUMNS = (
    "precision_at_5",
    "precision_at_10",
    "recall_at_10",
    "f1_at_10",
    "average_precision",
    "reciprocal_rank",
    "ndcg_at_10",
    "latency_ms",
)
_RESULT_COLUMNS = (
    "configuration",
    "query_id",
    "query_text",
    "target_category",
    "relevant_document_count",
    "retrieved_document_count",
    *_METRIC_COLUMNS,
)


def load_evaluation_queries(path: str) -> list[dict[str, str]]:
    """Load evaluation queries from a CSV file with the required columns."""
    with open(path, newline="", encoding="utf-8") as query_file:
        reader = csv.DictReader(query_file)
        columns = set(reader.fieldnames or [])
        missing = _REQUIRED_QUERY_COLUMNS - columns
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"Missing required query columns: {missing_names}")

        return [
            {column: row[column] for column in reader.fieldnames or []}
            for row in reader
        ]


def build_relevance_judgments(
    documents: list[dict[str, object]],
    queries: list[dict[str, str]],
) -> dict[str, set[str]]:
    """Map each query ID to documents carrying its target category label."""
    judgments: dict[str, set[str]] = {}
    for query in queries:
        target_category = query["target_category"]
        judgments[query["query_id"]] = {
            cast(str, document["document_id"])
            for document in documents
            if target_category in cast(list[str], document["categories"])
        }
    return judgments


def evaluate_system(
    queries_path: str = "evaluation/queries.csv",
    results_path: str = "results/evaluation_results.csv",
    top_k: int = 10,
    retrieval_depth: int = 100,
) -> list[dict[str, object]]:
    """Evaluate all retrieval configurations and write query and mean results."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if retrieval_depth <= 0:
        raise ValueError("retrieval_depth must be greater than 0")

    queries = load_evaluation_queries(queries_path)
    documents = load_reuters_documents()

    index = InvertedIndex()
    index.build(documents, normalization="lemmatize", remove_stopwords=True)
    spell_corrector = SpellCorrector(index)
    query_expander = QueryExpander(index)
    judgments = build_relevance_judgments(documents, queries)

    rows: list[dict[str, object]] = []
    for configuration, settings in EVALUATION_CONFIGURATIONS.items():
        configuration_rows: list[dict[str, object]] = []
        for query in queries:
            query_processor = QueryProcessor(
                normalization="lemmatize",
                remove_stopwords=True,
                spell_corrector=spell_corrector,
                enable_spelling_correction=cast(bool, settings["spelling"]),
                query_expander=query_expander,
                enable_query_expansion=cast(bool, settings["expansion"]),
            )
            search_engine = SearchEngine(documents, index, query_processor)
            start_time = time.perf_counter()
            results = search_engine.search(
                query["query_text"],
                method=cast(str, settings["method"]),
                top_k=retrieval_depth,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            retrieved = [cast(str, result["document_id"]) for result in results]
            relevant = judgments[query["query_id"]]
            row: dict[str, object] = {
                "configuration": configuration,
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "target_category": query["target_category"],
                "relevant_document_count": len(relevant),
                "retrieved_document_count": len(retrieved),
                "precision_at_5": precision_at_k(retrieved, relevant, 5),
                "precision_at_10": precision_at_k(retrieved, relevant, 10),
                "recall_at_10": recall_at_k(retrieved, relevant, 10),
                "f1_at_10": f1_at_k(retrieved, relevant, 10),
                "average_precision": average_precision(retrieved, relevant),
                "reciprocal_rank": reciprocal_rank(retrieved, relevant),
                "ndcg_at_10": ndcg_at_k(retrieved, relevant, 10),
                "latency_ms": latency_ms,
            }
            configuration_rows.append(row)
            rows.append(row)

        mean_row: dict[str, object] = {
            "configuration": configuration,
            "query_id": "MEAN",
            "query_text": "",
            "target_category": "",
            "relevant_document_count": "",
            "retrieved_document_count": "",
        }
        for metric in _METRIC_COLUMNS:
            mean_row[metric] = (
                fmean(cast(float, row[metric]) for row in configuration_rows)
                if configuration_rows
                else 0.0
            )
        rows.append(mean_row)

    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Evaluated {len(queries)} queries across "
        f"{len(EVALUATION_CONFIGURATIONS)} configurations; output: {results_path}"
    )
    return rows


if __name__ == "__main__":
    evaluate_system()
