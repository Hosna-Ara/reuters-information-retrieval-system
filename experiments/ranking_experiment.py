"""Create summary artifacts from the existing ranking evaluation results."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "results" / "evaluation_results.csv"
TABLE_FILE = PROJECT_ROOT / "results" / "tables" / "configuration_summary.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

SUMMARY_COLUMNS = [
    "configuration",
    "precision_at_5",
    "precision_at_10",
    "recall_at_10",
    "f1_at_10",
    "average_precision",
    "reciprocal_rank",
    "ndcg_at_10",
    "latency_ms",
]

DISPLAY_LABELS = {
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "bm25_spelling": "BM25 + Spelling",
    "bm25_expansion": "BM25 + Expansion",
    "bm25_spelling_expansion": "BM25 + Spelling + Expansion",
}


def read_mean_results() -> list[dict[str, str]]:
    """Read configuration-level mean metrics from the saved evaluation CSV."""
    with RESULTS_FILE.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing_columns = set(SUMMARY_COLUMNS + ["query_id"]) - set(
            reader.fieldnames or []
        )
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Evaluation results are missing columns: {missing}")
        rows = [row for row in reader if row["query_id"] == "MEAN"]

    if not rows:
        raise ValueError("Evaluation results contain no rows with query_id == 'MEAN'.")
    return rows


def write_summary_table(rows: list[dict[str, str]]) -> None:
    """Write the requested concise configuration summary."""
    TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_FILE.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: row[column] for column in SUMMARY_COLUMNS} for row in rows
        )


def create_bar_plot(
    rows: list[dict[str, str]], metric: str, title: str, output_file: Path
) -> None:
    """Create and save one labelled metric comparison bar chart."""
    labels = [DISPLAY_LABELS.get(row["configuration"], row["configuration"]) for row in rows]
    values = [float(row[metric]) for row in rows]

    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(labels, values, color="#35618f")
    axis.set_title(title)
    axis.set_ylabel("Score")
    axis.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    axis.tick_params(axis="x", labelrotation=25)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_file, dpi=300)
    plt.close(figure)


def main() -> None:
    rows = read_mean_results()
    write_summary_table(rows)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plots = [
        ("precision_at_10", "Precision@10 by Configuration", FIGURES_DIR / "precision_at_10.png"),
        ("average_precision", "Mean Average Precision by Configuration", FIGURES_DIR / "map_comparison.png"),
        ("ndcg_at_10", "nDCG@10 by Configuration", FIGURES_DIR / "ndcg_at_10.png"),
    ]
    for metric, title, output_file in plots:
        create_bar_plot(rows, metric, title, output_file)

    best_metrics = [
        ("Precision@10", "precision_at_10"),
        ("MAP", "average_precision"),
        ("nDCG@10", "ndcg_at_10"),
    ]
    for display_metric, metric in best_metrics:
        best = max(rows, key=lambda row: float(row[metric]))
        label = DISPLAY_LABELS.get(best["configuration"], best["configuration"])
        print(f"Best by {display_metric}: {label} ({float(best[metric]):.3f})")

    print("Generated files:")
    for path in [TABLE_FILE, *(output_file for _, _, output_file in plots)]:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
