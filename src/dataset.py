"""Load documents from the NLTK Reuters corpus."""

from typing import Dict, List, Optional

from nltk.corpus import reuters


def load_reuters_documents(
    split: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Return Reuters documents, optionally filtered by dataset split."""
    if split not in (None, "training", "test"):
        raise ValueError("split must be None, 'training', or 'test'")

    file_ids = reuters.fileids()
    if split is not None:
        prefix = f"{split}/"
        file_ids = [file_id for file_id in file_ids if file_id.startswith(prefix)]

    return [
        {
            "document_id": file_id,
            "text": reuters.raw(file_id),
            "categories": reuters.categories(file_id),
            "split": "training" if file_id.startswith("training/") else "test",
        }
        for file_id in file_ids
    ]
