"""
Dataset Loader for Model Benchmarking (Level 5).

Loads train, validation, and test splits cleanly using group_split_manifest.csv
and raw email files from data/raw/spamassassin/.

Guarantees strict isolation of validation and test sets.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

from src.data.parser import parse_raw_email
from src.data.verify_split import verify_and_report

RAW_DIR = Path("data/raw/spamassassin")
INTERIM_DIR = Path("data/interim")
MANIFEST_PATH = INTERIM_DIR / "group_split_manifest.csv"


def ensure_split_manifest() -> pd.DataFrame:
    """Ensure group_split_manifest.csv exists and is loaded."""
    if not MANIFEST_PATH.exists():
        print("[>] group_split_manifest.csv missing. Running verify_and_report()...")
        split_df, _, _ = verify_and_report()
        return split_df
    return pd.read_csv(MANIFEST_PATH)


def load_email_records_for_split(
    split_name: str,
    manifest_df: Optional[pd.DataFrame] = None
) -> Tuple[List[Dict[str, Any]], np.ndarray, pd.DataFrame]:
    """
    Load parsed email records, labels (y), and metadata DataFrame for a specified split.

    Args:
        split_name: One of 'train', 'val', or 'test'.
        manifest_df: Optional pre-loaded manifest DataFrame.

    Returns:
        Tuple of (records_list, y_labels_array, split_metadata_df)
    """
    if manifest_df is None:
        manifest_df = ensure_split_manifest()

    if split_name not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'.")

    split_df = manifest_df[manifest_df["split"] == split_name].copy().reset_index(drop=True)

    records = []
    labels = []

    for _, row in split_df.iterrows():
        rel_path = str(row["file_path"]).replace("\\", "/")
        full_path = RAW_DIR / rel_path

        if not full_path.exists():
            # Fallback for relative path matching
            parts = rel_path.split("/")
            if len(parts) >= 2:
                full_path = RAW_DIR / parts[-2] / parts[-1]

        if not full_path.exists():
            raise FileNotFoundError(f"Raw email missing: {full_path}")

        with open(full_path, "rb") as f:
            raw_bytes = f.read()

        parsed = parse_raw_email(raw_bytes)

        record = {
            "file_path": rel_path,
            "message_id": row["message_id"],
            "subject": parsed.get("subject", "") or "",
            "plain_body": parsed.get("plain_body", "") or "",
            "html_body": parsed.get("html_body", "") or "",
            "has_html": parsed.get("has_html", False),
            "urls": parsed.get("urls", []) or [],
            "url_count": parsed.get("url_count", 0),
            "attachment_count": parsed.get("attachment_count", 0),
            "is_multipart": parsed.get("is_multipart", False),
            "group_id": row["group_id"],
            "label": int(row["label"])
        }

        records.append(record)
        labels.append(int(row["label"]))

    return records, np.array(labels, dtype=int), split_df


def load_all_splits() -> Dict[str, Tuple[List[Dict[str, Any]], np.ndarray, pd.DataFrame]]:
    """Load train, val, and test splits into a structured dictionary."""
    manifest_df = ensure_split_manifest()
    return {
        "train": load_email_records_for_split("train", manifest_df),
        "val": load_email_records_for_split("val", manifest_df),
        "test": load_email_records_for_split("test", manifest_df)
    }
