"""
Dataset Inventory Script for Apache SpamAssassin Public Corpus.

Scans all raw files in data/raw/spamassassin/, parses email metadata using parser.py,
inspects X-Spam-* leakage headers, MIME types, sizes, URLs, encodings, and
outputs an inventory dataset into data/interim/corpus_inventory.csv.
"""

import os
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

# Import parser
from src.data.parser import parse_raw_email

RAW_DIR = Path("data/raw/spamassassin")
INTERIM_DIR = Path("data/interim")

# Mapping raw folder names to ground truth labels
FOLDER_LABEL_MAP = {
    "easy_ham": 0,
    "easy_ham_2": 0,
    "hard_ham": 0,
    "spam": 1,
    "spam_2": 1
}


def scan_and_inventory() -> pd.DataFrame:
    """Scan raw directory, parse email metadata, build inventory dataframe."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory {RAW_DIR} does not exist. Run acquire.py first.")

    subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    print(f"[>] Inventorying corpus across {len(subdirs)} subdirectories...")

    for folder in subdirs:
        folder_name = folder.name
        if folder_name not in FOLDER_LABEL_MAP:
            print(f"[!] Skipping unrecognized directory: {folder_name}")
            continue

        label = FOLDER_LABEL_MAP[folder_name]
        files = [f for f in folder.iterdir() if f.is_file() and f.name != "cmds"]
        print(f"    Scanning '{folder_name}' ({len(files)} files, label={label})...")

        for file_path in files:
            try:
                with open(file_path, "rb") as f:
                    raw_bytes = f.read()

                parsed = parse_raw_email(raw_bytes)

                record = {
                    "rel_path": str(file_path.relative_to(RAW_DIR)),
                    "folder": folder_name,
                    "label": label,
                    "file_size_bytes": parsed["raw_byte_length"],
                    "parse_status": parsed["parse_status"],
                    "parse_error": parsed["parse_error"],
                    "message_id": parsed["message_id"],
                    "from": parsed["from"],
                    "to": parsed["to"],
                    "reply_to": parsed["reply_to"],
                    "date": parsed["date"],
                    "subject": parsed["subject"],
                    "sender_domain": parsed["sender_domain"],
                    "content_type": parsed["content_type"],
                    "is_multipart": parsed["is_multipart"],
                    "has_plain": parsed["has_plain"],
                    "has_html": parsed["has_html"],
                    "plain_length": len(parsed["plain_body"]),
                    "html_length": len(parsed["html_body"]),
                    "url_count": parsed["url_count"],
                    "attachment_count": parsed["attachment_count"],
                    "encoding_info": parsed["encoding_info"],
                    # Leakage indicators
                    "has_x_spam_status": "X-Spam-Status" in parsed["leakage_headers_present"],
                    "has_x_spam_flag": "X-Spam-Flag" in parsed["leakage_headers_present"],
                    "has_x_spam_checker_version": "X-Spam-Checker-Version" in parsed["leakage_headers_present"],
                    "x_spam_status_val": parsed["leakage_headers_present"].get("X-Spam-Status", ""),
                    "raw_bytes": raw_bytes,
                    "plain_body": parsed["plain_body"],
                    "html_body": parsed["html_body"]
                }
                records.append(record)
            except Exception as e:
                records.append({
                    "rel_path": str(file_path.relative_to(RAW_DIR)),
                    "folder": folder_name,
                    "label": label,
                    "file_size_bytes": 0,
                    "parse_status": "failed",
                    "parse_error": str(e),
                    "message_id": "", "from": "", "to": "", "reply_to": "", "date": "", "subject": "",
                    "sender_domain": "", "content_type": "", "is_multipart": False,
                    "has_plain": False, "has_html": False, "plain_length": 0, "html_length": 0,
                    "url_count": 0, "attachment_count": 0, "encoding_info": "",
                    "has_x_spam_status": False, "has_x_spam_flag": False, "has_x_spam_checker_version": False,
                    "x_spam_status_val": "", "raw_bytes": b"", "plain_body": "", "html_body": ""
                })

    df = pd.DataFrame(records)
    print(f"[+] Total files inventorying complete: {len(df)} records.")

    # Save lightweight metadata summary without raw bytes/bodies to CSV
    csv_df = df.drop(columns=["raw_bytes", "plain_body", "html_body"])
    out_csv = INTERIM_DIR / "corpus_inventory.csv"
    csv_df.to_csv(out_csv, index=False)
    print(f"[+] Inventory CSV saved: {out_csv}")

    return df


if __name__ == "__main__":
    scan_and_inventory()
