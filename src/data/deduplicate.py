"""
Deduplication & Similarity Analysis Script.

Computes:
1. Exact raw-file SHA-256 hash.
2. Normalized-content SHA-256 hash (cleaned subject + body).
3. Near-duplicate similarity analysis (MinHash / Jaccard n-gram overlap).
4. Flags duplicate groups and label conflicts across ham and spam partitions.
"""

import hashlib
import re
from typing import Dict, List, Tuple, Set, Any
import pandas as pd
from pathlib import Path


def compute_raw_sha256(raw_bytes: bytes) -> str:
    """Compute SHA-256 of exact raw byte payload."""
    if not raw_bytes:
        return ""
    return hashlib.sha256(raw_bytes).hexdigest()


def compute_normalized_sha256(subject: str, plain_body: str, html_body: str) -> str:
    """
    Compute SHA-256 of normalized text payload (lowercase, stripped whitespace).
    Ignores dynamic headers and timestamp variations.
    """
    combined = (subject or "") + " " + (plain_body or "") + " " + (html_body or "")
    # Lowercase and collapse whitespace
    normalized = re.sub(r'\s+', ' ', combined.lower()).strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def extract_char_shingles(text: str, k: int = 5) -> Set[str]:
    """Extract character k-shingles for near-duplicate Jaccard similarity."""
    cleaned = re.sub(r'\s+', ' ', text.lower()).strip()
    if len(cleaned) < k:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i+k] for i in range(len(cleaned) - k + 1)}


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity score between two shingle sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def analyze_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform multi-level duplicate detection and label conflict audit on inventory DataFrame.
    """
    print("[>] Running exact and normalized deduplication analysis...")

    # Exact raw SHA-256 duplicates
    raw_sha_counts = df["raw_sha256"].value_counts()
    exact_raw_dup_hashes = set(raw_sha_counts[raw_sha_counts > 1].index)

    # Normalized content SHA-256 duplicates
    norm_sha_counts = df["normalized_sha256"].value_counts()
    norm_dup_hashes = set(norm_sha_counts[norm_sha_counts > 1].index)

    # Class conflicts audit
    # Group by normalized hash and check unique labels
    label_conflicts = []
    norm_hash_groups = df.groupby("normalized_sha256")["label"].unique()
    for norm_hash, labels in norm_hash_groups.items():
        if len(labels) > 1 and norm_hash != "":
            label_conflicts.append(norm_hash)

    print("[>] Running near-duplicate similarity audit (threshold >= 0.85)...")
    # Sample or pairwise near-duplicate check on normalized body texts
    near_duplicate_pairs = []
    # To keep performance optimal on ~6k emails, check emails sharing length buckets or top terms
    # Build light shingle representations for emails
    email_shingles = []
    for idx, row in df.iterrows():
        text = str(row.get("subject", "")) + " " + str(row.get("plain_body", ""))
        shingles = extract_char_shingles(text, k=7)
        email_shingles.append((row["rel_path"], row["label"], shingles, len(text)))

    # Compare candidates with similar lengths
    near_dup_count = 0
    for i in range(len(email_shingles)):
        path1, label1, s1, len1 = email_shingles[i]
        if not s1 or len1 < 50:
            continue
        # Compare with subset
        for j in range(i + 1, min(i + 50, len(email_shingles))):
            path2, label2, s2, len2 = email_shingles[j]
            if not s2 or len2 < 50:
                continue
            # Quick length ratio filter
            if abs(len1 - len2) / max(len1, len2) > 0.15:
                continue
            sim = jaccard_similarity(s1, s2)
            if sim >= 0.85 and path1 != path2:
                near_dup_count += 1
                near_duplicate_pairs.append({
                    "path_1": path1,
                    "label_1": label1,
                    "path_2": path2,
                    "label_2": label2,
                    "jaccard_similarity": round(sim, 4),
                    "is_class_conflict": label1 != label2
                })

    metrics = {
        "total_emails": len(df),
        "unique_raw_sha256": df["raw_sha256"].nunique(),
        "exact_raw_duplicates": len(df) - df["raw_sha256"].nunique(),
        "unique_normalized_sha256": df["normalized_sha256"].nunique(),
        "normalized_duplicates": len(df) - df["normalized_sha256"].nunique(),
        "label_conflict_count": len(label_conflicts),
        "near_duplicate_pairs_found": near_dup_count
    }

    print("[+] Deduplication analysis metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    near_dup_df = pd.DataFrame(near_duplicate_pairs)
    return near_dup_df, metrics
