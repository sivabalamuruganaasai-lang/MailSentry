"""
Unit tests for data-splitting integrity and graph connected components (src/data/grouping.py).
"""

import pytest
import pandas as pd
from src.data.grouping import (
    DisjointSetUnion,
    build_connected_components,
    group_stratified_split
)


def test_identical_normalized_hashes_grouped():
    df = pd.DataFrame([
        {"file_path": "email1.eml", "normalized_sha256": "hash_abc", "label": 0},
        {"file_path": "email2.eml", "normalized_sha256": "hash_abc", "label": 0},
        {"file_path": "email3.eml", "normalized_sha256": "hash_xyz", "label": 1},
    ])
    near_dup_df = pd.DataFrame()

    grouped_df, metrics, components = build_connected_components(df, near_dup_df)

    g1 = grouped_df.loc[grouped_df["file_path"] == "email1.eml", "group_id"].values[0]
    g2 = grouped_df.loc[grouped_df["file_path"] == "email2.eml", "group_id"].values[0]
    g3 = grouped_df.loc[grouped_df["file_path"] == "email3.eml", "group_id"].values[0]

    assert g1 == g2
    assert g1 != g3


def test_transitive_near_duplicates_grouped():
    # If A~B and B~C, then A, B, C must be in the same connected component
    df = pd.DataFrame([
        {"file_path": "A.eml", "normalized_sha256": "hA", "label": 0},
        {"file_path": "B.eml", "normalized_sha256": "hB", "label": 0},
        {"file_path": "C.eml", "normalized_sha256": "hC", "label": 0},
        {"file_path": "D.eml", "normalized_sha256": "hD", "label": 1},
    ])
    near_dup_df = pd.DataFrame([
        {"path_1": "A.eml", "path_2": "B.eml", "jaccard_similarity": 0.90},
        {"path_1": "B.eml", "path_2": "C.eml", "jaccard_similarity": 0.88},
    ])

    grouped_df, metrics, components = build_connected_components(df, near_dup_df)

    gA = grouped_df.loc[grouped_df["file_path"] == "A.eml", "group_id"].values[0]
    gB = grouped_df.loc[grouped_df["file_path"] == "B.eml", "group_id"].values[0]
    gC = grouped_df.loc[grouped_df["file_path"] == "C.eml", "group_id"].values[0]
    gD = grouped_df.loc[grouped_df["file_path"] == "D.eml", "group_id"].values[0]

    assert gA == gB == gC
    assert gD != gA


def test_connected_component_cannot_be_split():
    # Create 100 emails with 10 multi-member groups
    records = []
    for i in range(100):
        group_idx = i // 5
        records.append({
            "file_path": f"email_{i:03d}.eml",
            "normalized_sha256": f"hash_group_{group_idx}",
            "label": 1 if group_idx % 2 == 1 else 0
        })
    df = pd.DataFrame(records)
    near_dup_df = pd.DataFrame()

    grouped_df, metrics, components = build_connected_components(df, near_dup_df)
    split_df = group_stratified_split(grouped_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    # Verify zero group_id crosses multiple splits
    group_split_counts = split_df.groupby("group_id")["split"].nunique()
    cross_split_groups = (group_split_counts > 1).sum()

    assert cross_split_groups == 0


def test_class_conflicting_duplicate_groups_detected():
    df = pd.DataFrame([
        {"file_path": "ham1.eml", "normalized_sha256": "conflict_hash", "label": 0},
        {"file_path": "spam1.eml", "normalized_sha256": "conflict_hash", "label": 1},
        {"file_path": "ham2.eml", "normalized_sha256": "clean_hash", "label": 0},
    ])
    near_dup_df = pd.DataFrame()

    grouped_df, metrics, components = build_connected_components(df, near_dup_df)

    assert metrics["class_conflict_group_count"] == 1


def test_deterministic_grouping_between_runs():
    df = pd.DataFrame([
        {"file_path": "e1.eml", "normalized_sha256": "h1", "label": 0},
        {"file_path": "e2.eml", "normalized_sha256": "h1", "label": 0},
        {"file_path": "e3.eml", "normalized_sha256": "h2", "label": 1},
        {"file_path": "e4.eml", "normalized_sha256": "h3", "label": 1},
    ])
    near_dup_df = pd.DataFrame([
        {"path_1": "e3.eml", "path_2": "e4.eml", "jaccard_similarity": 0.86}
    ])

    grouped_df1, metrics1, _ = build_connected_components(df, near_dup_df)
    grouped_df2, metrics2, _ = build_connected_components(df, near_dup_df)

    split_df1 = group_stratified_split(grouped_df1, seed=42)
    split_df2 = group_stratified_split(grouped_df2, seed=42)

    pd.testing.assert_series_equal(split_df1["group_id"], split_df2["group_id"])
    pd.testing.assert_series_equal(split_df1["split"], split_df2["split"])
