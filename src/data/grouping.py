"""
Graph-Based Connected Component Grouping and Data-Splitting Integrity Module.

Implements Union-Find (Disjoint Set Union) to group emails sharing:
1. Identical normalized_sha256 hashes, OR
2. Jaccard similarity >= 0.85 (near-duplicates).

Ensures connected components are NEVER split across train/validation/test partitions.
"""

from typing import Dict, List, Tuple, Set, Any
import numpy as np
import pandas as pd
from pathlib import Path


class DisjointSetUnion:
    """Disjoint Set Union (Union-Find) with path compression and rank optimization."""

    def __init__(self, elements: List[Any]):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, item: Any) -> Any:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: Any, item2: Any) -> None:
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                root1, root2 = root2, root1
            self.parent[root2] = root1
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1

    def get_components(self) -> Dict[Any, List[Any]]:
        """Return mapping from root representative to list of element members."""
        components = {}
        for item in self.parent:
            root = self.find(item)
            if root not in components:
                components[root] = []
            components[root].append(item)
        return components


def build_connected_components(
    df: pd.DataFrame,
    near_dup_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Build connected components of duplicate and near-duplicate emails.
    Returns grouped DataFrame with 'group_id' and summary metrics.
    """
    email_ids = df["file_path"].tolist() if "file_path" in df.columns else df["rel_path"].tolist()
    dsu = DisjointSetUnion(email_ids)

    # 1. Union exact normalized_sha256 duplicates
    hash_groups = df.groupby("normalized_sha256")
    for norm_hash, group in hash_groups:
        if norm_hash and len(group) > 1:
            paths = group["file_path"].tolist() if "file_path" in group.columns else group["rel_path"].tolist()
            first = paths[0]
            for p in paths[1:]:
                dsu.union(first, p)

    # 2. Union near-duplicates from near_dup_df (Jaccard >= 0.85)
    if near_dup_df is not None and not near_dup_df.empty:
        for _, row in near_dup_df.iterrows():
            p1 = row["path_1"]
            p2 = row["path_2"]
            if p1 in dsu.parent and p2 in dsu.parent:
                dsu.union(p1, p2)

    # Extract component mapping
    components = dsu.get_components()

    # Assign deterministic integer component IDs sorted by component size (descending)
    sorted_roots = sorted(components.keys(), key=lambda r: (-len(components[r]), str(r)))

    path_to_group_id = {}
    for group_idx, root in enumerate(sorted_roots):
        for path in components[root]:
            path_to_group_id[path] = f"GROUP_{group_idx:04d}"

    path_col = "file_path" if "file_path" in df.columns else "rel_path"
    df_grouped = df.copy()
    df_grouped["group_id"] = df_grouped[path_col].map(path_to_group_id)

    # Calculate metrics
    group_sizes = df_grouped["group_id"].value_counts()
    num_components = len(group_sizes)
    singletons = int((group_sizes == 1).sum())
    multi_member_groups = int((group_sizes > 1).sum())
    largest_component_size = int(group_sizes.max()) if not group_sizes.empty else 0
    affected_messages = int((group_sizes[group_sizes > 1]).sum())

    # Detect class conflicts per component
    class_conflicts = []
    group_label_counts = df_grouped.groupby("group_id")["label"].nunique()
    for g_id, label_count in group_label_counts.items():
        if label_count > 1:
            class_conflicts.append(g_id)

    metrics = {
        "total_messages": len(df_grouped),
        "total_components": num_components,
        "singleton_count": singletons,
        "multi_member_group_count": multi_member_groups,
        "largest_component_size": largest_component_size,
        "affected_messages_count": affected_messages,
        "class_conflict_group_count": len(class_conflicts),
        "class_conflict_groups": class_conflicts
    }

    return df_grouped, metrics, components


def group_stratified_split(
    df_grouped: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Perform deterministic group-aware stratified train/val/test split.
    Guarantees NO group_id spans across multiple splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    np.random.seed(seed)

    # Aggregate by group_id
    group_info = []
    for g_id, group in df_grouped.groupby("group_id"):
        size = len(group)
        # Determine group primary label (majority vote or ham if tie)
        spam_count = (group["label"] == 1).sum()
        ham_count = (group["label"] == 0).sum()
        label = 1 if spam_count > ham_count else 0
        group_info.append({
            "group_id": g_id,
            "size": size,
            "label": label,
            "spam_count": spam_count,
            "ham_count": ham_count
        })

    group_df = pd.DataFrame(group_info)

    # Sort deterministically before shuffling
    group_df = group_df.sort_values(by=["group_id"]).reset_index(drop=True)

    # Separate ham and spam groups to stratify
    ham_groups = group_df[group_df["label"] == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    spam_groups = group_df[group_df["label"] == 1].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    group_split_map = {}

    def assign_splits(g_subdf):
        n = len(g_subdf)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for i, row in g_subdf.iterrows():
            gid = row["group_id"]
            if i < n_train:
                group_split_map[gid] = "train"
            elif i < n_train + n_val:
                group_split_map[gid] = "val"
            else:
                group_split_map[gid] = "test"

    assign_splits(ham_groups)
    assign_splits(spam_groups)

    df_split = df_grouped.copy()
    df_split["split"] = df_split["group_id"].map(group_split_map)
    return df_split
