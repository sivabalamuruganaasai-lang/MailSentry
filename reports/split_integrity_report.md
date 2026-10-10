# Data-Splitting Integrity & Connected Components Report

---

## Executive Summary
This report verifies that **connected components** of exact and near-duplicate emails (Jaccard similarity >= 0.85) are strictly grouped into unified partitions during dataset splitting. **Zero connected components cross partition boundaries between train, validation, and test splits.**

---

## 1. Connected Component Statistics

| Metric | Empirical Value |
|---|---|
| **Total Messages Analyzed** | **6046** |
| **Total Connected Components (Groups)** | **5777** |
| **Multi-Member Connected Components** | **187** |
| **Singleton Messages (Unique Emails)** | **5590** |
| **Largest Connected Component Size** | **9** messages |
| **Total Messages Affected by Grouping** | **456** messages |

---

## 2. Class Distribution of Affected Grouped Messages

Messages belonging to multi-member near-duplicate components (456 emails) exhibit the following label distribution:
- **HAM Messages in Groups**: **98** (21.49%)
- **SPAM Messages in Groups**: **358** (78.51%)
- **Class-Conflicting Groups**: **0** (Zero identical/near-duplicate messages possess conflicting ground-truth labels).

---

## 3. Proposed Group-Stratified Split Verification

Using group-aware stratification (70% Train / 15% Validation / 15% Test, Random State = 42):

| Partition | Total Messages | HAM Count | SPAM Count | SPAM % | Unique Group IDs |
|---|---|---|---|---|---|
| **Train (70%)** | **4233** | 2901 | 1332 | 31.47% | 4043 |
| **Validation (15%)** | **907** | 624 | 283 | 31.2% | 866 |
| **Test (15%)** | **906** | 625 | 281 | 31.02% | 868 |

---

## 4. Verification & Split Isolation Proof

- **Number of Connected Components Crossing Partition Boundaries**: **0**
- **Data Leakage Verdict**: 🟢 **PASSED**. 100% of connected components remain entirely isolated within exactly one dataset partition.

---

## 5. Verification Rules Enforced
1. **Identical Normalized Hashes**: Grouped into identical component IDs via Union-Find.
2. **Transitive Near-Duplicates**: If A ~ B and B ~ C, emails A, B, C are transitively merged into a single component.
3. **Group Isolation**: Every node in component C is mapped to the same split S in {train, val, test}.
4. **Deterministic Partitioning**: Group assignment depends strictly on random state seed 42 and deterministic component ID ordering.