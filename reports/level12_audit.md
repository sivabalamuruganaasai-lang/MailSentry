# Level 12 Audit Report — Persistence, Reproducibility, and Latency Verification

## Executive Summary

This report presents a focused, **READ-ONLY audit** of the Level 12 Model Persistence, Reproducibility, and Latency evaluation conducted for Candidate B (**E11**: `LinearSVC` with 10,036 features, `MaxAbsScaler`, operating at frozen threshold $\tau = 0.4013$).

All 9 audit verification points specified by the audit protocol were executed against existing deployment artifacts (`models/deployment_e11.joblib`), deployment manifests (`models/deployment_manifest.json`), frozen experiment manifests (`reports/final_model_manifest.json`), validation prediction files (`reports/val_predictions_E11.csv`), benchmark logs, and test suites.

**Audit Verdict**: **PASS WITH NO CRITICAL DEFECTS**. The deployment bundle SHA-256 hash matches the deployment manifest exactly, score-equivalence is verified down to floating-point precision ($4.44 \times 10^{-16}$), all 84 test suite items pass, and all latency and memory claims have been verified with appropriate methodology notes documented.

---

## 1. Audit Check 1: Deployment Artifact SHA-256 Hash Verification

### Verification Procedure
We recomputed the SHA-256 digest on the raw binary bytes of `models/deployment_e11.joblib` and compared it against the recorded checksum in `models/deployment_manifest.json`.

### SHA-256 Verification Table

| Artifact Name | Metric / Property | Value | Audit Verification |
| :--- | :--- | :--- | :---: |
| **`models/deployment_e11.joblib`** | File Size | **332,430 bytes** ($332.43\text{ KB} / 0.32\text{ MB}$) | Validated |
| **`models/deployment_e11.joblib`** | Recomputed SHA-256 | `7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06` | **MATCH** |
| **`models/deployment_manifest.json`**| Manifest SHA-256 | `7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06` | **MATCH** |

* **Result**: **PASS**. 100% hash integrity confirmed.

---

## 2. Audit Check 2: Serialized Component & Schema Verification

We unpickled `models/deployment_e11.joblib` and inspected its internal Python structure:

1. **Pipeline Container**: Encapsulates `SpamModelPipeline` with `is_fitted_ = True`.
2. **Feature Extractor**: `CombinedFeaturePipeline` holding:
   - `WordTfidfExtractor` (fitted `TfidfVectorizer`, 10,000 max features, n-gram (1,2)).
   - `StructuralFeatureExtractor` (19 features, `return_sparse=True`).
   - `URLFeatureExtractor` (17 features, `return_sparse=True`).
3. **Scaler**: `MaxAbsScaler` (fitted on 4,233 training records).
4. **Estimator**: `SpamLinearSVM` (fitted `LinearSVC` with $C=1.0$).
5. **Feature Dimensionality**: Transformed feature matrix shape verified at $(N, 10036)$.
6. **Threshold Representation**: Threshold $\tau = 0.4013$ is recorded in deployment manifest metadata.

---

## 3. Audit Check 3: Score-Equivalence Verification ($N = 907$)

Reproducibility was verified by executing inference with the reloaded `models/deployment_e11.joblib` artifact on the 907 validation split records (`data/interim/val.csv`) and comparing outputs against `reports/val_predictions_E11.csv`.

| Metric | Target | Measured Value | Audit Status |
| :--- | :--- | :--- | :---: |
| **Validation Sample Count ($N$)** | 907 samples | 907 samples | **PASS** |
| **Max Absolute Score Difference** | $< 1.0 \times 10^{-6}$ | **$4.440892098501 \times 10^{-16}$** | **PASS** |
| **Mean Absolute Score Difference**| $< 1.0 \times 10^{-6}$ | **$3.211437914567 \times 10^{-17}$** | **PASS** |
| **Label Mismatches at Threshold $0.4013$** | 0 mismatches | **0 mismatches** | **PASS** |

---

## 4. Audit Check 4: Benchmark Methodology & Additivity Inspection

We audited the latency benchmark methodology in `scratch/run_level12_benchmarks.py`:

1. **Warm-up**: 5 warm-up iterations were executed before timing to eliminate cold-start CPU cache and Python import skew.
2. **Repetitions**: 100 repeated timed iterations per email category.
3. **Percentiles**: Median (50th percentile) and 95th percentile (p95) were computed using `np.median` and `np.percentile`.
4. **Input Size Coverage**: Tested across 5 distinct size categories: Small Plain-Text (86 B), Medium Plain-Text (2.2 KB), Large Plain-Text (12 KB), HTML Newsletter (2.5 KB), and URL-Rich (800 B).
5. **Additivity Audit Notice**: Direct E2E Latency was measured as a single overall timer ($T_{\text{start}}$ to $T_{\text{end}}$) per iteration, while sub-stage timings were recorded concurrently. Mathematically, the sum of component medians ($\text{Median}(A) + \text{Median}(B) + \text{Median}(C)$) does not necessarily equal the median of total end-to-end time ($\text{Median}(A+B+C)$). The report has been updated to explicitly label Total E2E Latency as **Direct E2E Latency**.

---

## 5. Audit Check 5: Memory Footprint & RSS Measurement Clarification

We inspected the memory footprint measurement in `scratch/run_level12_benchmarks.py`:

* **Measurement Method**: Measured process Resident Set Size (RSS) delta via `psutil.Process().memory_info().rss` before and after unpickling `models/deployment_e11.joblib`.
* **Before Unpickling RSS**: 237.73 MB (includes Python 3.11 interpreter, scikit-learn, NumPy, Pandas, and loaded dataset splits).
* **After Unpickling RSS**: 238.73 MB.
* **Incremental Footprint Delta**: **1.01 MB** ($\text{RSS}_{\text{after}} - \text{RSS}_{\text{before}}$).
* **Audit Clarification**: The 1.01 MB figure represents the **incremental memory footprint** incurred by deserializing the fitted model pipeline object into Python RAM. Total process memory required to run the runtime environment is 238.73 MB RSS.

---

## 6. Audit Check 6: Smoke-Test Output Interpretation

We audited the diverse input smoke-test results:

* **Empty Email**: $S(x) = +0.1780$ (HAM)
* **Ordinary Plain-Text**: $S(x) = -0.0901$ (HAM)
* **HTML Newsletter**: $S(x) = +0.4624$ (SPAM)
* **URL-Rich Email**: $S(x) = -0.1272$ (HAM)
* **Malformed / Unicode**: $S(x) = +0.6334$ (SPAM)

> [!IMPORTANT]
> **Audit Finding on Smoke Tests**: Passing these synthetic smoke tests confirms **pipeline stability, non-crashing parsing, unicode resilience, and input schema tolerance**. It does **not** prove ground-truth label correctness for unlabelled synthetic inputs.

---

## 7. Audit Check 7: Test Suite Log Verification

We inspected existing pytest task execution logs (`task-1161`):

```text
================== 84 passed, 1 warning in 92.33s (0:01:32) ===================
```

* **Passed**: 84 tests (including 4 new tests in `tests/test_persistence_latency.py`).
* **Failed**: 0.
* **Warning**: 1 expected `UserWarning` in `test_models.py::test_cv_vectorizer_fitted_per_fold_only` (small synthetic CV class count fixture).
* **Audit Status**: Confirmed exact log match.

---

## 8. Audit Check 8: Security & Compatibility of Joblib / Pickle Artifacts

1. **Arbitrary Code Execution Risk**: `joblib.load` relies on Python `pickle` deserialization. Loading untrusted `.joblib` files can execute arbitrary Python bytecode. Production deployment pipelines MUST restrict loading to internally generated, SHA-256 verified artifacts (`7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06`).
2. **Version Pinning**: Serialized objects depend on exact C-extension ABIs (`scikit-learn 1.4.2`, `numpy 1.26.4`). Major library upgrades require re-verifying unpickled state.

---

## 9. Audit Check 9: Frozen Configuration & Test Isolation

* **Experiment Manifest**: `reports/final_model_manifest.json` (`config_sha256`: `61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2`) remains **100% frozen**.
* **Decision Threshold**: Threshold $\tau = 0.4013$ is intact.
* **Split Assignments**: `data/interim/group_split_manifest.csv` remains unchanged.
* **Locked Test Evaluation**: The Level 7 test set evaluation was **not** rerun or modified.

---

## 10. Audit Conclusion

The Level 12 deployment artifacts (`models/deployment_e11.joblib`, `models/deployment_manifest.json`), test suite (`tests/test_persistence_latency.py`), and documentation report (`reports/model_persistence_latency.md`) fully satisfy all requirements for accuracy, empirical verification, reproducibility, latency benchmarking, and test-set isolation.
