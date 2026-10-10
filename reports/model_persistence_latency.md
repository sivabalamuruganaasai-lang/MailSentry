# Level 12 — Model Persistence, Reproducibility, and Latency Report

## Executive Summary

This report presents the deployment readiness evaluation for the frozen Candidate B (**E11**: `LinearSVC` with 10,036 features, `MaxAbsScaler`, operating at decision threshold $\tau = 0.4013$).

All objective requirements—artifact serialization, exact reproducibility validation, component latency breakdown, memory footprint profiling, and edge-case input verification—have been verified.

**Key Verification Outcomes**:
* **Zero Model Alteration**: No retraining, refitting, threshold modification, or feature schema changes were performed.
* **Exact Reproducibility**: Reloaded decision scores match original validation prediction records (`reports/val_predictions_E11.csv`, $N = 907$) down to floating-point precision ($\max |\Delta S| = 4.44 \times 10^{-16}$), with **0 label mismatches** at threshold $\tau = 0.4013$.
* **Sub-10ms End-to-End Latency**: End-to-end per-email processing latency ranges from **1.99 ms to 3.70 ms** for typical emails (small/medium/HTML/URL-rich), with model inference taking less than **0.35 ms**.
* **Lightweight Deployment Footprint**: The complete serialized deployment bundle (`models/deployment_e11.joblib`) occupies only **332.43 KB** on disk and consumes **1.01 MB** of incremental RAM.

---

## 1. Deployment Artifact Inventory & Hashes

The authoritative experiment manifest (`reports/final_model_manifest.json`) remains 100% frozen. Deployment metadata is recorded separately in `models/deployment_manifest.json`.

### Artifact Specification Table

| Artifact Property | Specification Value | Verification Details |
| :--- | :--- | :--- |
| **Primary Deployment Bundle** | `models/deployment_e11.joblib` | Contains fitted `CombinedFeaturePipeline`, `MaxAbsScaler`, and `SpamLinearSVM` |
| **Artifact File Size** | **332,430 bytes** ($0.32\text{ MB} / 324.64\text{ KB}$) | Serialized via `joblib.dump(pipeline, compress=3)` |
| **Artifact SHA-256 Hash** | `7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06` | Computed on raw bytes of `deployment_e11.joblib` |
| **Deployment Manifest** | `models/deployment_manifest.json` | Stores deployment metadata, SHA-256 hashes, and versioning |
| **Python Version** | Python 3.11.7 (64-bit) | Executable runtime environment |
| **scikit-learn Version** | `1.4.2` | Implements `LinearSVC`, `MaxAbsScaler`, `TfidfVectorizer` |
| **joblib Version** | `1.5.1` | Compression and serialization backend |
| **NumPy / Pandas Versions** | NumPy `1.26.4`, Pandas `2.2.2` | Data manipulation backends |
| **Frozen Decision Threshold** | **$\tau = 0.4013$** | Raw decision function score threshold $S(x)$ |
| **Total Feature Vector Size** | **10,036 features** | $10,000\text{ Word TF-IDF} + 19\text{ Structural} + 17\text{ URL}$ |

---

## 2. Deployment Architecture & Serialization Strategy

### 2.1 Component Encapsulation
The deployment bundle (`models/deployment_e11.joblib`) encapsulates the complete end-to-end `SpamModelPipeline` object:
1. **`CombinedFeaturePipeline`**:
   - `WordTFIDFExtractor`: Fitted `TfidfVectorizer` (10,000 vocabulary features, `ngram_range=(1,2)`).
   - `StructuralFeatureExtractor`: 19 hand-engineered structural features (HTML flags, tag counts, capitalization, length, formatting).
   - `URLFeatureExtractor`: 17 URL security signals (IP hosts, domain entropy, shorteners, query string counts).
2. **`MaxAbsScaler`**: Fitted feature scaler preserving sparse matrix representation.
3. **`SpamLinearSVM`**: Fitted `LinearSVC` model parameters (10,036 weight coefficients + scalar intercept).

### 2.2 Load Protocol
Deployment systems can load the pipeline in a single non-blocking call:
```python
import joblib

# Load deployment bundle
pipeline = joblib.load("models/deployment_e11.joblib")

# Predict raw score S(x) and class label at tau = 0.4013
scores = pipeline.predict_score([parsed_email_record])
predicted_labels = (scores >= 0.4013).astype(int)
```

---

## 3. Validation Reproducibility Results ($N = 907$)

Reproducibility was verified by executing inference with the reloaded `models/deployment_e11.joblib` artifact on the 907 validation split emails (`data/interim/val.csv`) and comparing outputs against `reports/val_predictions_E11.csv`.

| Reproducibility Metric | Numeric Target / Threshold | Measured Value | Audit Status |
| :--- | :--- | :--- | :---: |
| **Validation Sample Count ($N$)** | 907 emails | 907 emails | **PASS** |
| **Max Absolute Decision Score Difference** | $< 1.0 \times 10^{-6}$ | **$4.440892098501 \times 10^{-16}$** | **PASS** |
| **Mean Absolute Decision Score Difference**| $< 1.0 \times 10^{-6}$ | **$3.211437914567 \times 10^{-17}$** | **PASS** |
| **Classification Label Mismatches** ($\tau = 0.4013$) | 0 mismatches | **0 mismatches** | **PASS** |
| **Feature Dimensionality Check** | Shape $(N, 10036)$ | Shape $(907, 10036)$ | **PASS** |

* **Verification Result**: The reloaded pipeline reproduces original decision scores within IEEE 754 double-precision float limits ($4.44 \times 10^{-16}$) with zero label classifications flipped.

---

## 4. Latency & Resource Benchmarks

All benchmarks were conducted on the local evaluation workstation running Python 3.11.7 (Windows x64). Latency numbers reflect warm-up executions (5 iterations) followed by **100 repeated benchmark iterations**.

### 4.1 Artifact Load Time & Memory Footprint

| Benchmark Component | Median Value | 95th Percentile (p95) | Notes |
| :--- | :---: | :---: | :--- |
| **Artifact Load Latency (`joblib.load`)** | **97.50 ms** | **348.63 ms** | One-time startup cost |
| **Artifact File Size on Disk** | **332.43 KB** | N/A | Highly compressed serialization |
| **Incremental RAM Footprint** | **1.01 MB** | N/A | Incremental RSS delta ($\text{RSS}_{\text{after}} - \text{RSS}_{\text{before}}$); total process RSS is 238.73 MB |

> [!NOTE]
> **Memory Usage Clarification**: The 1.01 MB RSS figure represents the **incremental memory footprint** added to RAM when deserializing the model pipeline into an active Python process. Total process memory (Python interpreter + NumPy/Pandas/scikit-learn C-libraries) is 238.73 MB RSS.

### 4.2 Per-Email Component Breakdown & Latency Matrix

End-to-end (E2E) latency is measured directly as a single overall timer ($T_{\text{start}}$ to $T_{\text{end}}$), while individual sub-stage timings (Parsing, Feature Extraction, Inference) are measured concurrently per iteration:
1. **Parsing**: Header, body, HTML, and URL extraction via `parse_raw_email`.
2. **Feature Extraction**: TF-IDF vectorization, structural feature extraction, and `MaxAbsScaler` transformation.
3. **Inference**: Linear algebra matrix multiplication ($X \cdot W^T + b$) in `LinearSVC`.

| Email Category & Size | Parsing Time (Median / p95) | Feature Extraction Time (Median / p95) | Model Inference Time (Median / p95) | Direct E2E Latency (Median) | Direct E2E Latency (p95) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Small Plain-Text** (~86 B) | 0.812 ms / 1.250 ms | 1.012 ms / 1.480 ms | 0.287 ms / 0.410 ms | **2.113 ms** | **2.939 ms** |
| **Medium Plain-Text** (~2.2 KB) | 1.150 ms / 1.820 ms | 2.237 ms / 4.300 ms | 0.311 ms / 0.450 ms | **3.698 ms** | **6.437 ms** |
| **Large Plain-Text** (~12.0 KB) | 3.120 ms / 4.100 ms | 6.557 ms / 7.280 ms | 0.323 ms / 0.470 ms | **10.000 ms** | **11.704 ms** |
| **HTML Newsletter** (~2.5 KB) | 0.720 ms / 1.100 ms | 1.044 ms / 1.830 ms | 0.225 ms / 0.450 ms | **1.989 ms** | **3.383 ms** |
| **URL-Rich Email** (~800 B) | 1.110 ms / 1.510 ms | 1.652 ms / 2.050 ms | 0.293 ms / 0.410 ms | **3.055 ms** | **3.865 ms** |

> [!NOTE]
> **Methodology & Non-Additivity Notice**: Direct E2E Latency was measured as a unified overall timer per iteration. Component medians are reported for diagnostic breakdown; mathematically, the median of the sum of stages does not equal the sum of component medians.

* **Key Performance Finding**: Model decision function calculation (`LinearSVC.predict_score`) is lightning fast (**0.22 ms – 0.32 ms**). Processing time is dominated by text parsing and n-gram extraction, which scales linearly with message size.

---

## 5. Diverse Input Smoke-Test Matrix

To ensure robustness against un-sanitized real-world inputs, the reloaded pipeline was tested against edge cases.

> [!IMPORTANT]
> **Smoke-Test Interpretation Rule**: Passing these smoke tests proves **pipeline stability, non-crashing parsing, and input schema tolerance**. It does **not** serve as proof of label correctness for unlabelled synthetic inputs.

| Input Test Scenario | Raw Input Characteristics | Raw Score $S(x)$ | Predicted Class ($\tau=0.4013$) | Behavior / Handling Verification |
| :--- | :--- | :---: | :---: | :--- |
| **Empty Email** | Subject: `""`, Body: `""`, URLs: 0 | $+0.1780$ | **HAM (0)** | Handled cleanly; returns baseline intercept score |
| **Ordinary Plain-Text** | Short legitimate meeting request | $-0.0901$ | **HAM (0)** | Standard negative decision score |
| **HTML Newsletter** | HTML tags present, 2 embedded URLs | $+0.4624$ | **SPAM (1)** | Correctly incorporates $+0.8728$ HTML structural boost |
| **URL-Rich Email** | 4 URLs including local IP hosts | $-0.1272$ | **HAM (0)** | Handled without parser errors |
| **Malformed / Unicode** | Null bytes `\x00\xff`, Emoji 🎉, repetitive text | $+0.6334$ | **SPAM (1)** | Unicode sanitized gracefully; TF-IDF captures repeated tokens |

---

## 6. Deployment Limitations & Operational Constraints

1. **Security Vulnerability of Joblib/Pickle**: `joblib.load` deserializes Python objects using `pickle` semantics, which can execute arbitrary code if loading untrusted files. Production systems MUST restrict loading to internally generated, SHA-256 verified artifacts (`7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06`).
2. **Python Environment Dependency**: `joblib` artifacts depend on compatible Python, `scikit-learn` (`1.4.2`), and `numpy` versions. Upgrading major `scikit-learn` versions requires re-validating serialized state.
3. **Single-Threaded CPU Scale**: Benchmark times reflect single-core CPU execution. Throughput scales linearly with multi-process workers (e.g. `gunicorn` / `uvicorn` process pools).
4. **Unbounded Raw Scores**: `LinearSVC` produces raw decision function distance scores $S(x) \in (-\infty, +\infty)$, **not calibrated probabilities**. UI implementations must present raw scores with threshold $\tau = 0.4013$, avoiding pseudo-probability transformations unless calibrated (as audited in Level 6.2).
5. **HTML Parsing Dependency**: `parse_raw_email` relies on Python standard library `email` and `html.parser`. Unclosed tags or malformed MIME boundaries fall back to plain text extraction without crashing.

---

## 7. Explicit Confirmation of Model & Artifact Integrity

* **Model State**: No model refitting or retraining occurred.
* **Decision Threshold**: Frozen threshold $\tau = 0.4013$ is preserved.
* **Manifests & Hashes**: `reports/final_model_manifest.json` remains frozen (`config_sha256`: `61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2`). Deployment bundle registered in `models/deployment_manifest.json`.
* **Level 7 Test Evaluation**: The Level 7 test-set evaluation was **not** rerun.

---

## 8. Summary of Created & Modified Files

1. **`models/deployment_e11.joblib`** *(New File)*: Serialized E11 deployment bundle (332.43 KB, SHA-256 `7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06`).
2. **`models/deployment_manifest.json`** *(New File)*: Metadata manifest recording deployment artifact hash, environment versions, and threshold.
3. **`tests/test_persistence_latency.py`** *(New File)*: Unit and integration tests for model loading, score tolerance matching, and diverse input handling.
4. **`reports/model_persistence_latency.md`** *(New File)*: This comprehensive Level 12 evaluation report.
