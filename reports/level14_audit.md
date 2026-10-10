# Level 14 Audit Report — Headless Inference Core, Batch CLI & Production Health Check

## Executive Summary

This report delivers the engineering audit and empirical verification for **Level 14**: Headless Inference Engine, Batch CLI, Deployment Health Check, Performance Benchmarking, and Operational Documentation for the frozen **E11** Spam Classification Engine (`models/deployment_e11.joblib`).

Level 14 was executed in strict adherence to the approved architecture and user decisions:
1. **Decision 1 (Approved — Option A)**: Created canonical headless inference module `src/models/inference.py`. Reused directly by the Streamlit dashboard (`src/ui/app.py`) and the batch CLI (`src/cli/batch_predict.py`) without duplicating decision thresholds, three-tier routing, or artifact verification logic.
2. **Decision 2 (Approved — Option A with Explicit Safeguards)**: Enforced a 3-state exit code contract:
   - Exit code `0`: Batch completed (clean or per-record failures recorded in output under default non-strict mode).
   - Exit code `1`: Fatal invocation, artifact integrity, dependency, or output destination error.
   - Exit code `2`: Batch completed with one or more per-record failures under `--strict` mode.
3. **Decision 3 (Deferred)**: Git commands (`git init`, staging, commits) were deferred. A proposed `.gitignore` and exact baseline staging inventory are documented in Section 6 of this report for user review.

**Overall Level 14 Audit Verdict**: **100% PASS (VERIFIED)**.
- Full test suite: **104 passed, 0 failed, 1 expected warning** (94 existing + 10 new CLI unit/integration tests).
- Empirical 100-email throughput: **0.6520 s** for Directory Mode (153.4 emails/sec) and **0.2857 s** for CSV Mode (350.0 emails/sec), exceeding the 100 emails / <3.0 s target.
- Pre-deserialization SHA-256 gate verified: `7752ac87...` verified before any Joblib unpickling.
- Frozen parameters, model weights, splits, and thresholds remain 100% untouched.

---

## 1. Architecture & Component Implementation

### 1.1 Canonical Headless Module (`src/models/inference.py`)
- Centralizes all production inference logic in a pure-Python library with zero UI dependencies:
  - `EXPECTED_ARTIFACT_SHA256`: `7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06`
  - `MAX_INPUT_BYTES`: `2,097,152` (2 MiB)
  - `BINARY_THRESHOLD`: `0.4013`
  - `LOWER_THREE_TIER`: `-0.3520`
  - `UPPER_THREE_TIER`: `0.6010`
- Provides core functions:
  - `verify_artifact_hash(path)`: Computes file SHA-256 and compares in constant time prior to deserialization.
  - `load_deployment_pipeline(path)`: Verifies hash, safely loads Joblib bundle, and asserts pipeline integrity.
  - `evaluate_binary_classification(score)`: Exact step-function ($\ge 0.4013 \implies \text{SPAM}$, else $\text{HAM}$).
  - `evaluate_three_tier_routing(score)`: Exact tier mapping ($<-0.3520 \implies \text{deliver}$, $\in [-0.3520, 0.6010] \implies \text{review}$, $>0.6010 \implies \text{quarantine}$).
  - `build_feature_record(parsed)`: Maps raw/parsed email dictionaries into feature-extractor compatible dictionaries.

### 1.2 Refactored Streamlit Web App (`src/ui/app.py`)
- Updated to import all constants and model-loading routines directly from `src.models.inference`.
- Re-exports all constants and helper functions to guarantee 100% backward compatibility with `tests/test_dashboard.py`.
- Preserves all UI tabs, explainability breakdowns, offline URL indicators, and 2 MiB file uploader validation.

### 1.3 Pre-Flight Deployment Health Check CLI (`src/cli/healthcheck.py`)
- Independent diagnostic utility with zero UI dependencies.
- Can be invoked as `python -m src.cli.healthcheck` or `python src/cli/healthcheck.py`.
- Supports human-readable ASCII table and machine-readable JSON (`--json`).
- Executes 4 verification checks:
  1. Python runtime & core library import checks.
  2. Pre-deserialization SHA-256 hash check of `models/deployment_e11.joblib`.
  3. Scikit-learn pipeline structure inspection (steps, vocabulary, scaler).
  4. Deterministic smoke test inference using canonical *Legitimate Team Sync* fixture. Expected score is `+0.020524`, validated against tolerance $\pm 10^{-4}$.
- Exits with code `0` on success, code `1` on failure.

### 1.4 High-Throughput Batch Inference CLI (`src/cli/batch_predict.py`)
- High-throughput streaming batch processor supporting:
  - Directory Mode: `--input-dir` scans `.eml` and `.txt` files.
  - CSV Mode: `--input-csv` processes tabular records in bounded chunks.
- Features:
  - **Memory Safeguards**: Enforces 2 MiB file size check using `os.path.getsize()` *before* file bytes are read into memory.
  - **Chunked Processing**: Default chunk size of 250 records keeps working memory $<100$ MB.
  - **1-to-1 Mapping**: Every input record generates exactly one output record in identical sequence. Oversized or corrupt records produce a failed record with diagnostic error details; records are never silently dropped.
  - **Parser Status Taxonomy**: Explicitly tracks `success`, `partial`, and `failed`.
  - **I/O Channel Purity**: Diagnostic summaries and progress logging stream strictly to `stderr`. Valid CSV/JSONL records stream cleanly to `stdout` when `--output-file` is omitted.
  - **Strict Mode**: Optional `--strict` flag triggers exit code `2` if any individual record encountered an error.

---

## 2. Test Suite Execution & Regression Audit

The automated test suite was executed via `pytest tests/`:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\email-spam-classifier
plugins: anyio-4.12.1
collected 104 items

tests\test_cli.py ..........                                             [  9%]
tests\test_dashboard.py ..........                                       [ 19%]
tests\test_data_validation.py ....                                       [ 23%]
tests\test_error_analysis.py ...                                         [ 25%]
tests\test_explainability.py ......                                      [ 31%]
tests\test_features.py .........                                         [ 40%]
tests\test_models.py ...........                                         [ 50%]
tests\test_parser.py .......                                             [ 57%]
tests\test_persistence_latency.py ....                                   [ 61%]
tests\test_robustness.py ....                                            [ 65%]
tests\test_splitting_integrity.py .....                                  [ 70%]
tests\test_threshold_optimization.py ..................                  [ 87%]
tests\test_url_security.py .............                                 [100%]

============================== warnings summary ===============================
tests/test_models.py::test_cv_vectorizer_fitted_per_fold_only
  UserWarning: The least populated class in y has only 2 members, which is less than n_splits=3.
================= 104 passed, 1 warning in 108.76s (0:01:48) =================
```

### Module-by-Module Test Breakdown

| Test Module | Tests Passed | Description / Coverage |
| :--- | :---: | :--- |
| **`tests/test_cli.py` (NEW)** | **10** | Pre-flight health check clean pass, tampered artifact detection (code 1), directory mode batch, CSV mode batch, 2 MiB oversized file pre-read rejection, duplicate ID preservation, bounded chunking, empty input handling, strict mode exit code 2, pure stdout streaming, and partial parse tracking. |
| **`tests/test_dashboard.py`** | **10** | Three-tier routing boundaries, binary threshold, pinned SHA-256 constant, 2 MiB file limit, cached inference, offline URL analysis, telemetry disabling. |
| **`tests/test_data_validation.py`** | **4** | Schema constraints, missing value guards, data types. |
| **`tests/test_error_analysis.py`** | **3** | Confusion matrices, error slicing, score margins. |
| **`tests/test_explainability.py`** | **6** | Exact linear score reconstruction, top feature pushers, family attribution. |
| **`tests/test_features.py`** | **9** | TF-IDF vectorizers, structural features, URL extraction, sparse matrices. |
| **`tests/test_models.py`** | **11** | Model training isolation, scaler invariance, probability calibration. |
| **`tests/test_parser.py`** | **7** | MIME decoding, encoding fallbacks, header parsing. |
| **`tests/test_persistence_latency.py`** | **4** | Bundle serializability, latency benchmarks, memory overhead. |
| **`tests/test_robustness.py`** | **4** | Character obfuscation, noise injection, zero-length edge cases. |
| **`tests/test_splitting_integrity.py`** | **5** | Group split isolation, zero hash overlap between splits. |
| **`tests/test_threshold_optimization.py`** | **18** | Grid search reproducibility, operating point selection. |
| **`tests/test_url_security.py`** | **13** | Static heuristic indicators, IP host regex, shortener detection, offline isolation. |
| **Total Test Suite** | **104 Passed** | **100% Pass Rate across 13 test modules (0 failed)** |

---

## 3. Empirical Performance Benchmarking Results

Empirical latency and throughput benchmarks were measured using `scratch/benchmark_level14_performance.py` over 100 raw emails drawn from the frozen validation split (`data/interim/group_split_manifest.csv`, `split == 'val'`).

### 3.1 Host Hardware & Runtime Environment
- **CPU**: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
- **Cores**: 12 Physical Cores, 14 Logical Cores
- **System Memory**: 15.48 GB RAM
- **Operating System**: Windows 10 (Build 10.0.26200)
- **Python Version**: 3.11.7 (CPython 64-bit)
- **Warm-Up**: 10 sample emails processed before benchmark trials to warm CPU caches and native libraries.

### 3.2 Benchmark Summary Table (100 Sample Emails, 5 Repeated Trials)

| Benchmark Mode | Mean Duration | Median | Min / Max | p95 Duration | Measured Throughput | Per-Email Latency | Target (< 3.0 s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Directory Mode** (100 `.eml` files on disk) | **0.6520 s** | 0.3889 s | 0.3557 s / 1.6618 s | 1.4250 s | **153.4 emails/sec** | **6.52 ms** | **PASSED** (4.6x faster) |
| **Structured CSV Mode** (100 rows in CSV) | **0.2857 s** | 0.2816 s | 0.2720 s / 0.3127 s | 0.3073 s | **350.0 emails/sec** | **2.86 ms** | **PASSED** (10.5x faster) |
| **Pure Pipeline Inference** (In-memory records) | **0.0087 s** | 0.0086 s | 0.0070 s / 0.0110 s | 0.0105 s | **11,441.5 emails/sec** | **0.09 ms** | **N/A** (Baseline) |

### 3.3 Benchmark Observations
- Both Directory Mode and CSV Mode comfortably satisfy the Level 14 performance target (< 3.0 seconds for 100 emails).
- Directory Mode includes disk I/O, MIME header/body decoding, offline URL extraction, feature matrix construction, and CSV serialization. Even with cold file system caches on Trial 1 (1.66 s), the trial remained well under the 3.0 s ceiling.
- Pure linear pipeline scoring runs at over 11,000 emails per second once features are assembled.

---

## 4. Operational Runbook & Documentation Deliverables

A comprehensive production runbook was authored and placed at the repository root:
- **`DEPLOYMENT_RUNBOOK.md`**: Covers system architecture, pre-flight health check commands, batch predict CLI arguments, exit code contract, memory bounds, schemas, Streamlit operation, operational monitoring, and incident response procedures.

---

## 5. Frozen Asset & Manifest Integrity Audit

All model weights, configuration manifests, dataset splits, and evaluation numbers remain completely untouched:

| Asset / Parameter | Value / Digest | Verification Status |
| :--- | :--- | :---: |
| **Model Artifact** | `models/deployment_e11.joblib` | **VERIFIED** |
| **Artifact SHA-256** | `7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06` | **VERIFIED (Unchanged)** |
| **Model Config SHA-256** | `61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2` | **VERIFIED (Unchanged)** |
| **Binary Threshold $\tau$** | `0.4013` | **VERIFIED (Unchanged)** |
| **Three-Tier Bounds** | `[-0.3520, 0.6010]` | **VERIFIED (Unchanged)** |
| **Dataset Splits** | Train: 4,233 / Val: 907 / Test: 906 | **VERIFIED (Unchanged)** |
| **Test Set Evaluation** | Level 7 test split evaluation was **not** rerun | **VERIFIED (Locked)** |

---

## 6. Proposed Git Baseline & Staging Plan (Decision 3 Proposal)

Per **Decision 3 (DEFERRED)**, Git has **not** been initialized, and no files have been staged or committed. Below is the proposed `.gitignore` configuration and baseline staging inventory presented for user review and approval.

### 6.1 Proposed `.gitignore` File

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Testing & Coverage
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/

# IDE & Editor files
.idea/
.vscode/
*.swp
*.swo
*~
.DS_Store

# Large Raw Datasets (Must NOT be tracked in Git)
data/raw/
data/raw/spamassassin/
data/raw/spamassassin/*

# Temporary / Scratch Files
scratch/
tmp/
temp/
*.tmp

# Streamlit Local Cache
.streamlit/cache/

# OS-specific files
Thumbs.db
Desktop.ini
```

### 6.2 Proposed Baseline Staging Inventory (What SHOULD be tracked)

The following core project files should be included in the initial baseline commit:

1. **Source Code (`src/`)**:
   - `src/__init__.py`
   - `src/cli/__init__.py`, `src/cli/healthcheck.py`, `src/cli/batch_predict.py`
   - `src/data/__init__.py`, `src/data/download.py`, `src/data/parser.py`, `src/data/validation.py`
   - `src/features/__init__.py`, `src/features/pipeline.py`, `src/features/structural.py`, `src/features/text.py`, `src/features/url.py`
   - `src/models/__init__.py`, `src/models/evaluate.py`, `src/models/inference.py`, `src/models/train.py`
   - `src/ui/__init__.py`, `src/ui/app.py`
2. **Configuration**:
   - `.streamlit/config.toml`
   - `pyproject.toml` (or `setup.py` / `requirements.txt`)
3. **Tests (`tests/`)**:
   - All 13 test files in `tests/`
4. **Interim Data & Manifests (`data/interim/`)**:
   - `data/interim/group_split_manifest.csv` (split mapping metadata without raw emails)
5. **Model Artifacts & Deployment Manifests (`models/`, `reports/`)**:
   - `models/deployment_e11.joblib` (or tracked via Git LFS if preferred; artifact size is ~332 KB)
   - `models/deployment_manifest.json`
   - `reports/final_model_manifest.json`
   - All audit reports (`reports/level*.md`) and figures (`reports/figures/*.png`)
6. **Documentation**:
   - `DEPLOYMENT_RUNBOOK.md`
   - `README.md`

### 6.3 What Must REMAIN Untracked (Excluded from Git)
- **`data/raw/spamassassin/`**: Contains thousands of raw emails totaling hundreds of megabytes.
- **`scratch/`**: Temporary benchmark data, test `.eml`/`.txt` files, and verification scripts.
- **`.pytest_cache/`**: Pytest runtime cache.
- **`__pycache__/`**: Compiled Python bytecode files.

---

## 7. Final Level 14 Audit Verdict

Level 14 is **COMPLETE, VERIFIED, AND FULLY AUDITED**:
- [x] Headless inference module `src/models/inference.py` created and unified across CLI and Streamlit UI.
- [x] Pre-flight health check `src/cli/healthcheck.py` implemented with pre-deserialization SHA-256 verification and deterministic smoke test.
- [x] Batch inference CLI `src/cli/batch_predict.py` implemented with 2 MiB memory bounds, chunked streaming, 1-to-1 record mapping, and 3-state exit code contract.
- [x] 10 new unit/integration tests in `tests/test_cli.py` passing; full regression suite passes with **104 passed, 0 failed, 1 warning**.
- [x] Empirical throughput verified on 100 validation emails: **0.652 s** (Directory Mode) and **0.286 s** (CSV Mode), surpassing the < 3.0 s target.
- [x] Operational runbook `DEPLOYMENT_RUNBOOK.md` published.
- [x] Model weights, frozen manifest, dataset splits, and thresholds remain 100% frozen.
- [x] Git proposal prepared without executing any git commands.
