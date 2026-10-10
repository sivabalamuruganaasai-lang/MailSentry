# Deployment Runbook: E11 Production Email Spam Classification Service

This runbook defines the operational procedures, deployment verification protocols, command-line interfaces, service monitoring guidelines, and troubleshooting paths for the frozen **E11** Spam Classification Engine (`models/deployment_e11.joblib`).

---

## 1. Architecture & Component Overview

The classification service is designed around a single canonical, headless inference core shared across both interactive and automated interfaces.

```
+-------------------------------------------------------------------------+
|                              Entrypoints                                |
|   +----------------------------+     +-------------------------------+  |
|   |   Streamlit Interactive    |     |      Batch Predict CLI        |  |
|   |         Web UI             |     |   (Directory / CSV Streaming) |  |
|   |      (src/ui/app.py)       |     |   (src/cli/batch_predict.py)  |  |
|   +-------------+--------------+     +---------------+---------------+  |
|                 |                                    |                  |
|                 +-----------------+------------------+                  |
|                                   |                                     |
|                                   v                                     |
|                 +------------------------------------+                  |
|                 |    Canonical Headless Inference    |                  |
|                 |     (src/models/inference.py)      |                  |
|                 +-----------------+------------------+                  |
|                                   |                                     |
|         +-------------------------+-------------------------+           |
|         v                                                   v           |
|  +-----------------------------+             +-----------------------+  |
|  | Cryptographic SHA-256 Gate  |             | Feature Extraction &  |  |
|  | (Pre-Deserialization Check) |             | URL Security Analyzer |  |
|  +--------------+--------------+             +-----------+-----------+  |
|                 |                                        |              |
|                 v                                        v              |
|  +-----------------------------+             +-----------------------+  |
|  |   models/deployment_e11     |             |  Frozen Decision Gate |  |
|  |     (Joblib Pipeline)       |             |  tau=0.4013, 3-Tier   |  |
|  +-----------------------------+             +-----------------------+  |
+-------------------------------------------------------------------------+
```

### Key Modules:
- **`src.models.inference`**: Canonical headless inference engine. Houses the pinned artifact SHA-256 hash, safe model loading, binary threshold evaluation ($\tau = 0.4013$), three-tier routing ($[-0.3520, 0.6010]$), and feature mapping.
- **`src.cli.healthcheck`**: Zero-dependency pre-flight deployment diagnostic tool.
- **`src.cli.batch_predict`**: High-throughput chunked batch inference CLI supporting raw `.eml`/`.txt` directories and tabular CSV streams.
- **`src.ui.app`**: Streamlit interactive dashboard importing from `src.models.inference`.

---

## 2. Environment Setup & Offline Dataset Acquisition

### 2.1 Verified Python & Package Dependencies
The E11 model and its pre-flight validation smoke tests have been verified under **Python 3.11.7** with the following pinned environment specification:
- `scikit-learn == 1.9.1`
- `joblib == 1.6.0`
- `numpy == 2.2.6`
- `pandas == 3.0.5`
- `streamlit == 1.65.0`
- `pytest == 9.1.1`

Install all required runtime dependencies via:
```bash
pip install -r requirements.txt
```

### 2.2 Dataset Acquisition & Pipeline Reproduction (Optional)
The production deployment artifact (`models/deployment_e11.joblib`) is fully self-contained. Downloading the raw email corpus is **not** required for batch prediction or running the Streamlit dashboard.

If reproducing the data preparation and validation pipeline from scratch:
1. **Download Raw SpamAssassin Corpus**:
   ```bash
   python -m src.data.acquire
   ```
   *Downloads the 5 official Apache SpamAssassin tar.bz2 archives into `data/raw/spamassassin/` and extracts raw emails.*
   > [!IMPORTANT]
   > All downloaded emails in `data/raw/` are strictly excluded from version control via `.gitignore` and must remain local.

2. **Forensic Validation & Deduplication**:
   ```bash
   python -m src.data.validate
   ```
   *Generates sanitized manifests and runs near-duplicate audits without modifying the frozen model or split boundaries.*

---

## 3. Pre-Flight Deployment Health Check

Before starting batch inference jobs or routing production traffic to the Streamlit dashboard, run the automated health check.

### 3.1 Execution Commands

Standard ASCII human-readable output (stdout):
```bash
python -m src.cli.healthcheck
# or directly:
python src/cli/healthcheck.py
```

Machine-readable JSON output for automated orchestrators (Kubernetes liveness/readiness, CI/CD):
```bash
python -m src.cli.healthcheck --json
```

### 3.2 Health Check Audit Steps
1. **Python Environment Verification**: Validates Python $\ge$ 3.10 and presence of core dependencies (`joblib`, `sklearn`, `numpy`, `pandas`).
2. **Pre-Deserialization Cryptographic Verification**: Computes the SHA-256 digest of `models/deployment_e11.joblib` *prior* to unpickling. If the digest fails to match the pinned hash (`7752ac87...`), execution halts immediately to prevent insecure deserialization.
3. **Pipeline Structure Inspection**: Verifies scikit-learn pipeline integrity, step counts, and vocabulary state.
4. **Deterministic Reference Inference Smoke Test**:
   - Uses the canonical deterministic benchmark fixture (*Legitimate Team Sync*).
   - Expected score: `+0.020524`.
   - Verified within a numeric tolerance of $\pm 10^{-4}$.
   - Classifies as: `HAM`, routing tier: `review`.

### 3.3 Health Check Exit Codes
- `0`: System healthy, artifact verified, smoke test passed.
- `1`: System unhealthy (missing file, hash mismatch, dependency issue, or smoke test divergence).

---

## 4. Batch Inference CLI (`batch_predict.py`)

The batch inference CLI processes email collections in bounded chunks to ensure bounded memory consumption and maximum throughput.

### 4.1 Command-Line Syntax

```bash
python -m src.cli.batch_predict [MODE_ARGUMENT] [OPTIONS]
```

#### Primary Mode Arguments (Mutually Exclusive):
- `--input-dir <DIR_PATH>`: Path to directory containing raw `.eml` or `.txt` email files.
- `--input-csv <CSV_PATH>`: Path to CSV file containing email records.

#### Options:
- `--output-file <FILE_PATH>`: Target output file path. If omitted, pure records stream to `stdout`.
- `--output-format {csv,jsonl}`: Output serialization format (default: `csv`).
- `--chunk-size <INT>`: Batch processing chunk size (default: `250`).
- `--strict`: Enable strict exit status. If any individual record fails parsing or scoring, exits with code `2`.
- `--quiet`: Suppress diagnostic progress bars and completion summaries on `stderr`.

---

## 5. Exit Code Contract

The batch CLI conforms to a strict Unix-style exit code contract:

| Exit Code | Classification | Condition |
| :---: | :--- | :--- |
| **`0`** | **Success / Clean Completion** | Batch completed successfully. In non-strict mode (default), individual per-record failures are recorded in the output file without failing the job. |
| **`1`** | **Fatal Failure** | Fatal runtime failure: missing input files, invalid CLI arguments, SHA-256 hash mismatch, dependency missing, unwriteable output destination, or catastrophic crash. |
| **`2`** | **Partial Record Failure (Strict Mode)** | Batch completed with one or more individual record failures when `--strict` is enabled. |

---

## 6. Streaming, Memory Bounds & Record Integrity

### 6.1 Enforced Memory Bounds
- **Pre-Read File Size Gate**: Every individual `.eml` or `.txt` file is inspected using `os.path.getsize()` *prior* to reading into memory. Files exceeding **2 MiB (2,097,152 bytes)** are rejected immediately without being loaded into RAM.
- **CSV Chunking**: CSV files are read via bounded chunks (`chunksize=250`) to keep working memory below 100 MB regardless of file size.
- **CSV Row Size Gate**: Each CSV row's string payload is validated against the 2 MiB boundary.

### 6.2 1-to-1 Input-to-Output Mapping
Every input record yields **exactly one** output record in identical order. Records are never dropped silently:
- If a file is oversized (>2 MiB): Output record status is marked `failed`, error message notes `File exceeds 2 MiB limit`.
- If an email is corrupt/unparseable: Output record status is marked `failed` with diagnostic error recorded.
- If an email has partial headers/body: Status is marked `partial`, and scoring proceeds on available text.

### 6.3 Parser Status Taxonomy
- `success`: Headers and body parsed normally without error.
- `partial`: Fallback decoding or partial structure parsing occurred (never misrepresented as `success`).
- `failed`: Extraction failed completely or record was oversized.

---

## 7. Input & Output Schemas

### 7.1 CSV Input Schema
When using `--input-csv`, the input file must provide at least one text column. Expected headers:
```csv
id,subject,body,urls
mail_001,"Weekly Update","Please review the attached project schedule.","http://internal.corp"
```
*Note*: Column headers are flexible (`id`/`email_id`/`message_id`, `subject`, `body`/`text`/`content`, `urls`/`url_list`).

### 7.2 Output Schema (CSV and JSONL)

Every output row contains the following schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` | Input identifier (filename or CSV row ID) |
| `status` | `str` | Record processing status (`success`, `partial`, `failed`) |
| `parse_status` | `str` | Email parsing diagnostic (`success`, `partial`, `failed`) |
| `score` | `float` | Raw decision function score ($s \in \mathbb{R}$) |
| `is_spam` | `int` | Binary classification label (`1` if $s \ge 0.4013$, else `0`) |
| `tier` | `str` | Three-tier routing (`deliver`, `review`, `quarantine`) |
| `url_risk` | `str` | URL security risk level (`None`, `Low`, `Medium`, `High`) |
| `flagged_urls_count` | `int` | Count of high-risk or suspicious URLs detected |
| `error` | `str` | Diagnostic error message (or empty string if success) |

---

## 8. Streamlit Web UI Operation

### 8.1 Starting the Dashboard
To start the interactive UI:
```bash
streamlit run src/ui/app.py --server.port 8501
```

### 8.2 Safety Guarantees
- The UI imports the model, constants, and feature extraction from `src.models.inference`.
- Pre-deserialization SHA-256 verification is enforced upon startup.
- The UI enforces identical binary thresholds ($\tau = 0.4013$) and routing boundaries ($[-0.3520, 0.6010]$).
- All URL security inspections run completely offline without outbound network calls.

---

## 9. Operational Monitoring & Production Alerts

### 9.1 Key Metrics to Track
1. **Review Tier Ratio**: Proportion of emails classified as `review` (routing score between $-0.3520$ and $+0.6010$). Baseline expectation: ~7%–10%. An increase above 20% indicates potential model distribution drift.
2. **Quarantine Rate**: Proportion of emails sent to `quarantine` ($s \ge 0.6010$). Baseline: matches expected spam volume (~20%–30%).
3. **High-Risk URL Flag Rate**: Track spikes in `url_risk == "High"`.
4. **Input Parse Failure Rate**: Alert if `parse_status == "failed"` exceeds 0.5% of total traffic.

### 9.2 Standard Logging Separation
- **`stderr`**: All diagnostic logs, progress indicators, warnings, and error traces.
- **`stdout`**: Reserved exclusively for valid CSV or JSONL record output when `--output-file` is omitted. Piping `python -m src.cli.batch_predict ... > results.csv` produces 100% valid CSV data.

---

## 10. Troubleshooting & Incident Response

### Problem 1: Artifact Integrity Verification Failure (`Exit Code 1`)
- **Symptom**: `ERROR: Model artifact SHA-256 mismatch!`
- **Cause**: The file `models/deployment_e11.joblib` was corrupted, tampered with, or replaced.
- **Action**: Do NOT bypass verification. Restore `models/deployment_e11.joblib` from verified backup matching expected hash:
  `7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06`.

### Problem 2: Batch Process Exits with Code 2
- **Symptom**: CLI exits with code 2 under `--strict` mode.
- **Cause**: One or more input files exceeded the 2 MiB limit, had corrupt MIME encoding, or failed processing.
- **Action**: Inspect the generated output CSV/JSONL file where `status == "failed"`. The `error` column identifies the exact failure reason per record.

### Problem 3: Smoke Test Numeric Score Divergence
- **Symptom**: Health check fails with `Smoke test score divergence`.
- **Cause**: Scikit-learn or NumPy version mismatch altering floating-point arithmetic.
- **Action**: Check virtual environment packages against `requirements.txt`. Expected deployment versions: `scikit-learn==1.9.1`, `numpy==2.2.6`, `joblib==1.6.0`, `pandas==3.0.5`.
