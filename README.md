# Competition-Grade Email Spam Classifier

An end-to-end, competition-ready machine learning system for email spam classification, prioritizing **false-positive minimization**, **explainability**, **security signal analysis**, and **model calibration**.

---

## 🚀 Quick Start & Reproducibility Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed.

Install pinned dependencies:
```bash
pip install -r requirements.txt
```

---

### 2. Dataset Acquisition & Forensic Validation (Level 3)

The system automatically downloads the official **Apache SpamAssassin Public Corpus** (5 archives, 6,046 raw emails), parses raw MIME components, computes SHA-256 hashes, performs deduplication and near-duplicate audits, inspects leakage headers, and generates the processed dataset manifest.

#### Step A: Acquire Raw Corpus
```bash
python -m src.data.acquire
```
*Downloads raw tar archives into `data/raw/spamassassin/` and extracts raw `.eml` files into `easy_ham`, `easy_ham_2`, `hard_ham`, `spam`, and `spam_2`.*

#### Step B: Validate Corpus & Build Dataset Manifest
```bash
python -m src.data.validate
```
*Parses all emails, calculates raw & normalized SHA-256 hashes, detects near-duplicates, audits `X-Spam-*` leakage headers, and writes `data/processed/dataset_manifest.csv` and `reports/dataset_quality_report.md`.*

#### Step C: Run Automated Test Suite
```bash
python -m pytest tests/
```
*Executes unit tests for standard-library email parsing, HTML link extraction, malformed MIME handling, deterministic hashing, and leakage header detection.*

---

## 📂 Project Architecture & Directory Layout

```
email-spam-classifier/
│
├── data/
│   ├── raw/
│   │   └── spamassassin/       # Untouched raw email corpus (6,046 files)
│   ├── interim/                # Interim CSV metadata & near-duplicate logs
│   └── processed/
│       └── dataset_manifest.csv # Final sanitized email dataset manifest
│
├── src/                        # Core Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── acquire.py          # Download & archive extractor
│   │   ├── parser.py           # Robust RFC 822 email & URL parser
│   │   ├── inventory.py        # Metadata scanner & header leakage auditor
│   │   ├── deduplicate.py      # Multi-level SHA-256 & Jaccard deduplication
│   │   └── validate.py         # Pipeline orchestrator & manifest generator
│   ├── preprocessing/          # NLP text cleaners (Level 4)
│   ├── features/               # TF-IDF & Security Extractors (Level 4)
│   ├── models/                 # Model trainers & calibration (Level 5)
│   ├── evaluation/             # Metrics & PR-AUC analysis (Level 5)
│   ├── explainability/         # Feature attribution engine (Level 6)
│   └── security/               # Domain & URL security engine (Level 6)
│
├── tests/
│   ├── test_parser.py          # Unit tests for MIME & URL parsing
│   └── test_data_validation.py # Unit tests for label & hash validation
│
├── reports/
│   └── dataset_quality_report.md # 16-point dataset quality & forensic report
│
├── models/                     # Saved joblib pipelines
├── configs/                    # YAML / Python configuration
├── requirements.txt            # Dependency matrix
├── PROJECT_ARCHITECTURE.md     # Architecture specification
├── DATASET_RESEARCH.md          # Dataset intelligence & benchmark evaluation
└── README.md                   # Usage & reproduction guide
```

---

## 📊 Dataset Audit Summary (SpamAssassin Public Corpus)

- **Total Emails Analyzed**: **6,046**
- **HAM (Legitimate)**: **4,150** (68.64%)
- **SPAM**: **1,896** (31.36%)
- **Parse Success Rate**: **100%** (0 unhandled exceptions)
- **Exact Raw SHA-256 Duplicates**: **0**
- **Normalized Content Duplicates**: **119**
- **Cross-Class Label Conflicts**: **0**
- **Near-Duplicate Pairs ($\ge 0.85$ Jaccard)**: **307**
- **Leakage Headers Audited**: `X-Spam-Status`, `X-Spam-Flag`, `X-Spam-Checker-Version` (*explicitly excluded from feature matrix*).

---

## 🤖 Candidate Model Benchmarking (Level 5)

### 1. Class & Problem Definitions
- **Positive Class (1)**: **SPAM**
- **Negative Class (0)**: **HAM**
- **False Positive (FP)**: HAM incorrectly classified as SPAM (*Primary operational risk: user losing legitimate email*).
- **False Positive Rate (FPR)**: $\frac{FP}{FP + TN}$
- **True Positive Rate (TPR / Recall)**: $\frac{TP}{TP + FN}$

### 2. Test Set Protection & Split Methodology
To guarantee zero data leakage and honest model evaluation:
- **TRAIN Set (70%)**: Used exclusively for model fitting and feature extractor fitting.
- **VALIDATION Set (15%)**: Used exclusively for candidate model selection and hyperparameter configuration ranking.
- **TEST Set (15%)**: **Strictly protected and untouched holdout**. Zero test samples were used for fitting, feature extraction, hyperparameter tuning, or candidate selection.

### 3. Baseline Experiment Matrix (E01 - E11)
Run the automated candidate model benchmarking suite:
```bash
python -m src.models.run_benchmark
```

### 4. Ranked Model Benchmarking Results (Validation Set)

| Rank | Exp ID | Model | Features | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **E11** | LinearSVC | `word+structural+url` | **0.9846** | **0.9719** | 0.9788 | **0.9754** | **0.0128** | 0.9983 | 0.9959 |
| **2** | **E03** | LinearSVC | `word` | 0.9824 | **0.9619** | **0.9823** | **0.9720** | **0.0176** | 0.9986 | 0.9969 |
| **3** | **E08** | LogisticRegression | `word+char` | 0.9802 | **0.9617** | 0.9753 | **0.9684** | **0.0176** | 0.9984 | 0.9965 |
| **4** | **E09** | LinearSVC | `word+char` | 0.9802 | **0.9617** | 0.9753 | **0.9684** | **0.0176** | 0.9984 | 0.9963 |
| **5** | **E10** | LinearSVC | `word+structural` | 0.9791 | **0.9615** | 0.9717 | **0.9666** | **0.0176** | 0.9984 | 0.9965 |

### 5. Reproducibility & Group-Aware Cross-Validation
- **Random Seed**: `random_state = 42` enforced across all feature extractors and models.
- **Group Isolation**: Connected components of exact and near-duplicates are strictly grouped. `StratifiedGroupKFold(n_splits=5)` is used on Training Data to ensure zero group crossing between folds.
- **Machine-Readable Report**: `reports/model_benchmark.csv`
- **Detailed Markdown Report**: `reports/model_benchmark.md`
- **Confusion Matrices**: `reports/figures/confusion_matrix_<EXP_ID>.png`
- **Validation Error Records**: `reports/val_predictions_<EXP_ID>.csv`
