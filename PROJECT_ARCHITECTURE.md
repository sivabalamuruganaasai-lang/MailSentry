# Project Architecture & Design Specification
## Competition-Grade Email Spam Classifier

---

## 1. Project Objective

The objective of this project is to build a high-performance, competition-ready, fully explainable, and security-aware Email Spam Classifier system capable of accurately classifying emails into **SPAM**, **HAM** (legitimate email), or **REVIEW** (uncertain/borderline email requiring audit).

The system prioritizes **false-positive minimization** (ensuring legitimate emails are never falsely marked as spam) above raw classification accuracy. It combines statistical NLP (Word & Character TF-IDF), domain/URL security analysis, structural email metadata, regularized linear ML models, threshold calibration, and granular feature-attribution explainability into a modular Python architecture accompanied by an interactive Streamlit dashboard and robust test suite.

---

## 2. Functional Requirements

1. **Email Parsing & Preprocessing**
   - Parse raw email inputs (plain text, MIME, header-rich text, `.eml` format).
   - Separate email into distinct components: `Subject`, `Body`, `Headers`, and `Attachments Metadata`.
   - Strip HTML tags while retaining structural signals (e.g., link text vs target URL mismatches).

2. **Multimodal Feature Extraction**
   - Extract Word-level TF-IDF features ($1$-gram and $2$-gram combinations).
   - Extract Character-level TF-IDF features ($3$-gram to $5$-gram combinations) to capture sub-word obfuscations.
   - Extract URL and Domain Security features (suspicious TLDs, IP hosts, link counts, shortening services, homoglyphs).
   - Extract Stylistic/Structural Metadata (ALL-CAPS ratio in subject/body, punctuation density, currency symbol counts, digit-to-alpha ratios).

3. **Candidate ML Models & Training**
   - Implement Multinomial Naive Bayes (`MultinomialNB`) as a fast probabilistic baseline.
   - Implement Logistic Regression (`LogisticRegression`) with L1/L2 regularization for direct log-odds modeling.
   - Implement Linear Support Vector Classifier (`LinearSVC` calibrated via `CalibratedClassifierCV`) for max-margin separation.
   - Execute 5-fold Stratified Cross-Validation for hyperparameter tuning and model comparison.

4. **Calibrated Decision & Risk Triaging**
   - Output calibrated probability scores $P(\text{Spam} \mid X) \in [0, 1]$.
   - Support a 3-tier classification strategy:
     - $P(\text{Spam} \mid X) < T_{\text{ham}} \implies \mathbf{HAM}$
     - $T_{\text{ham}} \le P(\text{Spam} \mid X) \le T_{\text{spam}} \implies \mathbf{REVIEW}$
     - $P(\text{Spam} \mid X) > T_{\text{spam}} \implies \mathbf{SPAM}$
   - Allow dynamic user adjustment of $T_{\text{ham}}$ and $T_{\text{spam}}$ thresholds.

5. **Explainability & Attribution Engine**
   - Compute top positive (spam-leaning) and top negative (ham-leaning) feature attributions for any given prediction.
   - Highlight exact words, n-grams, and security indicators driving the model's verdict.
   - Provide human-readable security risk badges for suspicious structural elements.

6. **Security & Obfuscation Analysis**
   - Extract URLs and check against structural risk rules (IP host addresses, port numbers, suspicious TLDs, shortened URLs).
   - Detect zero-width characters, homoglyphs (Cyrillic substitution), case-mixing tricks, and leetspeak obfuscations.

7. **Interactive Streamlit UI Dashboard**
   - Single-email Live Analyzer with confidence gauge, text highlighter, and feature contribution breakdown.
   - Model Comparison Hub showing side-by-side ROC, PR, and Confusion Matrix charts.
   - Threshold Calibration Sandbox to explore precision/recall trade-offs live.
   - Batch Processing & Error Audit Log to inspect misclassifications.

8. **Automated Testing & Reproducibility**
   - Comprehensive `pytest` suite covering parser, features, security models, and end-to-end pipeline.
   - Deterministic seed management for 100% reproducible training and evaluation runs.

---

## 3. Non-Functional Requirements

- **Inference Latency**: $\le 50\text{ ms}$ per single email on standard CPU hardware.
- **Dependency Efficiency**: Strict reliance on core Python data science stack (`pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`). Zero reliance on heavy LLM frameworks or GPU-bound deep learning libraries.
- **Data Integrity & Zero Leakage**: Vectorizer state (vocabulary, IDF weights, feature scales) must be fit *strictly* on training splits and applied deterministically to validation/test splits.
- **Maintainability & Modularity**: Clean separation of concerns across single-responsibility modules under `src/`.
- **Reproducibility**: Pinned seed parameters across random generators (`numpy.random`, `train_test_split`, cross-validation splitters).

---

## 4. ML Pipeline Architecture

```
                       ┌─────────────────────────┐
                       │       Raw Email         │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │     Email Parser        │
                       └────────────┬────────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
            ┌────────────────────┐    ┌────────────────────┐
            │ Text Cleaning/NLP  │    │  Security Analysis │
            └──────────┬─────────┘    └──────────┬─────────┘
                       │                         │
                       ▼                         │
            ┌────────────────────┐               │
            │ Feature Extractors │               │
            │ ├── Word TF-IDF    │               │
            │ └── Char TF-IDF    │               │
            └──────────┬─────────┘               │
                       │                         │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Feature Union / Column  │
                       │       Transformer       │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Candidate ML Models     │
                       │ ├── MultinomialNB       │
                       │ ├── LogisticRegression  │
                       │ └── LinearSVC (Calib.) │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Probability Calibration │
                       │    & Threshold Triage   │
                       └────────────┬────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
     ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
     │    HAM      │         │   REVIEW    │         │    SPAM     │
     └─────────────┘         └─────────────┘         └─────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Explainability Engine   │
                       │ & Security Indicators   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  Streamlit Dashboard    │
                       └────────────┬────────────┘
```

---

## 5. NLP Pipeline Architecture

1. **Email Decomposition**:
   - Extraction of `subject` and `body`.
   - MIME structure handling (extract plain text alternative if multi-part; strip HTML tags while extracting href attributes).

2. **Text Normalization**:
   - Unicode NFKD normalization (converts full-width and accented glyphs to standard Latin equivalents).
   - Case folding (lowercasing for word TF-IDF; preserving casing for uppercase ratio features).
   - Collapse excessive whitespace (`\s+` $\to$ single space).

3. **Custom Tokenization**:
   - Regex tokenizer tuned for email/web text: preserves currency symbols (`$`, `€`, `£`), domain patterns, percentage signs, and urgency punctuation (`!`, `?`).

4. **Stopword Strategy**:
   - Retain domain-critical spam/ham indicator tokens (e.g. `no`, `not`, `free`, `win`, `urgent`, `off`).
   - Remove non-informative grammatical stopwords.

---

## 6. Feature Engineering Strategy

The system utilizes a hybrid feature matrix combining statistical text representations with domain-specific security cues.

| Feature Category | Extraction Method | Purpose / Signal Captured |
|---|---|---|
| **Word TF-IDF** | `TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)` | Captures key vocabulary, phrases, and word associations. Sublinear scaling mitigates term repetition spamming. |
| **Character TF-IDF** | `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=15000)` | Captures sub-word obfuscations, typos, letter substitutions (`fr33`, `v!agra`), and character patterns. |
| **URL Security Features** | Custom Regex + `urllib.parse` Extractor | URL count, IP host detection, suspicious TLD count (.xyz, .top, .click, etc.), shortening service link count, anchor text mismatch. |
| **Stylistic Metadata** | Custom Numeric Transformer | Subject ALL-CAPS ratio, Body ALL-CAPS ratio, Exclamation density, Question mark density, Currency symbol count, Digit-to-letter ratio, Character length. |

All feature outputs are concatenated into a single sparse/dense matrix using scikit-learn's `FeatureUnion` or `ColumnTransformer`, preserving feature names for full downstream interpretability.

---

## 7. Candidate Models

1. **Multinomial Naive Bayes (`MultinomialNB`)**
   - *Role*: Fast, probabilistic baseline.
   - *Hyperparameters*: $\alpha \in [0.01, 0.1, 0.5, 1.0]$.
   - *Strengths*: Exceptionally fast, handles sparse text matrices efficiently, robust to irrelevant features.

2. **Logistic Regression (`LogisticRegression`)**
   - *Role*: Primary calibrated linear classifier.
   - *Hyperparameters*: $C \in [0.01, 0.1, 1.0, 10.0]$, penalty $\in [\text{'l1'}, \text{'l2'}]$, solver $\in [\text{'saga'}, \text{'lbfgs'}]$.
   - *Strengths*: Outputs true log-odds $w^T x$, directly providing feature contribution weights for explanation.

3. **Linear Support Vector Classifier (`LinearSVC` + `CalibratedClassifierCV`)**
   - *Role*: Maximum-margin classifier for high-dimensional text separation.
   - *Hyperparameters*: $C \in [0.01, 0.1, 1.0, 10.0]$, loss $\in [\text{'hinge'}, \text{'squared_hinge'}]$.
   - *Calibration*: Wrapped with `CalibratedClassifierCV(method='sigmoid', cv=5)` to obtain calibrated probability estimates $P(\text{Spam} \mid X)$.

---

## 8. Evaluation Strategy

### Validation Protocol
- **Stratified K-Fold Cross-Validation** ($K=5$) on training set to optimize hyperparameters.
- **Holdout Test Set Split**: 70% Train, 15% Validation, 15% Test (stratified by class label).

### Core Metrics
- **False Positive Rate (FPR)**: $FPR = \frac{FP}{FP + TN}$ (*Primary Constraint: Target $< 0.1\%$*).
- **Precision**: $\frac{TP}{TP + FP}$ (*Primary Target: $> 99.0\%$*).
- **Recall**: $\frac{TP}{TP + FN}$.
- **$F_{0.5}$-Score**: $\frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}}$ (weighing Precision $2\times$ higher than Recall).
- **PR-AUC**: Area under the Precision-Recall curve (critical for imbalanced datasets).
- **ROC-AUC**: Area under Receiver Operating Characteristic curve.
- **Confusion Matrix Analysis**: Explicit tracking of $TN, FP, FN, TP$ across threshold variations.

---

## 9. False-Positive Minimization Strategy

Falsely categorizing a legitimate email (HAM) as SPAM is far more detrimental than allowing a SPAM email into the inbox. The system enforces a multi-layered defense against false positives:

1. **Cost-Sensitive Class Weighting**:
   - Apply asymmetric class weights during model training (e.g. `class_weight={0: 5.0, 1: 1.0}`) to heavily penalize HAM misclassifications.
2. **$F_{0.5}$-Score Threshold Optimization**:
   - Select decision thresholds by maximizing $F_{0.5}$ rather than standard accuracy or $F_1$.
3. **Tri-Boundary Classification (Safe / Quarantine / Block)**:
   - Introduce a **REVIEW** zone between $T_{\text{ham}}$ (e.g., $0.15$) and $T_{\text{spam}}$ (e.g., $0.85$).
   - Low-confidence or borderline emails trigger a REVIEW verdict instead of being hard-blocked as SPAM.
4. **Whitelist / Low-Risk Heuristic Overrides**:
   - Known low-risk structural indicators (e.g., clean domain, zero suspicious URLs, verified sender pattern) automatically cap the maximum spam probability score or force a REVIEW classification.

---

## 10. Explainability Strategy

1. **Local Feature Attribution (Per-Email Explanation)**:
   - For Logistic Regression and Linear SVM, calculate term score contributions:
     $$\text{Score}_i = w_i \cdot x_i$$
   - Rank top positive contributions ($\text{Score}_i > 0 \implies \text{Spam Driver}$) and top negative contributions ($\text{Score}_i < 0 \implies \text{Ham Driver}$).
   - Present a visual breakdown of the top 5 Spam indicators and top 5 Ham indicators for each prediction.

2. **Security Signal Audit**:
   - Provide explicit diagnostic badges:
     - ⚠️ `URL Contains IP Host Address`
     - ⚠️ `Suspicious TLD (.xyz detected)`
     - ⚠️ `Anchor Text Mismatch (text != href)`
     - ℹ️ `High Subject Caps Ratio (80%)`

3. **Global Model Interpretability**:
   - Extract and report top global Spam terms and top global Ham terms across the entire trained model vocabulary.

---

## 11. Security-Analysis Strategy

The security engine (`src/security/`) inspects email structural elements independently of body text semantics:

1. **URL & Domain Extractor (`url_analyzer.py`)**:
   - Extract all URLs matching standard web schemas (`http`, `https`).
   - Flag URLs with raw IPv4 or IPv6 addresses as hosts (e.g. `http://192.168.1.50/login`).
   - Verify domain TLDs against a watchlist of high-abuse TLDs (`.xyz`, `.top`, `.click`, `.work`, `.gq`, `.cf`, etc.).
   - Identify shortened URLs via known shortening domains (`bit.ly`, `tinyurl.com`, `t.co`, `ow.ly`, `is.gd`).
   - Detect non-standard port numbers (e.g., `http://example.com:8080/path`).

2. **Obfuscation & Evasion Detector (`obfuscation.py`)**:
   - Detect zero-width unicode characters (`\u200b`, `\u200c`, `\u200d`, `\ufeff`).
   - Detect homoglyph replacement (e.g. Cyrillic `а` swapped for Latin `a`).
   - Measure character entropy and unusual casing patterns.

---

## 12. Robustness-Testing Strategy

To ensure the classifier cannot be easily bypassed, the test suite (`tests/robustness/`) evaluates model stability against synthetic adversarial perturbations:

- **Suite 1: Leetspeak & Character Insertion**: Spacing out letters (`f r e e`), symbol substitution (`v1agr4`, `$pam`).
- **Suite 2: Case Manipulation**: Alternating casing (`cLiCk HeRe Now`).
- **Suite 3: Random Noise Padding**: Appending legitimate text snippets or dictionary words to spam payloads.
- **Suite 4: Obfuscated Link Injection**: Hiding spam destinations inside IP-based or shortened links.

Metrics under perturbation are tracked and compared against baseline metrics to evaluate robustness degradation.

---

## 13. UI / Dashboard Architecture

The dashboard is implemented in Streamlit (`app/app.py`) with 4 primary views:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Email Spam Classifier — Competition Control Center          │
├──────────────┬───────────────────┬───────────────────┬──────────────────┤
│ Live Analyzer│ Model Comparison  │ Threshold Sandbox │ Error Audit Log  │
└──────────────┴───────────────────┴───────────────────┴──────────────────┘
```

1. **Live Email Analyzer**:
   - Input: Subject, Body, Raw Header Text.
   - Output: Large Status Banner (**SPAM** / **HAM** / **REVIEW**), Calibrated Probability Gauge, Top Feature Attribution Bar Chart, Interactive Text Highlight View, Security Risk Badges.

2. **Model Comparison Center**:
   - Side-by-side performance comparison of `MultinomialNB`, `LogisticRegression`, and `LinearSVC`.
   - Comparative ROC Curves, Precision-Recall Curves, Confusion Matrices, and detailed metric summary tables.

3. **Threshold Tuning Sandbox**:
   - Interactive sliders for $T_{\text{ham}}$ and $T_{\text{spam}}$.
   - Live dynamic recalculation of test-set Precision, Recall, FPR, and percentage of emails sent to REVIEW.

4. **Batch Processor & Error Audit Log**:
   - Drag-and-drop CSV/EML batch upload.
   - Filterable view of False Positives and False Negatives for error diagnosis.

---

## 14. Testing Architecture

Testing is implemented with `pytest`:

```
tests/
├── unit/
│   ├── test_parser.py          # Email MIME & component parsing
│   ├── test_cleaner.py         # NLP normalization & tokenization
│   ├── test_security.py        # URL analyzer & obfuscation detection
│   ├── test_features.py        # FeatureUnion shapes & vectorization
│   └── test_models.py          # Candidate model fit, predict & calibration
├── integration/
│   └── test_pipeline.py       # End-to-end classification pipeline
└── robustness/
    └── test_adversarial.py     # Adversarial noise & obfuscation tests
```

---

## 15. Directory Structure

```
email-spam-classifier/
│
├── data/
│   ├── raw/                    # Untouched raw email datasets
│   ├── processed/              # Cleaned & structured datasets
│   └── README.md               # Data dictionary & provenance
│
├── notebooks/                  # Exploratory data analysis notebooks
│
├── src/                        # Core Python package
│   ├── __init__.py
│   ├── data/                   # Data loaders & splitters
│   ├── preprocessing/          # Email parser, NLP cleaner, tokenizer
│   ├── features/               # TF-IDF, security & metadata feature extractors
│   ├── models/                 # Model wrappers, calibration & baseline models
│   ├── evaluation/             # Metrics, confusion matrix & PR-AUC analysis
│   ├── explainability/         # Feature attribution & contribution analyzer
│   ├── security/               # URL, domain & obfuscation analyzers
│   └── utils/                  # Logging, config & persistence helpers
│
├── tests/                      # Automated test suite
│
├── models/                     # Saved trained model artifacts (.joblib)
│
├── reports/                    # Generated metrics & charts
│   ├── figures/
│   └── metrics/
│
├── app/                        # Streamlit dashboard application
│
├── scripts/                    # Automation scripts (train, evaluate)
│
├── configs/                    # Configuration settings (YAML / python)
│
├── requirements.txt            # Project dependencies
├── README.md                   # Project overview & usage instructions
├── PROJECT_ARCHITECTURE.md     # Architectural design specification
└── .gitignore                  # Git ignore rules
```

---

## 16. Data Flow

```
1. Ingestion: Raw Email Text / File (.eml / CSV)
      │
2. Parsing: Extract Subject, Body, Links, Metadata
      │
3. Preprocessing: Unicode NFKD -> Normalization -> Tokenization -> HTML Strip
      │
4. Feature Extraction:
   ├── Word TF-IDF (1-2 grams)
   ├── Char TF-IDF (3-5 grams)
   ├── Security Signals (IP URLs, Suspicious TLDs, Shorteners)
   └── Metadata (ALL-CAPS ratio, Punctuation density, Lengths)
      │
5. Feature Concatenation: ColumnTransformer / FeatureUnion Sparse Matrix
      │
6. Model Inference: Score with Trained Classifier -> Probability P(Spam|X)
      │
7. Tri-Boundary Decision Logic:
   ├── P < T_ham               --> HAM
   ├── T_ham <= P <= T_spam    --> REVIEW
   └── P > T_spam              --> SPAM
      │
8. Explanation & Security Attribution: Compute Top Terms + Security Badges
      │
9. UI / Report Display: Streamlit Dashboard / Output Payload
```

---

## 17. Model Lifecycle & Reproducibility

1. **Configuration**: Centralized configuration management (`configs/default_config.py` / `configs/config.yaml`) defining random seeds, feature limits, hyperparameter grids, and paths.
2. **Deterministic Execution**:
   - `numpy.random.seed(42)`
   - `random.seed(42)`
   - Pinned `random_state=42` across all scikit-learn splitters and estimators.
3. **Artifact Serialization**: Trained pipelines are saved using `joblib` into `models/` along with metadata JSON files containing:
   - Training timestamp
   - Feature names array
   - Hyperparameter settings
   - Test set metrics (Precision, Recall, FPR, PR-AUC)
   - Decision thresholds ($T_{\text{ham}}, T_{\text{spam}}$)

---

## 18. Development Phases

```
Phase 1: Architecture Definition & Workspace Setup (COMPLETE)
Phase 2: Email Ingestion, Parsing & NLP Preprocessing Engine
Phase 3: Feature Engineering (Word TF-IDF, Char TF-IDF, Security & Metadata)
Phase 4: Candidate Model Implementation & Cross-Validation Benchmarking
Phase 5: Threshold Optimization, Calibration & False Positive Minimization
Phase 6: Explainability Engine & Security Analysis Integration
Phase 7: Streamlit Interactive Dashboard & Visualizations
Phase 8: Robustness Testing, Automated Test Suite & Documentation
```

---

## 19. Risks and Mitigation Strategies

| Risk | Potential Impact | Mitigation Strategy |
|---|---|---|
| **Data Leakage** | Optimistic performance estimates during validation | Strict use of scikit-learn `Pipeline` objects; vectorizers fit *only* on training fold data. |
| **High False-Positive Rate** | Legitimate emails sent to SPAM | Cost-sensitive class weighting, $F_{0.5}$-score optimization, and tri-boundary REVIEW tier. |
| **Evasion via Obfuscation** | Spam bypasses word TF-IDF | Dual character n-gram TF-IDF + Unicode NFKD normalization + security structural indicators. |
| **Uncalibrated Probabilities** | Misleading confidence scores | Probability calibration via `CalibratedClassifierCV` for SVM and sigmoid scaling for Logistic Regression. |
| **Overfitting on Keywords** | Model fails on novel spam campaigns | Feature regularization ($L_1 / L_2$ penalties) and character-level features. |

---

## 20. What Must NOT Be Done

- **Do NOT** download random datasets or use unverified third-party APIs during architecture setup.
- **Do NOT** fabricate or hardcode model performance metrics.
- **Do NOT** rely on deep learning, LLMs, or heavy GPU frameworks (e.g., PyTorch, HuggingFace, OpenAI).
- **Do NOT** reduce the system to a simple keyword matching list.
- **Do NOT** fit vectorizers or feature scalers before splitting dataset into Train/Test partitions.
- **Do NOT** swallow runtime errors or handle exceptions silently without log emission.

---

## 21. Definition of Done (DoD)

- [x] Complete, detailed `PROJECT_ARCHITECTURE.md` written and validated.
- [ ] Directory structure initialized cleanly.
- [ ] Email parser and NLP preprocessing engine verified via unit tests.
- [ ] Multimodal feature extraction pipeline implemented without data leakage.
- [ ] Candidate models (`MultinomialNB`, `LogisticRegression`, `LinearSVC`) trained, calibrated, and benchmarked using Stratified 5-Fold Cross-Validation.
- [ ] $F_{0.5}$ metric optimization and tri-boundary thresholding ($T_{\text{ham}}, T_{\text{spam}}$) validated.
- [ ] Explainability engine generating term attributions and security risk badges.
- [ ] Security module identifying suspicious URLs, IP hosts, and homoglyphs.
- [ ] Streamlit dashboard operational with all 4 functional tabs.
- [ ] Automated test suite (`pytest`) passing with high coverage.
- [ ] Model artifacts saved reproducibly with complete metadata logs.
