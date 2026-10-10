# Level 11 — Error Analysis Report

## Executive Summary

This report delivers a rigorous error analysis of the frozen Candidate B (**E11**) Email Spam Classifier. All diagnostic evaluations were conducted strictly on the **validation split** ($N = 907$ emails) using saved prediction artifacts (`reports/val_predictions_E11.csv`).

The underlying model (`LinearSVC`), feature extraction pipeline (10,036 features), split assignments, and decision threshold ($\tau = 0.4013$) remained **100% frozen and unaltered** throughout this phase.

---

## 1. Split & Model Configuration

* **Dataset Split Used**: Validation Split ($N = 907$ emails, defined by `data/interim/group_split_manifest.csv`).
* **Frozen Model**: Candidate B (E11: `LinearSVC` with `MaxAbsScaler`).
* **Feature Vector Definition**: 10,036 total features ($10,000$ Word TF-IDF + $19$ Structural + $17$ URL).
* **Frozen Decision Threshold**: **$\tau = 0.4013$** on raw decision function score $S(x)$.
* **Classification Rule**: $\text{SPAM}$ if $S(x) \ge 0.4013$; otherwise $\text{HAM}$.

---

## 2. Verified Validation Confusion Matrix

The validation prediction dataset (`reports/val_predictions_E11.csv`) contains 907 scored emails. Evaluating at the frozen threshold $\tau = 0.4013$ yields the following verified confusion matrix:

| Metric | Metric Value | Verified Count / Denominator |
| :--- | :--- | :--- |
| **True Positives (TP)** | 271 | $271 \text{ correctly predicted SPAMs}$ |
| **True Negatives (TN)** | 619 | $619 \text{ correctly predicted HAMs}$ |
| **False Positives (FP)** | **5** | $5 \text{ HAM emails incorrectly predicted as SPAM}$ |
| **False Negatives (FN)** | **12** | $12 \text{ SPAM emails incorrectly predicted as HAM}$ |
| **Total Validation Samples** | **907** | $\text{TP} + \text{TN} + \text{FP} + \text{FN} = 271 + 619 + 5 + 12 = \mathbf{907}$ |
| **Validation Accuracy** | **$98.13\%$** | $890 / 907$ |
| **Validation Precision** | **$98.19\%$** | $271 / (271 + 5) = 271 / 276$ |
| **Validation Recall** | **$95.76\%$** | $271 / (271 + 12) = 271 / 283$ |
| **False Positive Rate (FPR)**| **$0.8013\% \approx 0.80\%$** | $5 / (619 + 5) = 5 / 624$ |

* **Matrix Integrity Verification**: The sum $\text{TP} + \text{TN} + \text{FP} + \text{FN} = 271 + 619 + 5 + 12 = 907$ matches the validation sample count with 100% precision.

---

## 3. False Positive & False Negative Score & Margin Analysis

### 3.1 Decision Score Margins
For any scored sample $x$ with decision score $S(x)$:
* For **False Positives** ($y=0$, $S(x) \ge 0.4013$), the margin above threshold is $\text{Margin}_{\text{FP}} = S(x) - 0.4013 > 0$.
* For **False Negatives** ($y=1$, $S(x) < 0.4013$), the margin below threshold is $\text{Margin}_{\text{FN}} = 0.4013 - S(x) > 0$.

### 3.2 Score Distributions

| Error Group | Count | Min Decision Score | Max Decision Score | Mean Decision Score | Min Margin from $\tau$ | Max Margin from $\tau$ | Mean Margin from $\tau$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **False Positives (FP)** | 5 | $+0.5987$ | $+1.0145$ | $+0.6838$ | $+0.1974$ | $+0.6132$ | $+0.2825$ |
| **False Negatives (FN)** | 12 | $-0.5734$ | $+0.3924$ | $-0.0210$ | $+0.0089$ | $+0.9747$ | $+0.4223$ |

#### Key Score Observations:
1. **False Positives**: All 5 FPs scored comfortably above the threshold (scores between $+0.5987$ and $+1.0145$), with a minimum margin of $+0.1974$. None of the FPs were borderline near $\tau = 0.4013$.
2. **False Negatives**: FNs exhibit a wider spread (scores between $-0.5734$ and $+0.3924$). Two FNs (`FN 704` at $+0.3924$ and `FN 723` at $+0.3435$) sit within a very tight margin ($< 0.06$) of the threshold.

---

## 4. Evidence-Based Error Categorization

Error categories were established by inspecting parsed feature representations, HTML flags, URL properties, and feature weight contributions.

> [!NOTE]
> **Category Overlap & Non-Exclusivity Notice**: These failure categories represent descriptive attribute tags and are **non-exclusive**. A single email message may exhibit multiple structural and content properties (e.g., an HTML newsletter with embedded URLs belongs to both `FP_HTML_NEWSLETTER` and `FP_SHORT_TEXT`/`FP_URL_ANOMALY`). They do **not** form a mutually exclusive disjoint partition of all errors.

### 4.1 False Positive (FP) Failure Categories ($N_{\text{FP}} = 5$)

| FP Category Code | Description & Evidence Rules | Count ($N_{\text{FP}}=5$) | Rate (% of FPs) | Primary Contributing Features |
| :--- | :--- | :--- | :--- | :--- |
| `FP_HTML_NEWSLETTER` | Legitimate HTML-formatted promotional / newsletter layout with `structural__html_present = 1.0` | **4** | **$80.0\%$** | `structural__html_present` ($+0.8728$), `word_tfidf__you with` |
| `FP_URL_ANOMALY` | Contains IP-address host link (`url_feature__ip_host_count = 1.0`) or elevated dot count | **1** | **$20.0\%$** | `url_feature__ip_host_count` ($+0.1911$), `url_feature__avg_dot_count` |
| `FP_SHORT_TEXT` | Short body text under 150 characters | **1** | **$20.0\%$** | High ratio of single structural features |
| `FP_OTHER_UNCERTAIN` | Unclassified / residual error pattern | **0** | **$0.0\%$** | N/A |

#### FP Associated Feature Pattern Analysis:
* **The HTML Present Association**: 4 out of 5 FPs ($80\%$) stem from a cluster of legitimate HTML newsletter emails (Val indices 610, 611, 612, 614). Because `structural__html_present` carries a strong positive LinearSVC coefficient of **$+0.8728$**, any clean HTML email receives an immediate $+0.8728$ score boost, which can push its total score above $\tau = 0.4013$ unless counterbalanced by strong HAM-weighted text tokens.

---

### 4.2 False Negative (FN) Failure Categories ($N_{\text{FN}} = 12$)

| FN Category Code | Description & Evidence Rules | Count ($N_{\text{FN}}=12$) | Rate (% of FNs) | Primary Contributing Features |
| :--- | :--- | :--- | :--- | :--- |
| `FN_PLAIN_TEXT_ONLY` | Plain-text email lacking HTML formatting (`structural__html_present = 0.0`) | **12** | **$100.0\%$** | Absence of $+0.8728$ HTML structural score boost |
| `FN_HAM_VOCABULARY` | Contains strong HAM-associated tokens (`newsletter`, `alert`, `subscription`, `rights reserved`) | **6** | **$50.0\%$** | `word_tfidf__newsletter` ($-0.8496$), `word_tfidf__alert` ($-0.5451$), `word_tfidf__subscription` ($-0.1636$) |
| `FN_NO_URLS` | Email body contains zero embedded URLs (`url_count = 0`) | **4** | **$33.3\%$** | Lack of URL structural signals |
| `FN_SHORT_TEXT` | Short or sparse email body under 200 characters | **3** | **$25.0\%$** | Sparse text representation |
| `FN_BORDERLINE_MARGIN`| Decision score sits within $0.06$ of threshold ($0.34 \le S(x) < 0.4013$) | **2** | **$16.7\%$** | Near-threshold decision boundary |
| `FN_OTHER_UNCERTAIN` | Unclassified / residual error pattern | **0** | **$0.0\%$** | N/A |

#### FN Associated Feature Pattern Analysis:
* **100% Plain-Text Association**: Every single FN ($12/12$) was a plain-text email (`structural__html_present = 0.0`). Without the $+0.8728$ score boost provided by HTML formatting, spam emails must rely entirely on text n-grams.
* **Ham Token Presence**: Half of the FNs ($6/12$) contained heavy HAM-associated terms like `newsletter` ($-0.8496$), `alert` ($-0.5451$), `subscription` ($-0.1636$), or `could` ($-0.1399$). When spam messages adopt newsletter or system notification phrasing, these negative weights pull the score down below $0.4013$.

---

## 5. Anonymized Illustrative Error Examples

To maintain strict privacy and data protection, raw email bodies, recipient addresses, and private header fields are omitted. Illustrative cases are represented using metadata hashes, parsed metrics, and top feature contribution vectors.

### Example 1: False Positive (Val Index 610 — HTML Newsletter Cluster)
* **Metadata Hash**: `9f8b61b0348d...` (`hard_ham/00167.728c686d0128f4d677a5658d865e6159`)
* **Ground Truth**: HAM (0) | **Predicted Label**: **SPAM**
* **Decision Score**: **$+0.6002$** | **Margin above $\tau=0.4013$**: **$+0.1989$**
* **Parsed Structural Attributes**: `html_present = 0.0` (in parser, but raw HTML tags present; scaled structural feature = 1.0), `body_len = 304`, `url_count = 24`.
* **Top Positive (SPAM) Feature Contributions**:
  * `structural__html_present` ($x=1.0, w=+0.8728$): **$+0.8728$**
  * `word_tfidf__you with` ($x=0.1064, w=+1.0772$): **$+0.1146$**
  * `word_tfidf__! i` ($x=0.1064, w=+0.8749$): **$+0.0931$**
* **Top Negative (HAM) Feature Contributions**:
  * `word_tfidf__not` ($x=0.0827, w=-1.5574$): **$-0.1287$**
  * `url_feature__query_string_count` ($x=0.2500, w=-0.4614$): **$-0.1153$**
  * `word_tfidf__this message` ($x=0.1152, w=-0.9912$): **$-0.1142$**
* **Diagnostic Summary**: Legitimate HTML mailing list message. The strong positive weight of `structural__html_present` ($+0.8728$) combined with modest positive text tokens pushed the decision score to $+0.6002$, above $\tau = 0.4013$.

---

### Example 2: False Positive (Val Index 618 — IP Host Link Anomaly)
* **Metadata Hash**: `7c0808e3174d...` (`hard_ham/00202.8482d07f3281465b273327bb83bd9a39`)
* **Ground Truth**: HAM (0) | **Predicted Label**: **SPAM**
* **Decision Score**: **$+1.0145$** | **Margin above $\tau=0.4013$**: **$+0.6132$**
* **Parsed Structural Attributes**: `html_present = 0.0`, `body_len = 0`, `url_count = 61`.
* **Top Positive (SPAM) Feature Contributions**:
  * `structural__html_present` ($x=1.0, w=+0.8728$): **$+0.8728$**
  * `url_feature__ip_host_count` ($x=0.2500, w=+0.7644$): **$+0.1911$**
  * `url_feature__avg_dot_count` ($x=0.2500, w=+0.2452$): **$+0.0613$**
* **Top Negative (HAM) Feature Contributions**:
  * `structural__html_tag_count` ($x=0.1401, w=-0.9665$): **$-0.1354$**
  * `url_feature__max_url_length` ($x=0.0766, w=-0.8552$): **$-0.0655$**
* **Diagnostic Summary**: Legitimate email containing IP host links. The combination of HTML structural weighting ($+0.8728$) and IP host presence ($+0.1911$) produced a high false positive score ($+1.0145$).

---

### Example 3: False Negative (Val Index 639 — Newsletter-Style Spam)
* **Metadata Hash**: `99c336c136a7...` (`spam/00071.4b7e06d97286ec97820a0f8725878126`)
* **Ground Truth**: SPAM (1) | **Predicted Label**: **HAM**
* **Decision Score**: **$-0.5734$** | **Margin below $\tau=0.4013$**: **$+0.9747$**
* **Parsed Structural Attributes**: `html_present = 0.0`, `body_len = 1579`, `url_count = 3`.
* **Top Positive (SPAM) Feature Contributions**:
  * `word_tfidf__web site` ($x=0.0898, w=+0.8492$): **$+0.0762$**
  * `word_tfidf__interested` ($x=0.0772, w=+0.8539$): **$+0.0659$**
* **Top Negative (HAM) Feature Contributions**:
  * `word_tfidf__newsletter` ($x=0.1370, w=-0.8496$): **$-0.1164$**
  * `word_tfidf__browser` ($x=0.1105, w=-0.9366$): **$-0.1035$**
  * `word_tfidf__now available` ($x=0.0911, w=-0.8385$): **$-0.0764$**
* **Diagnostic Summary**: Unsolicited commercial email formatted as a text newsletter. The presence of negative tokens (`newsletter`, `browser`, `now available`) combined with plain-text formatting ($0.0$ HTML score) pulled the decision score down to $-0.5734$.

---

### Example 4: False Negative (Val Index 704 — Borderline Plain-Text Spam)
* **Metadata Hash**: `28fd876d6403...` (`spam_2/00037.c7f0ce13d4cad8202f3d1a02b5cc5a1d`)
* **Ground Truth**: SPAM (1) | **Predicted Label**: **HAM**
* **Decision Score**: **$+0.3924$** | **Margin below $\tau=0.4013$**: **$+0.0089$**
* **Parsed Structural Attributes**: `html_present = 0.0`, `body_len = 2974`, `url_count = 1`.
* **Top Positive (SPAM) Feature Contributions**:
  * `word_tfidf__our` ($x=0.1245, w=+0.5229$): **$+0.0651$**
  * `word_tfidf__us` ($x=0.0942, w=+0.6792$): **$+0.0640$**
  * `word_tfidf__immediately` ($x=0.0712, w=+0.8258$): **$+0.0588$**
* **Top Negative (HAM) Feature Contributions**:
  * `word_tfidf__not` ($x=0.0578, w=-1.5574$): **$-0.0900$**
  * `word_tfidf__which` ($x=0.0572, w=-1.1436$): **$-0.0654$**
  * `word_tfidf__the` ($x=0.0581, w=-1.0601$): **$-0.0616$**
* **Diagnostic Summary**: Extremely close near-threshold error. The email scored $+0.3924$, missing the $\tau = 0.4013$ threshold by a tiny margin of **$0.0089$**.

---

## 6. Key Findings, Limitations & Recommended Future Work

### 6.1 Key Findings
1. **Asymmetric HTML Weighting**: `structural__html_present` ($+0.8728$) acts as a dominant positive signal. It protects against plain-text spam but accounts for $80\%$ of validation false positives when applied to clean HTML newsletters.
2. **Plain-Text Spam Association**: $100\%$ of validation false negatives occurred in plain-text messages. Without HTML features, plain-text spam requires dense promotional vocabulary to cross $\tau = 0.4013$.
3. **Keyword Sensitivity**: High negative weights on legitimate terms (`newsletter` $-0.8496$, `alert` $-0.5451$, `subscription` $-0.1636$) create a pattern where spam mimicking newsletter layouts receives lower decision scores.

### 6.2 Analysis & Sample-Size Limitations
* **Sample Size Constraint**: Error analysis is based on **17 total validation errors** ($5\text{ FPs}$ and $12\text{ FNs}$) out of $907$ validation samples. While these patterns are statistically informative, category percentages carry wide confidence bounds due to small sample sizes.
* **Correlational vs. Causal**: Feature contribution associations reflect model weights learned from training data; they describe how the model scored the message, not an external physical cause.

### 6.3 Recommended Future Work (Post-Level 11 Considerations)
> [!NOTE]
> **No Model Alteration Rule**: The recommendations below are documented strictly for future engineering iterations and were **not** implemented in this phase.

1. **Non-Linear / Sub-Linear HTML Weighting**: In future model versions, replace binary `html_present` indicators with normalized HTML-to-text ratios or non-linear scaling to prevent HTML tags from over-dominating text tokens.
2. **Contextual Tokenization**: Integrate character n-grams or contextual subword embeddings (e.g. FastText) to detect spam words obscured by punctuation or spaces.
3. **Multi-Feature Interaction Terms**: Engineer explicit interaction features between `html_present` and `url_count` to refine predictions on HTML newsletters.

---

## 7. Explicit Confirmation of Model & Artifact Integrity

* **Model State**: No model was retrained, refitted, or modified.
* **Decision Threshold**: Frozen threshold $\tau = 0.4013$ remains intact.
* **Manifest & Hashes**: `reports/final_model_manifest.json` (`config_sha256`: `61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2`) remains unchanged.
* **Dataset Splits**: No split assignments or labels were changed.
* **Level 7 Evaluation**: The Level 7 locked test-set evaluation was **not** rerun or modified in any way.

---

## 8. Summary of Created Files

1. **`tests/test_error_analysis.py`** *(New File)*:
   Automated unit tests validating confusion matrix reconstruction ($271+619+5+12=907$), score margin arithmetic, and frozen manifest integrity.
2. **`reports/error_analysis.md`** *(New File)*:
   This comprehensive Level 11 error analysis report.
