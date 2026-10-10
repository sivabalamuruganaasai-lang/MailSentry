# Level 8 — Explainability Report: Frozen Email Spam Classifier

## Executive Summary

This report establishes the complete global and local explainability framework for the frozen Candidate B (**E11**) Email Spam Classifier. Candidate B is a linear support vector machine (`LinearSVC`) fitted on 10,036 MaxAbsScaler-scaled features across three distinct feature families (Word TF-IDF, Structural metadata, and URL characteristics) operating at a frozen decision threshold of **$\tau = 0.4013$**.

Because `LinearSVC` operates as a strictly linear scoring function over scaled features, every prediction is **100% transparent and mathematically reconstructible** to machine precision without relying on approximations or black-box surrogates (such as SHAP or LIME).

---

## 1. Global Model Explanations

### 1.1 Methodology

The frozen model computes a decision score $S(x)$ as a linear combination of feature values $x_i$ and model weights $w_i$, shifted by an intercept $b$:

$$S(x) = b + \sum_{i=1}^{D} w_i x_i$$

Where:
* $D = 10,036$ total features.
* Intercept $b = +0.062258$.
* Positive coefficients ($w_i > 0$) increase the score, pushing predictions toward **SPAM**.
* Negative coefficients ($w_i < 0$) decrease the score, pushing predictions toward **HAM**.

### 1.2 Top Global Coefficients

#### Top 15 SPAM-Associated Coefficients ($w_i > 0$)
| Feature Name | Feature Family | LinearSVC Weight ($w_i$) | Interpretation |
| :--- | :--- | :--- | :--- |
| `structural__html_present` | Structural | **+0.872750** | Strong structural indicator of HTML-formatted content |
| `word_tfidf__remove` | Word TF-IDF | **+0.433968** | Direct unsubscription/removal marketing phrasing |
| `word_tfidf__your` | Word TF-IDF | **+0.425011** | Second-person marketing language |
| `word_tfidf__free` | Word TF-IDF | **+0.365418** | Promotional incentive term |
| `word_tfidf__!` | Word TF-IDF | **+0.329277** | Exclamation emphasis token |
| `url_feature__ip_host_count` | URL | **+0.298608** | Presence of raw IP addresses in embedded links |
| `word_tfidf__you will` | Word TF-IDF | **+0.294298** | Persuasive phrase token |
| `word_tfidf__our` | Word TF-IDF | **+0.290634** | First-person plural marketing token |
| `url_feature__avg_hostname_length` | URL | **+0.261077** | Long/obfuscated URL domain hostnames |
| `word_tfidf__du` | Word TF-IDF | **+0.258249** | Foreign language promotional token |
| `word_tfidf__spamassassin sightings` | Word TF-IDF | **+0.252164** | Public spam archive artifact token |
| `word_tfidf__sightings` | Word TF-IDF | **+0.250025** | Public spam archive artifact token |
| `word_tfidf__access` | Word TF-IDF | **+0.249894** | Account/offer access solicitation |
| `word_tfidf__! http` | Word TF-IDF | **+0.231826** | Exclamation preceding link |
| `word_tfidf__sightings lists` | Word TF-IDF | **+0.230993** | Public spam archive artifact token |

#### Top 15 HAM-Associated Coefficients ($w_i < 0$)
| Feature Name | Feature Family | LinearSVC Weight ($w_i$) | Interpretation |
| :--- | :--- | :--- | :--- |
| `url_feature__query_string_count` | URL | **-1.597423** | Complex query parameter structures in legitimate web links |
| `word_tfidf__lockergnome` | Word TF-IDF | **-1.167353** | Legitimate newsletter mailing list source token |
| `structural__html_tag_count` | Structural | **-0.966533** | High count of structured/clean HTML tags in body |
| `word_tfidf__newsletter` | Word TF-IDF | **-0.849575** | Explicit newsletter identification |
| `word_tfidf__surprised` | Word TF-IDF | **-0.831662** | Conversational interpersonal expression |
| `url_feature__shortener_count` | URL | **-0.778566** | Negative weight associated with standard shortener usage in dataset |
| `word_tfidf__not` | Word TF-IDF | **-0.750366** | Negation and conversational word |
| `word_tfidf__12` | Word TF-IDF | **-0.642786** | Specific numeric date/time token |
| `word_tfidf__auction` | Word TF-IDF | **-0.617069** | Specialized interest group discussion token |
| `word_tfidf__hp` | Word TF-IDF | **-0.612838** | Technical workplace domain/brand token |
| `word_tfidf__products` | Word TF-IDF | **-0.582684** | Product discussion token |
| `url_feature__https_count` | URL | **-0.548855** | Secure HTTPS link usage |
| `word_tfidf__alert` | Word TF-IDF | **-0.545062** | System notification/alert token |
| `word_tfidf__or` | Word TF-IDF | **-0.537485** | Disjunctive conjunction word |
| `word_tfidf__shell` | Word TF-IDF | **-0.474803** | Technical/Unix administration term |

### 1.3 Feature Family Breakdown

| Feature Family | Feature Count | Min Weight | Max Weight | Mean Weight | Top SPAM Feature | Top HAM Feature |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Word TF-IDF** | 10,000 | -1.1674 | +0.4340 | -0.0030 | `word_tfidf__remove` | `word_tfidf__lockergnome` |
| **Structural** | 19 | -0.9665 | +0.8728 | +0.0075 | `structural__html_present` | `structural__html_tag_count` |
| **URL** | 17 | -1.5974 | +0.2986 | -0.1646 | `url_feature__ip_host_count` | `url_feature__query_string_count` |

### 1.4 Association vs. Causation Disclaimer

> [!IMPORTANT]
> **Model Associations vs. Physical Causation**:
> LinearSVC coefficients quantify statistical correlations between feature occurrences and spam labels within the training distribution. A positive coefficient indicates that the presence of a feature increases the model's spam score; it does **not** prove that inserting or removing the word causes an email to become spam in an external or adversarial sense. For instance, artifact tokens like `word_tfidf__sightings` reflect dataset distribution characteristics rather than causal spam properties.

---

## 2. Individual Email Explanations

### 2.1 Feature Contribution Arithmetic

For any given email vector $x$, the contribution $c_i$ of feature $i$ is calculated as:

$$c_i = x_i \cdot w_i$$

* If $c_i > 0$, feature $i$ actively **pushes the prediction toward SPAM**.
* If $c_i < 0$, feature $i$ actively **pushes the prediction toward HAM**.
* If $x_i = 0$ (inactive feature), $c_i = 0.0$.

### 2.2 Representative Validation Explanations

Below are three representative illustrations from the validation set demonstrating individual score decompositions. To protect privacy and prevent data leakage, raw email bodies and sensitive text are omitted; only metadata hashes and feature contributions are shown.

#### Example 1: True SPAM (High Confidence)
* **Metadata Hash**: `9f8b61b0348d...`
* **Ground Truth**: SPAM (1) | **Model Prediction**: **SPAM**
* **Raw Decision Score**: **+1.1999** (Threshold $\tau = 0.4013$)
* **Intercept ($b$)**: **+0.0623**

| Feature Name | Family | Value ($x_i$) | Weight ($w_i$) | Contribution ($c_i$) | Direction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `structural__html_present` | Structural | 1.0000 | +0.8728 | **+0.8728** | $\rightarrow$ SPAM |
| `word_tfidf__pay` | Word | 0.7058 | +0.1316 | **+0.0929** | $\rightarrow$ SPAM |
| `word_tfidf__life` | Word | 1.1140 | +0.0672 | **+0.0749** | $\rightarrow$ SPAM |
| `word_tfidf__why` | Word | 0.8915 | +0.0675 | **+0.0602** | $\rightarrow$ SPAM |
| `url_feature__avg_dot_count` | URL | 0.2500 | +0.2185 | **+0.0546** | $\rightarrow$ SPAM |
| `structural__html_tag_count` | Structural | 0.0796 | -0.9665 | **-0.0769** | $\rightarrow$ HAM |
| `url_feature__max_url_length` | URL | 0.0518 | -0.3938 | **-0.0204** | $\rightarrow$ HAM |

* **Family Contribution Summary**:
  * Word TF-IDF: $+0.2520$
  * Structural: $+0.8360$
  * URL: $+0.0496$
  * Intercept: $+0.0623$
  * **Total Reconstructed Score**: $+0.2520 + 0.8360 + 0.0496 + 0.0623 = \mathbf{+1.1999}$

---

#### Example 2: True HAM (High Confidence)
* **Metadata Hash**: `7c0808e3174d...`
* **Ground Truth**: HAM (0) | **Model Prediction**: **HAM**
* **Raw Decision Score**: **-1.3544** (Threshold $\tau = 0.4013$)
* **Intercept ($b$)**: **+0.0623**

| Feature Name | Family | Value ($x_i$) | Weight ($w_i$) | Contribution ($c_i$) | Direction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `word_tfidf__unsubscribe` | Word | 0.6486 | -0.1636 | **-0.1061** | $\rightarrow$ HAM |
| `word_tfidf__not` | Word | 0.1154 | -0.7504 | **-0.0866** | $\rightarrow$ HAM |
| `word_tfidf__the` | Word | 0.2550 | -0.2846 | **-0.0726** | $\rightarrow$ HAM |
| `word_tfidf__yahoo !` | Word | 0.7038 | -0.0933 | **-0.0656** | $\rightarrow$ HAM |
| `word_tfidf__or` | Word | 0.1065 | -0.5375 | **-0.0573** | $\rightarrow$ HAM |
| `word_tfidf__yahoo com` | Word | 0.7881 | +0.1977 | **+0.1558** | $\rightarrow$ SPAM |
| `word_tfidf__join` | Word | 0.6593 | +0.1253 | **+0.0826** | $\rightarrow$ SPAM |

* **Family Contribution Summary**:
  * Word TF-IDF: $-1.4952$
  * Structural: $+0.0294$
  * URL: $+0.0491$
  * Intercept: $+0.0623$
  * **Total Reconstructed Score**: $-1.4952 + 0.0294 + 0.0491 + 0.0623 = \mathbf{-1.3544}$

---

#### Example 3: Borderline HAM (Near Threshold $\tau = 0.4013$)
* **Metadata Hash**: `99c336c136a7...`
* **Ground Truth**: HAM (0) | **Model Prediction**: **HAM**
* **Raw Decision Score**: **+0.4001** (Threshold $\tau = 0.4013$, difference $= -0.0012$)
* **Intercept ($b$)**: **+0.0623**

| Feature Name | Family | Value ($x_i$) | Weight ($w_i$) | Contribution ($c_i$) | Direction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `word_tfidf__hi i` | Word | 1.4224 | -0.1694 | **-0.2409** | $\rightarrow$ HAM |
| `word_tfidf__co` | Word | 0.7090 | -0.1765 | **-0.1251** | $\rightarrow$ HAM |
| `word_tfidf__ot` | Word | 1.0068 | -0.0868 | **-0.0874** | $\rightarrow$ HAM |
| `word_tfidf__could` | Word | 0.5984 | -0.1399 | **-0.0837** | $\rightarrow$ HAM |
| `structural__html_present` | Structural | 1.0000 | +0.8728 | **+0.8728** | $\rightarrow$ SPAM |
| `word_tfidf__strategies` | Word | 2.0468 | +0.0614 | **+0.1257** | $\rightarrow$ SPAM |

* **Analysis**: This email contains HTML formatting (`+0.8728`), which strongly pushes it toward SPAM. However, technical conversational phrases (`word_tfidf__hi i`, `word_tfidf__co`, `word_tfidf__could`) provide $-0.5517$ negative weight, pulling the total score to **+0.4001**, which sits just **0.0012 below the 0.4013 threshold**, correctly classifying it as HAM.

---

## 3. Mathematical Verification & Identity Proof

### 3.1 Theorem of Additive Score Reconstruction

Because Candidate B uses a `MaxAbsScaler` (a linear elementwise transformation $x_i = u_i / m_i$) and a `LinearSVC` decision function, the score is strictly linear with respect to the input features.

$$S(x) = b + \sum_{i=1}^{10036} x_i \cdot w_i$$

### 3.2 Empirical Verification Results

The score reconstruction identity was verified across all **907 emails in the validation dataset**:

$$\Delta = \left| S_{\text{actual}}(x) - \left( b + \sum_{i=1}^{10036} x_i w_i \right) \right|$$

* **Maximum Absolute Difference ($\max \Delta$)**: **$3.9968 \times 10^{-15}$**
* **Mean Absolute Difference ($\text{mean} \Delta$)**: **$4.5090 \times 10^{-16}$**
* **Tolerance Bound ($\epsilon$)**: **$1.0 \times 10^{-5}$**
* **Exact Verification Rate ($\Delta < \epsilon$)**: **907 / 907 (100.00%)**

> [!TIP]
> **Machine Precision Proof**: The maximum error across the entire validation set is on the order of $10^{-15}$, proving exact mathematical equality down to standard IEEE 754 double-precision floating-point limits.

### 3.3 Strict Absence of Probability Conversions

> [!WARNING]
> **Decision Scores are Distances, Not Probabilities**:
> `LinearSVC` decision function values represent signed geometric distances from the separating hyperplane in scaled feature space. They are **not** probabilities and are never mapped through sigmoid or softmax functions. The classification rule remains:
> $$\text{Prediction} = \begin{cases} \text{SPAM} & \text{if } S(x) \ge 0.4013 \\ \text{HAM} & \text{if } S(x) < 0.4013 \end{cases}$$

---

## 4. Automated Unit & Integration Test Suite

All explainability capabilities are integrated into `src/models/explainability.py` and thoroughly validated by `tests/test_explainability.py`.

### 4.1 Test Suite Breakdown (60 Total Tests Passed)

| Test Module | Test Name | Target Verified | Status |
| :--- | :--- | :--- | :--- |
| `tests/test_explainability.py` | `test_feature_family_classification` | Prefix mapping (`structural__`, `url_feature__`, etc.) | **PASS** |
| `tests/test_explainability.py` | `test_global_feature_importance_extraction` | Global weight extraction, family count & ordering | **PASS** |
| `tests/test_explainability.py` | `test_individual_explanation_exact_score_reconstruction` | Exact score reconstruction on validation set | **PASS** |
| `tests/test_explainability.py` | `test_explanation_positive_and_negative_contributions` | Contribution sorting and SPAM/HAM direction | **PASS** |
| `tests/test_explainability.py` | `test_empty_email_edge_case` | Structural fallback defaults & empty record handling | **PASS** |
| `tests/test_explainability.py` | `test_model_non_mutation` | Pipeline non-mutation during explanation generation | **PASS** |
| `tests/test_data_validation.py` | *All 4 tests* | Split integrity and schema validation | **PASS** |
| `tests/test_features.py` | *All 9 tests* | Feature extraction pipelines | **PASS** |
| `tests/test_models.py` | *All 11 tests* | Model training, scoring, and benchmarking | **PASS** |
| `tests/test_parser.py` | *All 7 tests* | MIME email parsing | **PASS** |
| `tests/test_splitting_integrity.py` | *All 5 tests* | Split non-overlap and leak checks | **PASS** |
| `tests/test_threshold_optimization.py` | *All 18 tests* | Threshold optimization & metrics | **PASS** |

---

## 5. Limitations & Scope

1. **Linearity Constraint**: The explainability framework relies on linear additivity ($c_i = x_i \cdot w_i$). It does not capture non-linear feature interactions beyond the explicit n-grams (`word_tfidf__you will`) engineered into the pipeline.
2. **Dataset Distribution Dependency**: Feature coefficients reflect statistical associations present in the training set. Dataset artifacts (e.g. SpamAssassin headers) receive positive weights due to historical training corpus composition.
3. **No Model Alteration**: Explanations are strictly post-hoc read-only diagnostics; they do not alter model predictions, weights, or operating threshold $\tau = 0.4013$.

---

## 6. Summary of Created & Modified Artifacts

1. **`src/models/explainability.py`** *(New File)*:
   Exposes `get_feature_family`, `get_global_feature_importance`, and `explain_individual_prediction` for global and local score decomposition.
2. **`src/models/__init__.py`** *(Modified File)*:
   Exports explainability functions.
3. **`tests/test_explainability.py`** *(New File)*:
   Contains 6 comprehensive unit tests for global weight extraction, exact score reconstruction, edge cases, non-mutation, and threshold rules.
4. **`reports/explainability_report.md`** *(New File)*:
   This comprehensive Level 8 report.
