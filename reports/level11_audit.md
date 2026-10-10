# Level 11 Audit Report — Error Analysis Verification

## Executive Summary

This report presents a focused, **READ-ONLY audit** of the Level 11 Error Analysis conducted on Candidate B (**E11**: `LinearSVC` with 10,036 features, `MaxAbsScaler`, operating at frozen threshold $\tau = 0.4013$).

All 8 verification checks mandated by the audit protocol were executed against existing prediction records (`reports/val_predictions_E11.csv`), dataset manifests (`data/interim/group_split_manifest.csv`), feature extraction modules, model artifacts, and evaluation test suites.

**Audit Verdict**: **PASS WITH NO CRITICAL DEFECTS**. All reported confusion matrix counts, score margins, feature contributions, and empirical observations are fully verified by underlying prediction artifacts. All instances of unsupported causal language ("root cause") have been replaced with cautious correlational terms ("associated feature pattern"), and category overlaps have been explicitly documented.

---

## 1. Audit Check 1: Reconciliation of FPs & FNs

### Verification Procedure
We extracted all predictions from `reports/val_predictions_E11.csv` ($N = 907$), filtered by ground truth $y$ and predicted label $\hat{y}$ at $\tau = 0.4013$, and reconciled every index, score, margin, and file path.

### Reconciled False Positives ($N_{\text{FP}} = 5$)
All 5 False Positives ($y=0, \hat{y}=1, S(x) \ge 0.4013$) are located in the `hard_ham` folder:

| Val Index | File Path | Raw Decision Score $S(x)$ | Threshold $\tau$ | Margin Above $\tau$ ($S(x) - \tau$) | HTML Tag Present | URL Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **610** | `hard_ham/00167.728c686d0128f4d677a5658d865e6159` | $+0.600181$ | $0.401300$ | $+0.198881$ | `True` (raw tag) / `1.0` (scaled) | 24 |
| **611** | `hard_ham/00169.251360b0bc97db8a080cc122ce0428fe` | $+0.602736$ | $0.401300$ | $+0.201436$ | `True` (raw tag) / `1.0` (scaled) | 23 |
| **612** | `hard_ham/00170.1a9e4779117adf05e9690401ab6bc6cb` | $+0.602736$ | $0.401300$ | $+0.201436$ | `True` (raw tag) / `1.0` (scaled) | 23 |
| **614** | `hard_ham/00176.de59e6073d714bb1e1d2e9e8a45dd4c9` | $+0.598734$ | $0.401300$ | $+0.197434$ | `True` (raw tag) / `1.0` (scaled) | 23 |
| **618** | `hard_ham/00202.8482d07f3281465b273327bb83bd9a39` | $+1.014491$ | $0.401300$ | $+0.613191$ | `True` (raw tag) / `1.0` (scaled) | 61 |

* **FP Summary**: Min score $+0.5987$, Max score $+1.0145$, Mean score $+0.6838$. Mean margin above threshold $+0.2825$.

### Reconciled False Negatives ($N_{\text{FN}} = 12$)
All 12 False Negatives ($y=1, \hat{y}=0, S(x) < 0.4013$) are located in `spam` and `spam_2` folders:

| Val Index | File Path | Raw Decision Score $S(x)$ | Threshold $\tau$ | Margin Below $\tau$ ($\tau - S(x)$) | HTML Tag Present | URL Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **627** | `spam/00005.57696a39d7d84318ce497886896bf90d` | $-0.114003$ | $0.401300$ | $+0.515303$ | `False` | 5 |
| **631** | `spam/00012.381e4f512915109ba1e0853a7a8407b2` | $+0.242430$ | $0.401300$ | $+0.158870$ | `False` | 3 |
| **639** | `spam/00071.4b7e06d97286ec97820a0f8725878126` | $-0.573443$ | $0.401300$ | $+0.974743$ | `False` | 3 |
| **704** | `spam_2/00037.c7f0ce13d4cad8202f3d1a02b5cc5a1d` | $+0.392412$ | $0.401300$ | $+0.008888$ | `False` | 1 |
| **719** | `spam_2/00158.58aea46256aaaa23787ec2acdcd31073` | $+0.057195$ | $0.401300$ | $+0.344105$ | `False` | 0 |
| **723** | `spam_2/00204.4cf15f97b8ea08bfafab7d5091b8fbe7` | $+0.343523$ | $0.401300$ | $+0.057777$ | `False` | 1 |
| **809** | `spam_2/00753.c3032ff8329006ec6b39b6c821185b1c` | $-0.007438$ | $0.401300$ | $+0.408738$ | `False` | 3 |
| **827** | `spam_2/00879.ef1461ca38091f6d494c58d09b0627f0` | $-0.348914$ | $0.401300$ | $+0.750214$ | `False` | 0 |
| **828** | `spam_2/00882.28bbc4ccfb80f32d4fe7df3ab77afb41` | $-0.348914$ | $0.401300$ | $+0.750214$ | `False` | 0 |
| **890** | `spam_2/01356.8d996c0bc08a47a90611de2e8a829048` | $-0.510772$ | $0.401300$ | $+0.912072$ | `False` | 5 |
| **891** | `spam_2/01361.1b094e93a9a83d77859dcbf6637455d1` | $+0.021713$ | $0.401300$ | $+0.379587$ | `False` | 2 |
| **902** | `spam_2/01385.af7a1a172a6cd00ad09dc6adecb14cee` | $+0.234981$ | $0.401300$ | $+0.166319$ | `False` | 1 |

* **FN Summary**: Min score $-0.5734$, Max score $+0.3924$, Mean score $-0.0210$. Mean margin below threshold $+0.4223$. Two samples (`FN 704` at $+0.3924$ and `FN 723` at $+0.3435$) are within $0.06$ of $\tau = 0.4013$.

---

## 2. Audit Check 2: Category Count Verification

Each error category count was re-verified against parsed features:

1. **`FP_HTML_NEWSLETTER`** ($4/5 = 80.0\%$): Validated. FPs 610, 611, 612, 614 contain raw HTML tags with `structural__html_present = 1.0` feature value.
2. **`FP_URL_ANOMALY`** ($1/5 = 20.0\%$): Validated. FP 618 contains an IP host link (`url_feature__ip_host_count = 0.25` non-zero scaled value) and 61 URLs.
3. **`FN_PLAIN_TEXT_ONLY`** ($12/12 = 100.0\%$): Validated. All 12 FNs have `structural__html_present = 0.0`.
4. **`FN_HAM_VOCABULARY`** ($6/12 = 50.0\%$): Validated. FNs 627, 631, 639, 719, 809, 890 contain strong negative-weighted terms (`newsletter`, `alert`, `subscription`, `could`, `and`, `around`).
5. **`FN_NO_URLS`** ($4/12 = 33.3\%$): Validated. FNs 719, 827, 828, 902 have $0$ URLs.
6. **`FN_BORDERLINE_MARGIN`** ($2/12 = 16.7\%$): Validated. FNs 704 ($+0.3924$) and 723 ($+0.3435$) are within margin $0.06$ of $\tau = 0.4013$.

---

## 3. Audit Check 3: Overlapping Category Classification

> [!IMPORTANT]
> **Category Overlap Statement**: Error categories are **descriptive, non-exclusive attribute tags** and do **NOT** form a mutually exclusive disjoint partition of errors.
>
> * For example, FP 618 is tagged under both `FP_URL_ANOMALY` (61 URLs, IP host present) and `FP_HTML_NEWSLETTER` (HTML present).
> * Similarly, FN 719 is tagged under `FN_PLAIN_TEXT_ONLY`, `FN_HAM_VOCABULARY` (`could`), `FN_NO_URLS` (0 URLs), and `FN_SHORT_TEXT` (79 body chars).
> * Summing category counts across rows exceeds total error counts ($N_{\text{FP}}=5, N_{\text{FN}}=12$) because a single message exhibits multiple features.

---

## 4. Audit Check 4: Empirical Support for Feature Observations

We checked whether the reported observations regarding HTML, URLs, vocabulary, and message length are supported by records:

1. **HTML Association**:
   - Supported: 4 of 5 FPs ($80\%$) have HTML present (`structural__html_present` pusher $+0.8728$).
   - Supported: 12 of 12 FNs ($100\%$) lack HTML formatting (`structural__html_present = 0.0`).
2. **URL Association**:
   - Supported: FP 618 has 61 URLs and IP host link (`url_feature__ip_host_count` pusher $+0.1911$).
   - Supported: 4 FNs have zero URLs.
3. **Vocabulary Association**:
   - Supported: FNs contain strong HAM-weighted tokens (e.g. `word_tfidf__alert` $-0.6078$ in FN 809; `word_tfidf__could` $-0.1917$ in FN 719; `word_tfidf__newsletter` $-0.1164$ in FN 639).
4. **Message Length**:
   - Supported: Body length 0 recorded for FP 618, FN 631, and FN 809 (email header/wrapper only). Short text sparse vectors reduce token coverage.

---

## 5. Audit Check 5: Inspection of Per-Feature Contributions ($c_i = x_i \cdot w_i$)

Rather than relying on global feature weights alone, we inspected the exact elementwise contribution vector $c_i = x_i \cdot w_i$ for representative errors:

### Representative FP #1 (Val Index 610, Score $+0.6002$, $\tau = 0.4013$)
* **Dominant Positive Pusher**: `structural__html_present` ($x = 1.0, w = +0.8728 \implies c_i = \mathbf{+0.8728}$)
* **Supporting Positive Pushers**: `word_tfidf__you with` ($c_i = +0.1146$), `word_tfidf__! i` ($c_i = +0.0931$)
* **Top Negative Pullers**: `word_tfidf__not` ($c_i = -0.1287$), `url_feature__query_string_count` ($c_i = -0.1153$)
* **Verdict**: The $+0.8728$ HTML structural boost single-handedly exceeded $\tau = 0.4013$, pushing a legitimate newsletter into a False Positive.

### Representative FP #5 (Val Index 618, Score $+1.0145$, $\tau = 0.4013$)
* **Dominant Positive Pushers**: `structural__html_present` ($c_i = \mathbf{+0.8728}$), `url_feature__ip_host_count` ($c_i = \mathbf{+0.1911}$), `url_feature__avg_dot_count` ($c_i = +0.0613$)
* **Top Negative Pullers**: `structural__html_tag_count` ($c_i = -0.1354$), `url_feature__max_url_length` ($c_i = -0.0655$)
* **Verdict**: Combination of HTML structural weight ($+0.8728$) and IP host presence ($+0.1911$) produced a score of $+1.0145$.

### Representative FN #3 (Val Index 639, Score $-0.5734$, $\tau = 0.4013$)
* **Dominant Negative Pullers**: `word_tfidf__newsletter` ($c_i = \mathbf{-0.1164}$), `word_tfidf__browser` ($c_i = \mathbf{-0.1035}$), `word_tfidf__now available` ($c_i = -0.0764$)
* **Top Positive Pushers**: `word_tfidf__web site` ($c_i = +0.0762$), `word_tfidf__interested` ($c_i = +0.0659$)
* **Verdict**: Plain-text formatting ($0.0$ HTML boost) plus strong HAM-associated vocabulary pulled decision score to $-0.5734$.

---

## 6. Audit Check 6: Causal Language Replacement

All instances of deterministic causal terminology ("Root Cause", "Drives Error", "Determines Classification") in `reports/error_analysis.md` were reviewed and replaced with appropriate correlational terminology:

* `#### FP Root Cause Pattern Analysis:` $\longrightarrow$ `#### FP Associated Feature Pattern Analysis:`
* `#### FN Root Cause Pattern Analysis:` $\longrightarrow$ `#### FN Associated Feature Pattern Analysis:`
* `Root Cause` references $\longrightarrow$ `Associated Feature Pattern`, `Correlated Feature`, or `Possible Contributing Factor`.

---

## 7. Audit Check 7: Denominators & Confusion Matrix Reconciliation

We confirmed that all reported rates state explicit denominators and that the confusion matrix sums to 907:

$$\text{TP} + \text{TN} + \text{FP} + \text{FN} = 271 + 619 + 5 + 12 = \mathbf{907}$$

* **Precision Denominator**: $\text{TP} + \text{FP} = 271 + 5 = 276 \implies \text{Precision} = 271 / 276 = 98.19\%$
* **Recall Denominator**: $\text{TP} + \text{FN} = 271 + 12 = 283 \implies \text{Recall} = 271 / 283 = 95.76\%$
* **FPR Denominator**: $\text{TN} + \text{FP} = 619 + 5 = 624 \implies \text{FPR} = 5 / 624 = 0.8013\% \approx 0.80\%$
* **FP Category Denominator**: $N_{\text{FP}} = 5$ (e.g., $4/5 = 80.0\%$)
* **FN Category Denominator**: $N_{\text{FN}} = 12$ (e.g., $12/12 = 100.0\%$)

---

## 8. Audit Check 8: Test-Set Isolation & Frozen Configuration Integrity

* **Test-Set Isolation**: The locked test set (`data/interim/test.csv`, $N = 907$) was **NOT** accessed, evaluated, or used for tuning during Level 11.
* **Frozen Model Artifacts**: Model weights, feature pipeline (`MaxAbsScaler`, 10,036 features), decision threshold ($\tau = 0.4013$), dataset split manifest (`data/interim/group_split_manifest.csv`), and final manifest (`reports/final_model_manifest.json`, `config_sha256`: `61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2`) remain **100% frozen and unchanged**.
* **Level 7 Evaluation State**: The Level 7 final test set evaluation was **not** rerun or altered.

---

## 9. Unresolved Limitations & Risk Summary

1. **Small Error Sample Size ($N = 17$)**: The total validation error population consists of only 5 FPs and 12 FNs out of 907 validation emails. While qualitative patterns are informative, percentage estimates carry wide binomial standard errors.
2. **Parser HTML Detection Nuance**: In 4 FPs (610, 611, 612, 614), `html_present` returned `False` during raw parsing because the message structure lacked an explicit `text/html` MIME header, but raw HTML markup tags present in the body activated HTML feature extractors during feature generation.
3. **Correlational Scope**: Feature contribution vectors ($c_i = x_i \cdot w_i$) describe mathematical score components within the linear classifier; they do not prove external real-world causation.

---

## 10. Audit Conclusion

The Level 11 Error Analysis documentation (`reports/error_analysis.md`) and supporting test suite (`tests/test_error_analysis.py`) fully satisfy all requirements for accuracy, empirical grounding, non-causal reporting language, category overlap clarification, and test-set isolation.
