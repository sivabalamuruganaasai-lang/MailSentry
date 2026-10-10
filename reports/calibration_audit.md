# Probability Calibration Audit Report (Level 6.2 Updated)

---

## Executive Summary

This report presents a thorough audit of the Platt-scaling probability calibration experiment conducted on `LinearSVC` candidate models (E03 and E11). We evaluate data split isolation, PR-AUC and ROC-AUC calculations, monotonicity preservation, computational overhead, and whether probability calibration is necessary or beneficial for threshold-based classification.

---

## 1. Data Isolation & Metric Verification

### 1.1 Evaluated Dataset & Ground Truth
- Both uncalibrated and calibrated evaluations were conducted on the **exact same 907 validation examples** ($N=907$; 624 HAM, 283 SPAM).
- The positive class is strictly defined as **SPAM (1)**.
- Ground truth labels $Y_{\text{val}}$ remained identical across both evaluations.

### 1.2 Calibrator Fitting Protocol
- `CalibratedClassifierCV(estimator=model, method='sigmoid', cv=5)` was fitted strictly on the **Training split ONLY** (`train`, $N=4,233$).
- Zero validation or test data was used to fit the calibration parameters $(A_k, B_k)$.

### 1.3 Test Set Protection
- The 906-sample locked **Test Set** was **100% untouched and unaccessed**. Zero test samples were used for fitting or evaluating calibration mappings.

---

## 2. Metric Reproduction Results (E11 & E03)

Validation split results (N = 907):

| Model ID | Pipeline Description | Metric Type | Uncalibrated Raw Score | Calibrated Probability | Delta ($\Delta$) | Spearman Rank Corr ($r_s$) | Kendall Rank Corr ($\tau$) |
|---|---|---|---|---|---|---|---|
| **E03** | `LinearSVC` (Word TF-IDF) | **ROC-AUC** | **0.9986** | 0.9984 | -0.0002 | 0.9818 | 0.8841 |
| **E03** | `LinearSVC` (Word TF-IDF) | **PR-AUC** | **0.9969** | 0.9965 | -0.0004 | 0.9818 | 0.8841 |
| **E11** | `LinearSVC` (Word+Struct+URL) | **ROC-AUC** | **0.9983** | 0.9977 | -0.0006 | 0.9709 | 0.8616 |
| **E11** | `LinearSVC` (Word+Struct+URL) | **PR-AUC** | **0.9959** | 0.9940 | -0.0019 | 0.9709 | 0.8616 |

---

## 3. Detailed Investigation of Monotonicity & Score Ties

### 3.1 Monotonicity Audit
- **Unique Uncalibrated Scores**: 885 unique values across 907 validation samples.
- **Unique Calibrated Probabilities**: 885 unique values across 907 validation samples.
- **Monotonicity Violations**: Sorting validation samples by uncalibrated decision score revealed **430 strict monotonicity violations out of 884 adjacent pairs (48.6% of adjacent pairs)**.

### 3.2 Root Cause of Monotonicity Violations
When `CalibratedClassifierCV` is run with 5-fold cross-validation, scikit-learn fits 5 separate fold classifiers $f_k(x)$ and 5 sigmoid calibrators $(A_k, B_k)$. Prediction on a new sample $x$ evaluates:

$$P(y=1|x) = \frac{1}{5} \sum_{k=1}^5 \frac{1}{1 + \exp(A_k \cdot f_k(x) + B_k)}$$

Because each fold classifier $f_k$ is trained on $4/5$ of the training data with slightly different feature weights, taking the average of 5 non-linear sigmoids across different sub-models produces non-monotonic score oscillations when plotted against the single main model's decision score $f(x)$.

### 3.3 Metric Impact
These 430 rank reversals introduce minor noise into sample ordering, causing PR-AUC for E11 to drop from **0.9959** (raw) to **0.9940** (calibrated) and ROC-AUC to drop from **0.9983** to **0.9977**.

---

## 4. Final Verdict & Recommendation

```text
================================================================================
CALIBRATION AUDIT VERDICT
================================================================================
LEAKAGE STATUS:       CLEAN (0 validation/test leakage)
MONOTONICITY:         VIOLATED (430 rank reversals out of 884 adjacent pairs)
METRIC IMPACT:        PR-AUC dropped from 0.9959 to 0.9940 (-0.0019)
RECOMMENDATION:       REJECT CALIBRATION WRAPPER. Retain raw decision_function()
                      scores for LinearSVC (E11) as the primary decision metric.
================================================================================
```
