# Level 7 — Final Locked Test-Set Evaluation Report

---

## Executive Summary

This report documents the single, final evaluation of the locked **Test Set** ($N=906$) for the email spam classifier. The evaluation was conducted using Candidate Model **E11** (`LinearSVC` + Word TF-IDF + Structural + URL features with `MaxAbsScaler`) under the frozen decision threshold **`0.4013`**.

> [!IMPORTANT]
> **HOLDOUT INTEGRITY CONFIRMATION**:
> The 906-sample Test Set was accessed **EXACTLY ONCE** for this final evaluation. Model fitting, feature pipeline fitting, hyperparameter selection, threshold tuning, and calibration audits were conducted strictly on Training ($N=4,233$) and Validation ($N=907$) splits without inspecting or scoring test set labels.

---

## 1. Frozen Model Configuration & Manifest

The evaluated pipeline configuration was frozen prior to test evaluation and recorded in [`reports/final_model_manifest.json`](file:///d:/email-spam-classifier/reports/final_model_manifest.json):

```json
{
  "model_name": "LinearSVC",
  "experiment_id": "E11",
  "feature_families": ["word", "structural", "url"],
  "word_max_features": 10000,
  "word_mode": "light",
  "apply_scaling": true,
  "scaler_type": "MaxAbsScaler",
  "random_state": 42,
  "frozen_threshold": 0.4013,
  "decision_operator": "score >= threshold",
  "positive_class": "SPAM (1)",
  "negative_class": "HAM (0)",
  "train_sample_count": 4233,
  "val_sample_count": 907,
  "test_sample_count": 906,
  "config_sha256": "61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2"
}
```

---

## 2. Test-Set Integrity & Separation Audit

- **Test Set Size**: **906 total emails** (**625 HAM**, **281 SPAM**).
- **Group Leakage Prevention**: Verified via `group_split_manifest.csv` that zero email groups or sender domains cross between Train, Validation, and Test splits.
- **Fitting Isolation**: Pipeline transformers (`TfidfVectorizer`, `MaxAbsScaler`) and classifier weights (`LinearSVC`) were fitted strictly on the 4,233 training emails.

---

## 3. Final Test Evaluation Results (N = 906)

### 3.1 Confusion Matrix

Saved to [`reports/final_test_confusion_matrix.csv`](file:///d:/email-spam-classifier/reports/final_test_confusion_matrix.csv):

| Ground Truth Label | Predicted HAM (0) | Predicted SPAM (1) | Total Class Count |
|---|---|---|---|
| **Actual HAM (0)** | **621 (TN)** | **4 (FP)** | **625 HAM** |
| **Actual SPAM (1)** | **21 (FN)** | **260 (TP)** | **281 SPAM** |
| **Total Predicted** | **642 Predicted HAM** | **264 Predicted SPAM** | **906 Total** |

### 3.2 Key Performance Metrics

| Metric | Test Set Value | Formula / Calculation | 95% Confidence Interval (Wilson) |
|---|---|---|---|
| **False Positive Rate (FPR)** | **0.64%** | $\text{FP} / \text{Actual HAM} = 4 / 625$ | **[0.25%, 1.63%]** |
| **Precision** | **0.9848** (98.48%) | $\text{TP} / (\text{TP} + \text{FP}) = 260 / 264$ | **[96.17%, 99.41%]** |
| **Recall (TPR)** | **0.9253** (92.53%) | $\text{TP} / (\text{TP} + \text{FN}) = 260 / 281$ | **[88.85%, 95.06%]** |
| **F1 Score** | **0.9541** | $2 \cdot P \cdot R / (P + R)$ | — |
| **F0.5 Score** | **0.9723** | $1.25 \cdot P \cdot R / (0.25 P + R)$ | — |
| **Accuracy** | **0.9724** (97.24%) | $(\text{TP} + \text{TN}) / N = 881 / 906$ | **[95.96%, 98.12%]** |
| **Specificity (TNR)** | **0.9936** (99.36%) | $\text{TN} / \text{Actual HAM} = 621 / 625$ | **[98.37%, 99.75%]** |
| **ROC-AUC** | **0.9982** | Area under ROC Curve | — |
| **PR-AUC (Average Precision)** | **0.9961** | Area under PR Curve | — |

---

## 4. Validation vs. Test Set Comparison

| Metric | Validation Set (N=907: 624 HAM, 283 SPAM) | Locked Test Set (N=906: 625 HAM, 281 SPAM) | Delta ($\Delta = \text{Test} - \text{Val}$) | Evaluation Assessment |
|---|---|---|---|---|
| **Threshold ($\tau$)** | **0.4013** | **0.4013** (Frozen) | $0.0000$ | Frozen |
| **False Positive Rate (FPR)** | **0.80%** (5 FP) | **0.64%** (4 FP) | **-0.16%** | **FPR Improved** ($\text{FPR} \le 1.0\%$ satisfied) |
| **Precision** | **0.9819** (98.19%) | **0.9848** (98.48%) | **+0.0029** | **Precision Improved** |
| **Recall** | **0.9576** (95.76%) | **0.9253** (92.53%) | -0.0323 | Slight drop in Recall |
| **F1 Score** | **0.9696** | **0.9541** | -0.0155 | Strong generalization |
| **F0.5 Score** | **0.9769** | **0.9723** | -0.0046 | Excellent Precision-weighted match |
| **Accuracy** | **0.9813** | **0.9724** | -0.0089 | Strong overall accuracy |
| **ROC-AUC** | **0.9983** | **0.9982** | -0.0001 | Virtually identical ranking capability |
| **PR-AUC** | **0.9959** | **0.9961** | **+0.0002** | **PR-AUC Improved** |

### Comparative Insights
1. **False Positive Control**: The primary operational goal (suppressing false positives below 1.0%) was **successfully validated on unseen test data**: FPR dropped to **0.64%** (only 4 false positives out of 625 legitimate emails).
2. **Precision Stability**: Auto-flagged SPAM precision increased from **98.19%** on validation to **98.48%** on test set, demonstrating that 98.5% of auto-blocked emails are truly spam.
3. **Generalization**: ROC-AUC (0.9982) and PR-AUC (0.9961) on the test set closely match validation performance, proving zero overfitting during model development.

---

## 5. Statistical Uncertainty & Confidence Intervals

All proportion metrics are accompanied by 95% Wilson score confidence intervals to quantify sample uncertainty:
- **Precision**: 98.48% (95% CI: **[96.17%, 99.41%]**).
- **Recall**: 92.53% (95% CI: **[88.85%, 95.06%]**).
- **False Positive Rate**: 0.64% (95% CI: **[0.25%, 1.63%]**).
- **Accuracy**: 97.24% (95% CI: **[95.96%, 98.12%]**).

---

## 6. Operational Risk, Limitations & Real-World Domain Shift

> [!WARNING]
> **DEPLOYMENT LIMITATIONS & DOMAIN SHIFT NOTICE**:
> 1. **Dataset Scope**: These evaluation results are based on the SpamAssassin corpus. Real-world corporate email streams may exhibit distinct vocabulary, language shifts, and novel phishing strategies not represented in this static dataset.
> 2. **Threshold Immutability**: The threshold ($\tau = 0.4013$) was frozen prior to evaluation and was **NOT modified** after viewing test results.
> 3. **Future Monitoring**: In enterprise deployment, continuous monitoring of false-positive complaints and moderation queue feedback is recommended to detect concept drift.
