# Candidate Model Benchmarking & Evaluation Report (Level 5)

---

## Executive Summary
This report details the empirical benchmarking of baseline classification models for email spam detection on the **Validation Set**. **The test set remained 100% untouched and protected** throughout all experiments.

A total of **11 controlled experiments (E01 - E11)** were conducted across three candidate classifiers (`MultinomialNB`, `LogisticRegression`, `LinearSVC`) combined with different feature representations (Word TF-IDF, Char TF-IDF, Word+Char TF-IDF, Structural, and URL features).

---

## 1. Class & Problem Definitions

- **Positive Class (1)**: **SPAM**
- **Negative Class (0)**: **HAM**
- **False Positive (FP)**: HAM incorrectly classified as SPAM (*Primary operational risk: user losing legitimate email*).
- **False Positive Rate (FPR)**: $FP / (FP + TN)$
- **True Positive Rate (TPR / Recall)**: $TP / (TP + FN)$

---

## 2. Ranked Validation Benchmark Results

Experiments are ranked by **1. F1 Score**, **2. Precision**, **3. False Positive Rate (lower is better)**, **4. Recall**, and **5. PR-AUC**:

| Rank | Exp ID | Model | Features | Accuracy | Precision | Recall | F1 | F0.5 | FPR | ROC-AUC | PR-AUC | Latency (ms/sample) | Size (KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **E11** | LinearSVC | `word+structural+url` | 0.9846 | **0.9719** | 0.9788 | **0.9754** | 0.9733 | **0.0128** | 0.9983 | 0.9959 | 1.5994 ms | 604.0 KB |
| **2** | **E03** | LinearSVC | `word` | 0.9824 | **0.9619** | 0.9823 | **0.9720** | 0.9659 | **0.0176** | 0.9986 | 0.9969 | 0.9436 ms | 446.5 KB |
| **3** | **E08** | LogisticRegression | `word+char` | 0.9802 | **0.9617** | 0.9753 | **0.9684** | 0.9644 | **0.0176** | 0.9984 | 0.9965 | 4.9663 ms | 1063.4 KB |
| **4** | **E09** | LinearSVC | `word+char` | 0.9802 | **0.9617** | 0.9753 | **0.9684** | 0.9644 | **0.0176** | 0.9984 | 0.9963 | 4.8210 ms | 1063.2 KB |
| **5** | **E10** | LinearSVC | `word+structural` | 0.9791 | **0.9615** | 0.9717 | **0.9666** | 0.9636 | **0.0176** | 0.9984 | 0.9965 | 1.5861 ms | 603.5 KB |
| **6** | **E06** | LinearSVC | `char` | 0.9757 | **0.9579** | 0.9647 | **0.9613** | 0.9592 | **0.0192** | 0.9972 | 0.9932 | 4.5536 ms | 618.5 KB |
| **7** | **E02** | LogisticRegression | `word` | 0.9757 | **0.9644** | 0.9576 | **0.9610** | 0.9630 | **0.0160** | 0.9983 | 0.9962 | 0.9413 ms | 446.6 KB |
| **8** | **E05** | LogisticRegression | `char` | 0.9735 | **0.9609** | 0.9541 | **0.9574** | 0.9595 | **0.0176** | 0.9970 | 0.9931 | 5.6249 ms | 618.6 KB |
| **9** | **E07** | MultinomialNB | `word+char` | 0.9350 | **0.9375** | 0.8481 | **0.8905** | 0.9181 | **0.0256** | 0.9893 | 0.9746 | 4.8896 ms | 1649.2 KB |
| **10** | **E01** | MultinomialNB | `word` | 0.9294 | **0.9700** | 0.7986 | **0.8760** | 0.9300 | **0.0112** | 0.9944 | 0.9867 | 0.9429 ms | 680.9 KB |
| **11** | **E04** | MultinomialNB | `char` | 0.9151 | **0.9153** | 0.8021 | **0.8550** | 0.8902 | **0.0337** | 0.9786 | 0.9496 | 5.5034 ms | 970.1 KB |

---

## 3. Stratified Group-Aware 5-Fold Cross-Validation on Training Data

To verify stability and rule out overfitting to the validation split, 5-fold cross-validation was performed **strictly on Training Data**, enforcing zero group-crossing between folds:

| Exp ID | Model | Mean F1 ± Std | Mean Precision ± Std | Mean Recall ± Std | Mean FPR ± Std |
|---|---|---|---|---|---|
| **E11** | LinearSVC | **0.9709 ± 0.0097** | 0.9750 ± 0.0076 | 0.9670 ± 0.0122 | **0.0114 ± 0.0034** |
| **E03** | LinearSVC | **0.9712 ± 0.0063** | 0.9694 ± 0.0087 | 0.9730 ± 0.0087 | **0.0141 ± 0.0041** |
| **E08** | LogisticRegression | **0.9609 ± 0.0080** | 0.9636 ± 0.0194 | 0.9587 ± 0.0109 | **0.0169 ± 0.0095** |
| **E10** | LinearSVC | **0.9680 ± 0.0078** | 0.9699 ± 0.0063 | 0.9662 ± 0.0117 | **0.0138 ± 0.0029** |

---

## 4. Top 3 Candidate Models Analysis

Based on multi-objective validation criteria, the top 3 performing candidates are:

### Top 1: Experiment E11 (LinearSVC + `word+structural+url`)
- **F1 Score**: **0.9754** (F0.5 = 0.9733)
- **Precision**: **0.9719** (8 False Positives out of 624 HAM emails)
- **False Positive Rate**: **0.0128** (1.28%)
- **Recall**: **0.9788** (277 SPAM detected out of 283)
- **PR-AUC**: **0.9959** | **ROC-AUC**: **0.9983**
- **Inference Speed**: **1.5994 ms/email** | **Model Size**: **604.0 KB**

### Top 2: Experiment E03 (LinearSVC + `word`)
- **F1 Score**: **0.9720** (F0.5 = 0.9659)
- **Precision**: **0.9619** (11 False Positives out of 624 HAM emails)
- **False Positive Rate**: **0.0176** (1.76%)
- **Recall**: **0.9823** (278 SPAM detected out of 283)
- **PR-AUC**: **0.9969** | **ROC-AUC**: **0.9986**
- **Inference Speed**: **0.9436 ms/email** | **Model Size**: **446.5 KB**

### Top 3: Experiment E08 (LogisticRegression + `word+char`)
- **F1 Score**: **0.9684** (F0.5 = 0.9644)
- **Precision**: **0.9617** (11 False Positives out of 624 HAM emails)
- **False Positive Rate**: **0.0176** (1.76%)
- **Recall**: **0.9753** (276 SPAM detected out of 283)
- **PR-AUC**: **0.9965** | **ROC-AUC**: **0.9984**
- **Inference Speed**: **4.9663 ms/email** | **Model Size**: **1063.4 KB**

---

## 5. Confusion Matrices Summary

Confusion matrix plots for top candidates have been saved under `reports/figures/`:
- `reports/figures/confusion_matrix_E08.png`
- `reports/figures/confusion_matrix_E09.png`
- `reports/figures/confusion_matrix_E06.png` (or corresponding top models)

---

## 6. Error Analysis Preparation & Validation Isolation

- Validation predictions, decision scores, message IDs, ground truth, and predicted labels are persisted under `reports/val_predictions_<EXP_ID>.csv` for post-hoc error analysis.
- **Test Set Protection Verification**: Zero test samples were used in model fitting, feature extraction fitting, hyperparameter selection, or ranking.