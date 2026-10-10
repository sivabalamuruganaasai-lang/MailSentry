# Read-Only Execution Audit Report: Level 7 Test-Set Evaluation

---

## Executive Summary

This report presents a READ-ONLY empirical audit of the Level 7 locked test-set evaluation execution in workspace `D:\email-spam-classifier`. The purpose of this audit is to verify the execution history, determine whether the initial script execution scored the test set or aborted prior to scoring, confirm that the test set was scored exactly once, and validate that all saved artifacts correspond to frozen candidate model **E11** under threshold **`0.4013`**.

> [!IMPORTANT]
> **READ-ONLY AUDIT GUARANTEE**:
> No models were retrained, no scripts were re-executed, no thresholds were altered, and no test set samples were re-evaluated during this audit. Findings are derived strictly from task execution logs, filesystem timestamps, file hashes, and prediction artifact inspection.

---

## 1. Timeline & Detailed Execution Reconstruction

Two background task invocations of `scratch/run_level7_evaluation.py` occurred during Level 7:

```text
====================================================================================================
EXECUTION TIMELINE & TASK AUDIT
====================================================================================================
1. Task ID: task-428 | Timestamp: 2026-10-09T12:07:10Z | Exit Code: 1 (FAILED)
   Script:  scratch/run_level7_evaluation.py
   Status:  FAILED AT IMPORT/COMPILATION PARSING (NameError)
   Inference/Scoring Executed: NO (0 test samples loaded, 0 predictions generated)

2. Task ID: task-438 | Timestamp: 2026-10-09T12:07:36Z | Exit Code: 0 (SUCCESS)
   Script:  scratch/run_level7_evaluation.py (after adding 'from typing import Tuple...')
   Status:  COMPLETED SUCCESSFULLY IN 37 SECONDS
   Inference/Scoring Executed: YES (906 test samples scored exactly once)
====================================================================================================
```

### 1.1 Detailed Audit of First Execution (Task ID: `task-428`)
- **Launch Timestamp**: `2026-10-09T12:07:10Z`
- **Result**: FAILED with exit code `1`.
- **Exact Error Log** ([`task-428.log`](file:///C:/Users/Lenovo/.gemini/antigravity-ide/brain/1af3fd98-1669-4de8-a71c-b0bd783228d3/.system_generated/tasks/task-428.log)):
  ```text
  Traceback (most recent call last):
    File "C:\Users\Lenovo\.gemini\antigravity-ide\brain\1af3fd98-1669-4de8-a71c-b0bd783228d3\scratch\run_level7_evaluation.py", line 24, in <module>
      def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
                                                                         ^^^^^
  NameError: name 'Tuple' is not defined. Did you mean: 'tuple'?
  ```
- **Execution Stage Reached**: Python module compilation at line 24.
- **Evidence**: The script crashed during initial function signature parsing. It terminated **before line 47** (`load_email_records_for_split`), **before opening dataset files**, **before fitting the model pipeline**, and **before writing any artifacts to disk**. Zero test predictions were generated.

### 1.2 Detailed Audit of Second Execution (Task ID: `task-438`)
- **Launch Timestamp**: `2026-10-09T12:07:36Z`
- **Completion Timestamp**: `2026-10-09T12:08:13Z` (~37 seconds runtime)
- **Result**: SUCCESS with exit code `0`.
- **Log Summary** ([`task-438.log`](file:///C:/Users/Lenovo/.gemini/antigravity-ide/brain/1af3fd98-1669-4de8-a71c-b0bd783228d3/.system_generated/tasks/task-438.log)):
  ```text
  Train samples: 4233 (HAM=2901, SPAM=1332)
  Val samples:   907 (HAM=624, SPAM=283)
  Test samples:  906 (HAM=625, SPAM=281)

  [>] Fitting frozen E11 pipeline strictly on Train split (4,233 emails)...
  [>] Generating predictions on locked Test split (906 emails)...

  =================== FINAL TEST EVALUATION METRICS ===================
  Confusion Matrix: TP=260, TN=621, FP=4, FN=21
  Accuracy:    0.9724 (95% CI: [0.9596, 0.9812])
  Precision:   0.9848 (95% CI: [0.9617, 0.9941])
  Recall:      0.9253 (95% CI: [0.8885, 0.9506])
  F1 Score:    0.9541
  F0.5 Score:  0.9723
  FPR:         0.64% (95% CI: [0.25%, 1.63%])
  Specificity: 0.9936
  ROC-AUC:     0.9982
  PR-AUC:      0.9961

  [+] Saved D:\email-spam-classifier\reports\final_model_manifest.json
  [+] Saved D:\email-spam-classifier\reports\final_test_predictions.csv
  [+] Saved D:\email-spam-classifier\reports\final_test_confusion_matrix.csv
  ```
- **Execution Stage Reached**: Complete end-to-end execution.

---

## 2. Empirical Proof of Test Prediction Generation

| Metric / Property | Execution 1 (`task-428`) | Execution 2 (`task-438`) |
|---|---|---|
| **Exit Status** | `FAILED (Exit Code 1)` | `SUCCESS (Exit Code 0)` |
| **Data Loaded** | No (0 records loaded) | Yes (4,233 Train, 907 Val, 906 Test) |
| **Model Fitted** | No (0 model fits) | Yes (`LinearSVC` fitted on 4,233 Train emails) |
| **Test Predictions Generated** | **NO (0 predictions)** | **YES (906 predictions generated)** |
| **Artifact Creation** | None | Created at `05:38:13 PM` (12:08:13Z) |

---

## 3. Verification of Saved Predictions & Model Manifest Alignment

Inspection of stored artifacts in `reports/` confirms complete alignment with the frozen audited configuration:

1. **[`final_model_manifest.json`](file:///d:/email-spam-classifier/reports/final_model_manifest.json)**:
   - **Model Name**: `LinearSVC`
   - **Experiment ID**: `E11`
   - **Feature Families**: `["word", "structural", "url"]` (10,000 max word features)
   - **Scaling**: `MaxAbsScaler`
   - **Frozen Threshold**: `0.4013`
   - **Config SHA256**: `61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2`

2. **`final_test_predictions.csv`** *(Local evaluation record preserved on host; excluded from Git)*:
   - **Total Prediction Rows**: Exactly 906.
   - **True Label Distribution**: 625 HAM (0), 281 SPAM (1).
   - **Recorded Optimal Threshold**: `0.4013` across all 906 rows.
   - **Decision Rule Consistency**: `pred_optimal_thresh == 1` if and only if `decision_score >= 0.4013` for all 906 rows.

3. **[`final_test_confusion_matrix.csv`](file:///d:/email-spam-classifier/reports/final_test_confusion_matrix.csv)**:
   - **TP (True SPAM)**: 260
   - **TN (True HAM)**: 621
   - **FP (False Positives / Legitimate HAM blocked)**: 4 ($\text{FPR} = 4 / 625 = 0.64\%$)
   - **FN (False Negatives / Missed SPAM)**: 21

---

## 4. Verification of Single-Evaluation Integrity

- **Claim**: "The locked test set was scored exactly once."
- **Audit Findings**:
  - `task-428` crashed prior to data loading or inference execution.
  - `task-438` was the **first and only** execution that loaded test samples and performed scoring.
  - No subsequent executions, parameter adjustments, or threshold tuning occurred.
- **Verdict**: The claim that the test set was scored **EXACTLY ONCE** is **100% SUPPORTED by empirical evidence**.

---

## 5. Unresolved Uncertainties

- **Unresolved Uncertainty**: **NONE**.
- All log files, execution exit codes, timestamp metadata, configuration hashes, and prediction rows match completely without contradiction.
