# Three-Tier Decision Policy Audit Report (Level 6.2 Terminology Update)

---

## Executive Summary

This report presents a thorough mathematical audit of the 3-tier classification policy (**Auto HAM**, **REVIEW Queue**, **Auto SPAM**) evaluated on candidate model **E11** (`LinearSVC` + Word TF-IDF + Structural + URL features) using the 907-sample Validation split. All metric definitions are explicitly disambiguated to prevent confusion between automatic actions and human escalation workflows.

---

## 1. Decision Policy Definitions & Boundary Rules

### 1.1 Policy Rules
The 3-tier classification framework routes incoming emails based on decision score boundaries:

```text
Score < Lower Threshold                 => AUTO HAM  (Direct Delivery)
Lower Threshold <= Score <= Upper Thresh => REVIEW    (Human Moderation Queue)
Score > Upper Threshold                 => AUTO SPAM (Auto Blocked)
```

### 1.2 Boundary Equality Inclusion Audit
- **Lower Boundary ($\tau_{\text{lower}} = -0.3520$)**:
  - Emails with $\text{score} < -0.3520$ are assigned to **AUTO HAM**.
  - Emails with $\text{score} == -0.3520$ belong to the **REVIEW Queue**.
- **Upper Boundary ($\tau_{\text{upper}} = 0.6010$)**:
  - Emails with $\text{score} == 0.6010$ belong to the **REVIEW Queue**.
  - Emails with $\text{score} > 0.6010$ are assigned to **AUTO SPAM**.

> **Summary**: Both boundary equalities ($\text{score} == \tau_{\text{lower}}$ and $\text{score} == \tau_{\text{upper}}$) are strictly included in the **REVIEW Queue** to ensure ambiguous boundary cases receive human verification.

---

## 2. Audited Exact Counts & Performance Metrics (Validation Set, N = 907)

Validation Dataset Composition: **907 total emails** (**624 HAM**, **283 SPAM**).

### 2.1 Tier Breakdown Table

| Tier Name | Decision Condition | Total Volume | % Total Volume | True HAM | True SPAM | Tier Precision |
|---|---|---|---|---|---|---|
| **AUTO HAM** | $\text{score} < -0.3520$ | **615 emails** | 67.81% | **613 (TN)** | **2 (FN)** | 99.67% (NPV) |
| **REVIEW Queue** | $-0.3520 \le \text{score} \le 0.6010$ | **36 emails** | **3.97%** | **8 (HAM)** | **28 (SPAM)** | N/A (Human Queue) |
| **AUTO SPAM** | $\text{score} > 0.6010$ | **256 emails** | 28.22% | **3 (FP)** | **253 (TP)** | **98.83%** |
| **Total** | — | **907 emails** | 100.00% | **624 HAM** | **283 SPAM** | — |

---

## 3. Disambiguated Metric Definitions & Exact Values

To ensure strict clarity, 5 core metrics are explicitly defined and distinguished:

### 3.1 Automatic Spam Recall
$$\text{Automatic Spam Recall} = \frac{\text{TP}_{\text{Auto SPAM}}}{\text{Total Actual SPAM}} = \frac{253}{283} = \mathbf{0.8940} \quad (89.40\%)$$
*Measures the proportion of all actual spam emails automatically caught and blocked without human intervention.*

### 3.2 Spam Captured-or-Escalated Rate
$$\text{Spam Captured-or-Escalated Rate} = \frac{\text{TP}_{\text{Auto SPAM}} + \text{SPAM}_{\text{REVIEW}}}{\text{Total Actual SPAM}} = \frac{253 + 28}{283} = \frac{281}{283} = \mathbf{0.9929} \quad (99.29\%)$$
*Measures the combined proportion of actual spam emails that were either auto-blocked or sent to moderation queue (only 2 spam emails slipped into Auto HAM).*

### 3.3 Automatic Decision Coverage
$$\text{Automatic Decision Coverage} = \frac{N_{\text{Auto HAM}} + N_{\text{Auto SPAM}}}{N_{\text{Total}}} = \frac{615 + 256}{907} = \frac{871}{907} = \mathbf{0.9603} \quad (96.03\%)$$
*Measures the proportion of total email traffic processed automatically without requiring human moderation.*

### 3.4 Auto-SPAM Precision
$$\text{Auto-SPAM Precision} = \frac{\text{TP}_{\text{Auto SPAM}}}{\text{TP}_{\text{Auto SPAM}} + \text{FP}_{\text{Auto SPAM}}} = \frac{253}{253 + 3} = \frac{253}{256} = \mathbf{0.9883} \quad (98.83\%)$$
*Measures the proportion of auto-blocked emails that are truly spam.*

### 3.5 HAM False-Positive Rate (Auto-SPAM FPR)
$$\text{HAM False-Positive Rate} = \frac{\text{FP}_{\text{Auto SPAM}}}{\text{Total Actual HAM}} = \frac{3}{624} = \mathbf{0.0048} \quad (0.48\%)$$
*Measures the proportion of legitimate user HAM emails incorrectly auto-blocked as spam.*

---

## 4. Audit of Narrative Terminology

1. **Automatic Spam Recall (89.40%) vs. Spam Captured-or-Escalated Rate (99.29%)**:
   - Calling 99.29% "Spam Recall" without qualification is misleading because 28 of those spam emails are in the REVIEW queue and require human action to block.
   - True **Automatic Spam Recall** is **89.40%**.
2. **Auto-SPAM Precision (98.83%) vs. HAM FPR (0.48%)**:
   - The policy achieves 98.83% precision with **3 false positives** in Auto-SPAM. Claims of "zero false positives" are inaccurate and corrected.
