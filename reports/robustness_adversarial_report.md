# Level 10 — Robustness and Adversarial Testing Report

## Executive Summary

This report presents the robustness, input resilience, and adversarial sensitivity evaluation of the frozen Candidate B (**E11**) Email Spam Classifier. All diagnostic experiments were conducted deterministically using `random_state=42` on validation records.

The underlying model (`LinearSVC`), feature pipeline (10,036 Word TF-IDF, Structural, and URL features), and decision threshold ($\tau = 0.4013$) remained strictly **frozen and unaltered** throughout this phase.

### Model Manifest & Artifact Integrity Proof
* **Manifest Path**: `reports/final_model_manifest.json`
* **Config SHA-256**: `61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2`
* **File SHA-256**: `f3bb1d91944247b22330d1149fe0ef5cae3eea3633682882d352485217d71530`
* **Decision Threshold**: $\tau = 0.4013$ (Raw `decision_function` score)
* **Integrity Status**: **VERIFIED UNCHANGED** (100% hash match before and after Level 10).

---

## 1. Task 1 — Input Robustness & Edge-Case Audit

The pipeline was evaluated against 11 extreme, malformed, and non-standard input cases to test processing stability, memory bounds, and exception safety.

### 1.1 Robustness Test Results

| Case Name | Description / Input Type | Result | Decision Score ($S$) | Prediction ($\tau=0.4013$) | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Empty Subject & Body** | `subject=""`, `body=""`, `urls=[]` | **SUCCESS** | $+0.1780$ | HAM | $3.76\text{ ms}$ |
| **Null / Missing Fields** | `subject=None`, `body=None`, `urls=None` | **SUCCESS** | $+0.1780$ | HAM | $2.33\text{ ms}$ |
| **Very Long Subject** | 10,000 characters in subject | **SUCCESS** | $-0.7909$ | HAM | $7.72\text{ ms}$ |
| **Very Long Body (1 MB)** | 1 MB payload text (30,000 lines) | **SUCCESS** | $+1.0666$ | SPAM | $623.25\text{ ms}$ |
| **Unicode & Emojis** | Emojis (`🎉 FREE 💰 PRIZE 🚀 CLICK`) | **SUCCESS** | $+0.5650$ | SPAM | $36.92\text{ ms}$ |
| **Zero-Width Whitespace** | `F\u200br\u200be\u200be`, `P\u00a0a\u00a0y` | **SUCCESS** | $-0.2672$ | HAM | $3.01\text{ ms}$ |
| **HTML-Only Email** | `plain_body=""`, HTML tag content | **SUCCESS** | $+0.6269$ | SPAM | $2.67\text{ ms}$ |
| **Plain-Text-Only Email** | `html_body=""`, simple plain text | **SUCCESS** | $+0.4037$ | SPAM | $5.60\text{ ms}$ |
| **Malformed Header/Syntax** | Header injection `\n\r`, invalid host | **SUCCESS** | $-0.1511$ | HAM | $4.48\text{ ms}$ |
| **Repeated Words & Punct.** | 2,000x "FREE" + `!!!!?????` | **SUCCESS** | $+0.7063$ | SPAM | $13.68\text{ ms}$ |
| **Unexpected Non-String Types**| `subject=12345`, `body=["list"]`, `urls="str"`| **SUCCESS** | $+0.6664$ | SPAM | $3.23\text{ ms}$ |

* **Audit Summary**: **11 / 11 cases processed cleanly** with zero crashes, unhandled exceptions, memory errors, or infinite loops.

---

## 2. Task 2 — Adversarial Transformations & Sensitivity Experiments

Controlled, deterministic text transformations were applied across the **907 validation dataset emails** (276 SPAM, 631 HAM) to measure score shifts ($\Delta S = S_{\text{trans}} - S_{\text{orig}}$) and prediction class flip rates.

### 2.1 Transformation-Level Metrics

| Transformation Category | Meaning Classification | Mean $\Delta S$ | Max $\|\Delta S\|$ | SPAM $\rightarrow$ HAM Flips | HAM $\rightarrow$ SPAM Flips | Total Class Flips | Total Flip Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **UPPERCASE All Text** | Meaning-Preserving | $+0.0676$ | $0.0925$ | 0 / 276 ($0.00\%$) | 4 / 631 ($0.63\%$) | 4 / 907 | **$0.44\%$** |
| **lowercase All Text** | Meaning-Preserving | $-0.0070$ | $0.0968$ | 3 / 276 ($1.09\%$) | 0 / 631 ($0.00\%$) | 3 / 907 | **$0.33\%$** |
| **Extra Whitespace & Newlines**| Meaning-Preserving | $+0.0008$ | $0.0853$ | 0 / 276 ($0.00\%$) | 1 / 631 ($0.16\%$) | 1 / 907 | **$0.11\%$** |
| **Excessive Punctuation (`!`, `?`)**| Meaning-Preserving | $+0.2105$ | $1.2856$ | 0 / 276 ($0.00\%$) | 8 / 631 ($1.27\%$) | 8 / 907 | **$0.88\%$** |
| **Insert Legitimate Ham Signature**| Meaning-Preserving | $-0.0803$ | $0.8006$ | 15 / 276 ($5.43\%$) | 1 / 631 ($0.16\%$) | 16 / 907 | **$1.76\%$** |
| **Obfuscate High-Weight Spam Terms**| Meaning-Altering | $-0.0042$ | $0.7045$ | 7 / 276 ($2.54\%$) | 0 / 631 ($0.00\%$) | 7 / 907 | **$0.77\%$** |

### 2.2 Semantic Meaning Classification

1. **Meaning-Preserving Transformations**:
   * *Case changes, whitespace insertions, punctuation additions, and ham signature attachments* preserve the original semantic intent of the sender.
   * *Observation*: Appending a legitimate workplace email signature (*"Thanks & Best Regards, John Doe..."*) shifted decision scores by an average of **$-0.0803$**, flipping 15 marginal SPAM emails ($5.43\%$) to HAM. This occurs because words like "thanks", "regards", and "management" carry negative (HAM-associated) coefficients in the model.
2. **Meaning-Altering / Word-Splitting Transformations**:
   * *Word-splitting obfuscation* (`free` $\rightarrow$ `f r e e`, `click` $\rightarrow$ `c.l.i.c.k`) alters character n-grams and token boundaries.
   * *Observation*: Because E11 relies on word-level unigrams/bigrams and excludes character n-grams, splitting words neutralizes exact word TF-IDF matches, shifting score downwards and flipping 7 SPAM emails ($2.54\%$) to HAM.

---

## 3. Task 3 — Safety, Resilience, and Isolation Proofs

### 3.1 Network Isolation Proof
* **Verification**: In `tests/test_robustness.py` and `tests/test_url_security.py`, `socket.socket` and `socket.gethostbyname` were monkeypatched to throw a `RuntimeError` on execution.
* **Result**: All URL security checks and robustness evaluations executed strictly offline with **ZERO network calls**.

### 3.2 Memory and Processing Latency Bounds
* **1 MB Payload Test**: Processing a 1 MB payload body (30,000 lines of text) completed in **623.25 ms** with bounded peak memory allocation.
* **Standard Payload Average**: Mean inference latency across standard validation emails remains $< 5\text{ ms}$ per sample.

---

## 4. Known Model Vulnerabilities & Recommended Mitigations

### 4.1 Identified Model Vulnerabilities
1. **Unigram Token Obfuscation**: Because E11 relies on unigram/bigram tokenization without character n-grams, simple spaces between letters (`f r e e`) prevent exact token matching.
2. **Signature / Salutation Poisoning**: Appending long, legitimate business signatures or boilerplate disclaimers introduces multiple HAM-associated word tokens, reducing the overall decision score.
3. **Punctuation Sensitivity**: Excessive exclamation points (`!`) increase the spam score ($+0.2105$ mean delta), occasionally causing false positives on excited HAM messages.

### 4.2 Recommended Operational Mitigations
1. **Text Normalization Preprocessor**: Apply character normalization (stripping zero-width spaces, converting `f r e e` to `free`) prior to feature extraction in future model iterations.
2. **Decoupled URL Security Inspection**: Maintain the Level 9 offline URL security analyzer (`src/features/url_security.py`) to detect obfuscated links and IP-address hosts independently of textual TF-IDF tokens.
3. **Threshold Stability**: Retain the audited threshold $\tau = 0.4013$, which maintains strict false-positive control ($\text{FPR} \le 0.80\%$).

---

## 5. Automated Unit & Integration Test Suite

All 77 project unit and integration tests passed cleanly.

### 5.1 Test Suite Breakdown (77 Total Passed)

| Test Module | Test Count | Target Scope | Status |
| :--- | :--- | :--- | :--- |
| `tests/test_robustness.py` *(New)* | **4** | Manifest SHA-256 hash, edge cases, deterministic transformations, non-mutation | **PASS (20.79s)** |
| `tests/test_url_security.py` | 13 | URL structural indicators & network isolation | **PASS (0.16s)** |
| `tests/test_explainability.py` | 6 | Score reconstruction & contribution arithmetic | **PASS** |
| `tests/test_data_validation.py` | 4 | Split integrity & schema validation | **PASS** |
| `tests/test_features.py` | 9 | Feature extraction pipelines | **PASS** |
| `tests/test_models.py` | 11 | Estimator fitting and benchmark metrics | **PASS** |
| `tests/test_parser.py` | 7 | Email MIME parsing | **PASS** |
| `tests/test_splitting_integrity.py` | 5 | Group split integrity & zero-leak checks | **PASS** |
| `tests/test_threshold_optimization.py` | 18 | Threshold optimization & operating points | **PASS** |
| **TOTAL TEST SUITE** | **77** | Full regression and unit test suite | **PASS (99.71s)** |

---

## 6. Summary of Created and Modified Files

1. **`tests/test_robustness.py`** *(New File)*:
   Automated unit tests for manifest hash verification, input robustness edge cases, deterministic transformations, and pipeline non-mutation.
2. **`reports/robustness_adversarial_report.md`** *(New File)*:
   This comprehensive Level 10 report.
