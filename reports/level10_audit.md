# Level 10 Audit Report — Robustness, Adversarial Sensitivity & Pipeline Integrity

## Executive Summary

This read-only audit report verifies the experimental methodology, mathematical formulas, flip count statistics, threat model classifications, input validation mechanisms, test suite warnings, and artifact SHA-256 hashes from Level 10 (**Robustness and Adversarial Testing**).

The frozen Candidate B (**E11**) model, feature extraction pipeline (10,036 features), dataset split manifests, and decision threshold ($\tau = 0.4013$) remain **100% frozen and unaltered**.

---

## 1. Audit Point 1 — Signed vs. Absolute Mean Score Shift ($\Delta S$)

### 1.1 Formula Verification in Code
In `scratch/run_robustness_experiments.py`, the score shift $\Delta S_i$ for sample $i$ is calculated as:

$$\Delta S_i = S_{\text{transformed}, i} - S_{\text{original}, i}$$

The reported **Mean $\Delta S$** is the **Signed Arithmetic Mean**:

$$\mu_{\Delta S} = \frac{1}{N} \sum_{i=1}^{N} \left( S_{\text{transformed}, i} - S_{\text{original}, i} \right)$$

The reported **Max Score Shift** is the **Maximum Absolute Magnitude Shift**:

$$\text{Max } \|\Delta S\| = \max_{1 \le i \le N} \left| S_{\text{transformed}, i} - S_{\text{original}, i} \right|$$

### 1.2 Audit Findings
* **Signed Mean Directionality**:
  * A **positive** mean delta (e.g. $+0.2105$ for Excessive Punctuation) indicates that the transformation shifted the average decision score towards SPAM.
  * A **negative** mean delta (e.g. $-0.0803$ for Ham Signature Insertion) indicates that the transformation shifted the average decision score towards HAM.
* **Distinction**: Signed mean reflects aggregate directional drift, whereas maximum absolute shift captures peak individual sample vulnerability.

---

## 2. Audit Point 2 — Verification of Flip Counts & Denominators

### 2.1 Dataset Denominators
* Total Validation Samples ($N$): **907**
* Baseline SPAM Predictions ($N_{\text{SPAM}}$): **276** (at $\tau = 0.4013$)
* Baseline HAM Predictions ($N_{\text{HAM}}$): **631** (at $\tau = 0.4013$)
* Validation Split Integrity Check: $N_{\text{SPAM}} + N_{\text{HAM}} = 276 + 631 = 907$.

### 2.2 Reconciled Flip Metrics Table

| Transformation Category | SPAM $\rightarrow$ HAM Flips (Evasion) | Denominator ($N_{\text{SPAM}}$) | SPAM Flip Rate (%) | HAM $\rightarrow$ SPAM Flips (False Pos.) | Denominator ($N_{\text{HAM}}$) | HAM Flip Rate (%) | Total Flips | Denominator ($N$) | Total Flip Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **UPPERCASE All Text** | 0 | 276 | **0.00%** | 4 | 631 | **0.63%** | 4 | 907 | **0.44%** |
| **lowercase All Text** | 3 | 276 | **1.09%** | 0 | 631 | **0.00%** | 3 | 907 | **0.33%** |
| **Extra Whitespace** | 0 | 276 | **0.00%** | 1 | 631 | **0.16%** | 1 | 907 | **0.11%** |
| **Excessive Punctuation** | 0 | 276 | **0.00%** | 8 | 631 | **1.27%** | 8 | 907 | **0.88%** |
| **Ham Signature Insert** | 15 | 276 | **5.43%** | 1 | 631 | **0.16%** | 16 | 907 | **1.76%** |
| **Spam Word Obfuscation**| 7 | 276 | **2.54%** | 0 | 631 | **0.00%** | 7 | 907 | **0.77%** |

* **Audit Finding**: All reported counts, denominators, and calculated percentages are **100% verified** against empirical JSON artifacts.

---

## 3. Audit Point 3 — Semantic Meaning Preservation & Threat Classification

### 3.1 Dual Security Perspective Classification

| Transformation | Human Reader Perception | Vectorizer Tokenization | Security Classification |
| :--- | :--- | :--- | :--- |
| **UPPERCASE / Lowercase** | Identical Meaning | Case-folded unigrams preserved | Pure Formatting-Preserving |
| **Extra Whitespace** | Identical Meaning | Normalized by whitespace split | Pure Formatting-Preserving |
| **Excessive Punctuation** | Preserved Intent, Amplified Tone | Increases exclamation/question features | Stylistic Perturbation Attack |
| **Ham Signature Insertion** | Preserved Core Message + Added Sign-off | Injects HAM-associated tokens & metadata | **Contextual Injection Attack** |
| **Spam-Word Obfuscation (`f r e e`)** | May Preserve Human Sense (Case-Dependent) | Destroys exact unigram match | **Token-Altering Evasion Attack** |

### 3.2 Audit Findings & Clarifications
* **Contextual Injection (Ham Signature)**: Appending a workplace email signature (*"Thanks & Best Regards, John Doe..."*) changes the **contextual evidence** of the email by adding external salutations, corporate disclaimers, and workplace metadata. It should be explicitly described as **contextual injection**, not a purely formatting-preserving transformation.
* **Token-Altering Obfuscation (Word Splitting)**: Splitting spam words (`free` $\rightarrow$ `f r e e`, `click` $\rightarrow$ `c.l.i.c.k`) is fundamentally **token-altering**. While human readers may infer the underlying word in simple cases, word splitting breaks token boundaries and is **not universally meaning-preserving** (e.g. it can alter compound words, domain terms, and semantic context).


---

## 4. Audit Point 4 — Handling of Unexpected Input Types

### 4.1 Input Coercion vs. Strict Validation
Inspection of `TextSelector` (`src/features/pipeline.py`) and `StructuralFeatureExtractor` (`src/features/structural.py`) reveals:

```python
# TextSelector coercion
subj = str(r.get("subject", "") or "")
body = str(r.get("plain_body", "") or "")

# Structural URL list coercion
urls = record.get("urls", [])
if not isinstance(urls, list):
    urls = []
```

### 4.2 Audit Findings
* **Silent Type Coercion**: The ingestion pipeline does **not** raise formal validation errors or reject malformed data types. Instead, it **silently coerces** non-string scalars (`int`, `float`, `bool`) to strings via `str()`, and coerces non-list URL inputs into empty lists `[]`.
* **System Stability vs. Ingestion Hygiene**: Silent coercion ensures 100% crash resistance (zero runtime exceptions), but API consumers should be aware that numeric scalars (e.g. `subject=12345`) are transformed to string tokens (`"12345"`) and scored rather than rejected.

---

## 5. Audit Point 5 — Investigation of Test Suite Warning

### 5.1 Warning Isolation Traceback
```text
tests/test_models.py::test_cv_vectorizer_fitted_per_fold_only
  sklearn/model_selection/_split.py:1036: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=3.
```

### 5.2 Root Cause Analysis
* `test_cv_vectorizer_fitted_per_fold_only` in `tests/test_models.py` uses a tiny 4-sample mock dataset (`y = [0, 1, 0, 1]`) to test that TF-IDF vectorizers in cross-validation folds are fitted strictly on training folds and never see validation fold data.
* Splitting 4 samples across $K=3$ folds in `StratifiedGroupKFold` causes class 0 and class 1 to have only 2 members per split, prompting scikit-learn to issue a warning.

### 5.3 Action Item
* **No action required**. The warning is strictly confined to a 4-sample synthetic unit test fixture and does not affect production pipeline code, dataset splits ($N=4,233$), or evaluation results.

---

## 6. Audit Point 6 — Comprehensive Artifact Hashes & Model Integrity

### 6.1 Verified Artifact Hashes

| Artifact File | Normalized LF SHA-256 Hash | Raw Binary SHA-256 Hash (Windows) | Verification Status |
| :--- | :--- | :--- | :--- |
| `reports/final_model_manifest.json` | `f3bb1d91944247b22330d1149fe0ef5cae3eea3633682882d352485217d71530` | `d6a84610390ece6bf1f96814eb3e5bac1b3c7df01aa8f16961bc717ac10c5066` | **VERIFIED UNCHANGED** |
| `data/interim/group_split_manifest.csv` | `eaef62d1df0abf27091ab8cb88dd88d17933162c54d9fcdee086c441be275d88` | `28fd876d6403000f68555bac8d8ca4488de11b17c368dcebe95dc88e68796c5c` | **VERIFIED UNCHANGED** |
| `reports/val_predictions_E11.csv` | `e21c72f90a3d06faec36b62c9da31fb483b465bd68b736095182be9891b8f7e2` | `fb825a30fabe9f585b88fa57fed0c33ac729b7e028208347e51112f8430b70e9` | **VERIFIED UNCHANGED** |
| `src/features/url_features.py` | `5d132ec2e4fa9e6076df9fb3de941d244906fc650e2b2d50eebb1ab1c2b7467b` | `5d132ec2e4fa9e6076df9fb3de941d244906fc650e2b2d50eebb1ab1c2b7467b` | **VERIFIED UNCHANGED** |
| `src/features/url_security.py` | `a9bf1f9de3cc3210a0dd194ca7195f37cdbeac9b73d2ba7b6de87e7aab401300` | `a9bf1f9de3cc3210a0dd194ca7195f37cdbeac9b73d2ba7b6de87e7aab401300` | **VERIFIED UNCHANGED** |
| `src/models/explainability.py` | `6ec5c34039bc5b355f384e815d136c9868636399225dc19859af9061e80f1820` | `6ec5c34039bc5b355f384e815d136c9868636399225dc19859af9061e80f1820` | **VERIFIED UNCHANGED** |

* **Frozen Manifest Inner Hash**: `config_sha256 = 61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2` (verified matching stored value).
* **Frozen Threshold**: $\tau = 0.4013$ (verified matching).

### 6.2 Serialized Binary Model Artifact Scope Statement
> [!IMPORTANT]
> **Serialized Model Artifact Integrity Disclaimer**:
> This audit has verified the SHA-256 integrity of model manifests (`reports/final_model_manifest.json`), split records (`data/interim/group_split_manifest.csv`), validation prediction CSVs (`reports/val_predictions_E11.csv`), and python source code.
> However, because no pre-experiment SHA-256 hash of a binary serialized model file (e.g. `.joblib` or `.pkl`) was recorded prior to Level 10, this audit **does not establish the integrity of a serialized binary model artifact**. Model pipeline reproducibility relies on the verified manifest parameters and deterministic random seeds (`random_state=42`).


---

## 7. Summary of Unresolved Limitations

1. **Unigram Tokenizer Vulnerability**: The model lacks character n-grams and is susceptible to token-boundary splitting attacks (`f r e e`).
2. **Signature Injection Vulnerability**: Ingesting long workplace email signatures reduces spam decision scores and can cause marginal evasion flips ($5.43\%$).
3. **Silent Input Coercion**: Malformed input data types are silently coerced into strings/empty lists rather than raising structured validation rejection errors.

---

## 8. Audit Conclusion

The Level 10 audit confirms that all robustness and adversarial testing was conducted strictly on validation data, with complete mathematical rigor, exact denominator tracking, zero model retraining, and zero threshold modifications.
