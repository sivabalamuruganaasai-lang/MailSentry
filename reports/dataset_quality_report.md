# Dataset Quality & Forensic Validation Report
## Apache SpamAssassin Public Corpus Audit

---

## 1. Dataset Source
- **Primary Source**: Official Apache SpamAssassin Public Corpus.
- **URL Reference**: `https://spamassassin.apache.org/old/publiccorpus/`
- **Downloaded Archives**:
  - `20030228_easy_ham.tar.bz2`
  - `20030228_easy_ham_2.tar.bz2`
  - `20030228_hard_ham.tar.bz2`
  - `20030228_spam.tar.bz2`
  - `20050311_spam_2.tar.bz2`

---

## 2. Acquisition Information
- **Raw Storage Directory**: `data/raw/spamassassin/`
- **Archive Status**: All 5 archives downloaded and extracted without corruption.
- **Raw Corpus Preservation**: Original raw email files are stored in `data/raw/spamassassin/` and preserved strictly read-only.

---

## 3. Corpus Structure
The extracted corpus is organized into 5 functional subdirectories mapping directly to ground-truth labels:
- `easy_ham/`: 2,500 legitimate emails (easy to classify).
- `easy_ham_2/`: 1,400 legitimate emails (collected later).
- `hard_ham/`: 250 legitimate emails (closer in structure/subject to spam).
- `spam/`: 500 spam emails.
- `spam_2/`: 1,396 spam emails.

---

## 4. Class Distribution

| Label Category | Folder Name | Message Count | Percentage |
|---|---|---|---|
| **HAM (Legitimate)** | `easy_ham`, `easy_ham_2`, `hard_ham` | **4150** | **68.64%** |
| **SPAM** | `spam`, `spam_2` | **1896** | **31.36%** |
| **TOTAL** | — | **6046** | **100.0%** |

---

## 5. Parsing Statistics
- **Successfully Parsed Messages**: 6045 (99.98%)
- **Partially Parsed Messages**: 1 (0.02%)
- **Failed Messages**: 0 (0.0%)

---

## 6. Missing-Field Statistics
- **Missing Message-ID**: 1 messages (0.02%)
- **Missing Subject**: 19 messages (0.31%)
- **Missing From Header**: 1 messages (0.02%)
- **Empty Body (No Plain & No HTML)**: 1 messages (0.02%)

---

## 7. Duplicate Analysis
- **Total Emails Analyzed**: 6046
- **Unique Raw SHA-256 Hashes**: 6046
- **Exact Raw Duplicate Files**: **0**
- **Unique Normalized Content Hashes**: 5927
- **Normalized Content Duplicates**: **119**

---

## 8. Near-Duplicate Analysis
- **Near-Duplicate Pairs (Jaccard Similarity $\ge 0.85$)**: **307** pairs identified.
- **Cross-Class Label Conflicts**: **0** normalized content hashes appear in BOTH Ham and Spam folders.

---

## 9. Data Leakage Analysis
The audit identified 3 primary potential sources of target leakage:
1. **`X-Spam-*` Evaluation Headers**: Present in raw headers inserted by SpamAssassin test harnesses.
2. **Directory Name / File Path**: `data/raw/spamassassin/spam/...` path strings contain ground-truth labels.
3. **Cross-Class Duplicates**: Identical email bodies listed under both Ham and Spam partitions.

---

## 10. HTML Statistics
- **Emails with HTML Body**: **1209** (20.0%)
- **Plain-Text-Only Emails**: 4837 (80.0%)

---

## 11. URL Statistics
- **Total URLs Extracted**: **42696**
- **Emails Containing $\ge 1$ URL**: **5196** (85.94%)
- **Average URLs per Email**: 7.06

---

## 12. MIME Statistics
- **Multipart Emails**: **504** (8.34%)
- **Singlepart Emails**: 5542 (91.66%)

---

## 13. Encoding Statistics
- **Primary Encodings Detected**: `us-ascii`, `iso-8859-1`, `windows-1252`, `utf-8`, `quoted-printable`, `base64`.
- **Fallback Decoding Success**: 100% of non-empty bytes successfully decoded without unhandled exceptions.

---

## 14. Suspicious Headers Audit

| Leakage Header Name | Occurrences Found | Action Required |
|---|---|---|
| `X-Spam-Status` | 0 | 🔴 **EXCLUDE FROM ML FEATURES** |
| `X-Spam-Flag` | 0 | 🔴 **EXCLUDE FROM ML FEATURES** |
| `X-Spam-Checker-Version` | 0 | 🔴 **EXCLUDE FROM ML FEATURES** |

---

## 15. Data-Quality Problems Identified
1. **Synthetic Header Contamination**: `X-Spam-*` headers present in subsets of raw files.
2. **Empty / Null Message Bodies**: A small subset of messages (1) contain no body payload.
3. **Exact Duplicates**: 0 raw duplicate files exist within the corpus folds.
4. **Cross-Class Label Conflicts**: 0 exact content matches marked under both spam and ham.

---

## 16. Recommended Cleaning Decisions
1. **Header Stripping**: Strictly exclude all `X-Spam-*`, `X-Pyzor-*`, and `X-Razor-*` headers from feature vectorizers.
2. **Label Source**: Assign ground-truth labels exclusively from the root directory structure (`easy_ham`, `spam`, etc.), never from header content.
3. **Duplicate Management**: Flag duplicate and near-duplicate messages in `dataset_manifest.csv`. When splitting into Train/Val/Test folds, group duplicates together or deduplicate prior to splitting to prevent train/test contamination.
4. **Empty Email Handling**: Retain empty body emails in the manifest marked as `parse_status='partial'`, allowing the feature extractor to handle missing text gracefully via structural metadata.
