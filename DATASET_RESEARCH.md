# Dataset Intelligence Specification
## Competition-Grade Email Spam Classifier

---

## 1. Project Dataset Requirements

To build a competition-grade, security-aware Email Spam Classifier, the underlying dataset must fulfill strict structural, linguistic, and operational criteria. Unlike generic SMS or plain-text classification tasks, real-world email classification relies heavily on rich email metadata, MIME structure, subject/body segregation, security signals (URLs, domain names, headers), and realistic class imbalance.

### Key Requirements Checklist:
1. **Real Email Format**: Must represent actual email protocol structures (RFC 5322 / MIME), not abbreviated text messages (SMS) or stripped text-only strings.
2. **Component Granularity**: Must provide distinct, unparsed or cleanly separable `Subject`, `Body`, `Headers`, and `HTML` elements to test subject-vs-body models and security extractors.
3. **URL & Security Signal Richness**: Must contain real-world spam URLs, domain references, IP addresses, TLDs, and HTML link tag structures.
4. **Authenticity & Diversity**: Sourced from real-world email streams (not synthetic generation), reflecting authentic human writing in HAM and diverse spam campaigns.
5. **Clean Data Lineage & No Leakage**: Free from severe train/test contamination, identical duplicates, or structural artifacts that allow trivial shortcut learning (e.g. static system-injected header tags).
6. **Reproducibility & Legal Compliance**: Openly accessible, non-proprietary or public domain / research licensed, and redistributable for competition validation.

---

## 2. SpamAssassin Public Corpus Analysis

The **SpamAssassin Public Corpus** is an established benchmark dataset created by the Apache SpamAssassin project, containing raw emails received from public forums, newsletters, and direct submissions.

### 26-Point Evaluation

1. **Dataset Name**: Apache SpamAssassin Public Corpus (2002–2005 Collections).
2. **Source/Origin**: Sourced by the SpamAssassin development team from public mailing lists, developer inboxes, and user-submitted spam.
3. **Authenticity**: **[Verified Fact]** Authentic raw email messages collected from real email traffic.
4. **Number of Spam Messages**: **[Verified Fact]** 1,896 spam messages (`spam-1`: 500, `spam-2`: 1,396).
5. **Number of Ham Messages**: **[Verified Fact]** 4,151 ham messages (`easy-ham-1`: 2,500, `easy-ham-2`: 1,400, `hard-ham`: 251).
6. **Class Balance**: **[Verified Fact]** Total: 6,047 emails. ~31.4% Spam, ~68.6% Ham (Imbalance ratio ~ 1:2.2).
7. **Raw Email Availability**: **[Verified Fact]** 100% available in raw RFC 822 / RFC 5322 text format.
8. **Header Availability**: **[Verified Fact]** Full original email headers present (`Received:`, `From:`, `To:`, `Subject:`, `Date:`, `Message-ID:`, `Content-Type:`).
9. **Subject Availability**: **[Verified Fact]** Present in raw header lines (`Subject:` field).
10. **Body Availability**: **[Verified Fact]** Full body text available across all messages.
11. **HTML Availability**: **[Verified Fact]** Both plain text and raw HTML body content present (especially in `spam-2` and `hard-ham`).
12. **MIME/Multipart Availability**: **[Verified Fact]** Multi-part MIME boundaries, boundary tags, and inline text/html parts intact.
13. **Encoding Diversity**: **[Verified Fact]** Diverse encodings included (ASCII, UTF-8, ISO-8859-1, Quoted-Printable, Base64).
14. **Sender Information Availability**: **[Verified Fact]** `From:`, `Return-Path:`, and `Received:` hops intact (with limited privacy anonymization, e.g. `spamassassin.taint.org`).
15. **URL Availability**: **[Verified Fact]** Rich URL presence in both HTML href tags and plain-text body strings.
16. **Duplicate/Repeated-Message Risk**: **[Verified Fact]** Moderate. Contains repeated spam campaign templates with minor variation, as well as multiple replies in ham mailing lists.
17. **Data Leakage Risk**: **[Reasonable Inference]** Risk of header leakage if synthetic headers inserted during dataset creation (e.g. `X-Spam-Status`) are not explicitly stripped before feature extraction.
18. **Licensing/Usage Considerations**: **[Verified Fact]** Open Apache / Public Domain for research. Official maintainers explicitly note: *Do not re-inject into live production mail systems*.
19. **Accessibility/Download Method**: Direct HTTP download of `.tar.bz2` archives from official Apache SpamAssassin repository (`https://spamassassin.apache.org/old/publiccorpus/`).
20. **Dataset Maintenance/Status**: **[Verified Fact]** Legacy static dataset (last updated ~2005); highly stable benchmark.
21. **Suitability for Supervised ML**: **[High]** Excellent benchmark for MultinomialNB, Logistic Regression, and SVM due to realistic feature space.
22. **Suitability for NLP**: **[High]** Preserves complete un-stemmed, un-tokenized natural text with punctuation and subject/body separation.
23. **Suitability for Character N-gram Analysis**: **[High]** Preserves raw casing, typos, leetspeak, and HTML syntax suitable for 3–5 char n-grams.
24. **Suitability for URL/Security Analysis**: **[High]** Contains real-world 2000s phishing links, IP hosts, and domain paths.
25. **Suitability for Explainability**: **[High]** Preserves human-readable emails with distinct vocabulary features.
26. **Suitability for Competition Demonstration**: **[High]** Gold-standard benchmark in academic anti-spam literature.

---

## 3. Enron Spam Dataset Analysis

The **Enron Spam Dataset** (Metsis, Androutsopoulos, Paliouras, 2006) was compiled by combining legitimate emails from the Enron corporate email release with spam emails from various public spam archives.

### 26-Point Evaluation

1. **Dataset Name**: Enron Spam Dataset (Enron1 through Enron6).
2. **Source/Origin**: Sourced from the FTC Enron corporate email database (Ham) and public spam archives (Spam, including SpamAssassin, Bruce Guenter's archive, and Rosal).
3. **Authenticity**: **[Verified Fact]** Real corporate correspondence (Ham) matched with real spam collection (Spam).
4. **Number of Spam Messages**: **[Verified Fact]** 17,171 spam messages across Enron1–Enron6.
5. **Number of Ham Messages**: **[Verified Fact]** 16,545 ham messages across Enron1–Enron6.
6. **Class Balance**: **[Verified Fact]** Total: 33,716 emails. ~50.9% Spam, ~49.1% Ham (Nearly balanced 1:1 ratio).
7. **Raw Email Availability**: **[Verified Fact]** Available as individual `.txt` files per email.
8. **Header Availability**: **[Verified Fact]** Minimal/Partial. Most distributions retain only basic header lines (`Subject:`, `From:`, `Date:`) while dropping full MIME headers and `Received:` path headers.
9. **Subject Availability**: **[Verified Fact]** 100% available as the first line in message files (`Subject: ...`).
10. **Body Availability**: **[Verified Fact]** 100% body text available following the subject line.
11. **HTML Availability**: **[Verified Fact]** Low/Moderate. Most raw HTML tags were stripped or sanitized during original dataset formatting.
12. **MIME/Multipart Availability**: **[Verified Fact]** Low. MIME structures converted into flat plain-text files.
13. **Encoding Diversity**: **[Verified Fact]** Standardized text encoding (mostly UTF-8 / ASCII plain text).
14. **Sender Information Availability**: **[Reasonable Inference]** Partial. Sender email addresses exist in subject/headers when preserved, but full routing paths are absent.
15. **URL Availability**: **[Verified Fact]** High. URLs within text body are preserved.
16. **Duplicate/Repeated-Message Risk**: **[Verified Fact]** High. Corporate mailing threads in Enron ham contain repeated quoted text, reply headers (`RE:`), and corporate signature blocks.
17. **Data Leakage Risk**: **[Verified Fact]** High cross-domain leakage if Ham comes exclusively from `@enron.com` domain while Spam comes from external domains. Models risk learning "enron.com" as an absolute Ham feature.
18. **Licensing/Usage Considerations**: **[Verified Fact]** Publicly released by Federal Energy Regulatory Commission (FERC) for research purposes.
19. **Accessibility/Download Method**: Direct tarball archives from AUEB NLP Group website or pre-packaged Hugging Face / Kaggle datasets.
20. **Dataset Maintenance/Status**: **[Verified Fact]** Static benchmark dataset (2006).
21. **Suitability for Supervised ML**: **[High]** Excellent for pure text-based classification algorithms.
22. **Suitability for NLP**: **[High]** Large sample size (33,716 emails) provides rich vocabulary coverage.
23. **Suitability for Character N-gram Analysis**: **[Moderate]** Effective on body text, but lacks raw HTML markup.
24. **Suitability for URL/Security Analysis**: **[Moderate]** Good for body text URLs; limited for HTML anchor tag mismatch analysis due to HTML stripping.
25. **Suitability for Explainability**: **[High]** Corporate context provides intuitive term attributions.
26. **Suitability for Competition Demonstration**: **[High]** Highly recognized benchmark in ML research.

---

## 4. Other Candidate Datasets

### A. TREC 2007 Public Spam Corpus (TREC07p)
- **Source**: Text REtrieval Conference (TREC) 2007 Spam Track.
- **Message Count**: **[Verified Fact]** 75,419 total emails (50,199 Spam, 25,220 Ham).
- **Class Balance**: ~66.6% Spam, ~33.4% Ham.
- **Key Characteristics**: Real chronologically ordered server stream received over 3 months in 2007. Preserves complete RFC 822 emails including full headers, raw HTML, attachments, and MIME boundaries.
- **Pros**: Largest raw email corpus with complete header/MIME structure; includes realistic temporal sequencing.
- **Cons**: Large file size (~250 MB compressed, 1 GB+ uncompressed); requires authorization agreement for official TREC distribution access.

### B. Ling-Spam Corpus (Androutsopoulos et al., 2000)
- **Source**: Linguist List email group (Ham) + unsolicited commercial email (Spam).
- **Message Count**: **[Verified Fact]** 2,893 total emails (481 Spam, 2,412 Ham).
- **Class Balance**: ~16.6% Spam, ~83.4% Ham.
- **Key Characteristics**: Historically significant early benchmark. Standard version provides pre-tokenized subject and body text without raw headers or HTML.
- **Pros**: Clean, lightweight baseline for NLP testing.
- **Cons**: Lacks raw email headers, MIME structures, and HTML tags; small scale; domain-restricted ham vocabulary (linguistics mailing list).

---

## 5. Dataset Comparison Table

| Metric / Feature | SpamAssassin Public Corpus | Enron Spam (Enron1–6) | TREC 2007 (TREC07p) | Ling-Spam |
|---|---|---|---|---|
| **Total Messages** | **6,047** | **33,716** | **75,419** | **2,893** |
| **Spam Count** | 1,896 | 17,171 | 50,199 | 481 |
| **Ham Count** | 4,151 | 16,545 | 25,220 | 2,412 |
| **Spam Ratio** | 31.4% | 50.9% | 66.6% | 16.6% |
| **Format** | Raw RFC 822 (.eml) | Flattened Text (.txt) | Raw RFC 822 | Tokenized Text (.txt) |
| **Full Headers** | ✅ Yes | ❌ Partial / Minimal | ✅ Yes | ❌ No |
| **MIME / HTML** | ✅ Yes | ❌ Stripped | ✅ Yes | ❌ Stripped |
| **URL Security Signals** | ✅ High | ⚠️ Moderate | ✅ High | ⚠️ Low |
| **Domain Leakage Risk** | ⚠️ Moderate | 🔴 High (`@enron.com`) | ⚠️ Low | 🔴 High (`linguistlist`) |
| **License / Access** | Public Domain / Apache | Public Domain (FERC) | TREC Research Agreement | Research Free |
| **Overall ML Suitability**| ⭐⭐⭐⭐⭐ (Best Multi-Feature) | ⭐⭐⭐⭐ (Best Large Text) | ⭐⭐⭐⭐⭐ (Best Scale) | ⭐⭐⭐ (Legacy Only) |

---

## 6. Data Leakage Risks

Data leakage occurs when information from outside the training dataset is inadvertently used to create the model, leading to artificially inflated performance metrics that fail on real-world test data.

### 1. Duplicate & Near-Duplicate Messages
- **Risk**: Spam campaigns send millions of identical or slightly perturbed emails. If duplicate emails exist across both training and test sets, the model simply memorizes specific strings rather than generalizing.
- **Mitigation**: Perform exact MD5/SHA256 content hashing and MinHash / Jaccard similarity near-duplicate deduplication prior to train/test splitting.

2. **Domain & Header Shortcut Leakage**
- **Risk**: In datasets like Enron, 100% of Ham emails contain corporate headers (`@enron.com`, `X-FileName: enron...`). An un-sanitized classifier learns `enron` as an absolute indicator of HAM, achieving fake 99.9% accuracy while collapsing on non-Enron ham.
- **Mitigation**: Explicitly strip dataset-specific header artifacts (`X-FileName`, `X-Folder`, `X-Spam-Status`, `spamassassin.taint.org`) and domain-specific header tokens from the feature matrix.

3. **Thread Contamination (RE: / FW:)**
- **Risk**: In email chains, a Ham response may quote an earlier Spam email or vice-versa, or long conversation threads may appear in both train and test folds.
- **Mitigation**: Group emails by thread ID or subject line during split creation, using `GroupKFold` or `StratifiedGroupKFold`.

4. **Vectorization Leakage**
- **Risk**: Fitting `TfidfVectorizer` or feature scalers on the entire combined dataset (train + val + test) before splitting allows test-set vocabulary and IDF statistics to leak into the training process.
- **Mitigation**: Enforce strict `scikit-learn` `Pipeline` boundaries: `.fit()` is called *only* on the training partition, while `.transform()` is called independently on validation and test partitions.

---

## 7. Combining Datasets (Multi-Corpora Synthesis)

### Scientific & Technical Appropriateness Analysis
Combining multiple corpora (e.g. SpamAssassin + Enron Spam) can expand vocabulary diversity and balance class ratios, but presents significant methodological challenges:

* **Pros**:
  - Increased sample diversity across both personal ham (SpamAssassin) and corporate ham (Enron).
  - Broader variety of spam campaign templates from different time periods.
  - Better stress-testing of classifier generalization.

* **Cons & Threats to Validity**:
  - **Distributional Shift**: Differences in header structures, formatting artifacts, and collection eras create domain shift where the model splits on source dataset rather than spam vs ham.
  - **Covariate Shift Leakage**: If Dataset A supplies 80% of Ham and Dataset B supplies 80% of Spam, the classifier learns to distinguish Dataset A from Dataset B instead of Spam from Ham.

* **Conclusion**: Combining datasets is scientifically defensible **ONLY IF** header artifacts are rigorously stripped and features are restricted to standardized subject, body, and domain security metrics.

---

## 8. Recommended Dataset

### Primary Recommended Dataset: **Apache SpamAssassin Public Corpus**

**Justification**:
1. **Raw Email Integrity**: SpamAssassin preserves full RFC 822 email structures (Subject, Body, Headers, MIME boundaries, HTML tags, and embedded URLs). This is strictly required to fulfill our project's architecture (URL security analysis, character n-gram obfuscation defense, and subject/body separation).
2. **Realistic Class Imbalance**: ~31.4% Spam / 68.6% Ham accurately reflects real-world inbox security scenarios where Ham predominates.
3. **Multi-Feature Support**: Enables true multimodal feature extraction (Word TF-IDF + Char TF-IDF + Security Indicators + Metadata).

*Secondary Option for Large-Scale Text Benchmark*: **Enron Spam (Enron1–Enron6)** as a secondary evaluation benchmark for pure body text classification.

---

## 9. Recommended Dataset Strategy

1. **Primary Corpus**: Use the **SpamAssassin Public Corpus** (6,047 raw emails) as our primary benchmark for system implementation, feature engineering, security signal extraction, and Streamlit demonstration.
2. **Preprocessing Sanitization**: Strip all synthetic evaluation headers (`X-Spam-Status`, `X-Spam-Checker-Version`, `spamassassin.taint.org`) during email ingestion to eliminate header shortcut leakage.
3. **Data Splitting**:
   - 70% Stratified Training Set
   - 15% Stratified Validation Set (for hyperparameter tuning and threshold selection)
   - 15% Untouched Holdout Test Set (strictly reserved for final experimental report)
4. **No Synthetic / Fabricated Data**: Zero synthetic email text generation.

---

## 10. Acquisition Plan

1. **Source Download**: Programmatically fetch official `.tar.bz2` archives directly from Apache SpamAssassin repository:
   - `20030228_easy_ham.tar.bz2`
   - `20030228_easy_ham_2.tar.bz2`
   - `20030228_hard_ham.tar.bz2`
   - `20030228_spam.tar.bz2`
   - `20050311_spam_2.tar.bz2`
2. **Storage Location**: Save raw archives in `data/raw/` and extract raw `.eml` / message files into labeled subdirectories (`data/raw/easy_ham/`, `data/raw/spam/`, etc.).
3. **Data Parsing & Indexing**: Parse raw emails into a structured pandas DataFrame saved to `data/processed/spamassassin_parsed.parquet` containing standard columns:
   `[message_id, subject, body_plain, body_html, raw_headers, sender, urls, is_spam]`.

---

## 11. Data Validation Plan

Before any model training occurs, execute automated data validation checks:

1. **Null / Empty Record Check**: Verify every record contains non-empty body or subject text. Drop corrupted/empty parsing outputs.
2. **Exact Duplicate Removal**: Compute SHA-256 hash of cleaned `Subject + Body` text and remove identical duplicate emails.
3. **Label Integrity Audit**: Ensure labels are binary integers (`0` for HAM, `1` for SPAM) with zero missing targets.
4. **Distribution Verification**: Confirm train/val/test splits maintain identical class ratios (~31.4% spam).
5. **Leakage Audit**: Run keyword frequency inspection on top features to confirm synthetic headers (`X-Spam-*`) are 100% absent from the vocabulary.

---

## 12. Decisions Required Before Implementation

Before proceeding to Level 3 implementation, the architect/user must confirm:

1. **Dataset Approval**: Approval of **SpamAssassin Public Corpus** as the primary dataset.
2. **Raw Archive Handling**: Confirmation to execute an automated acquisition script to download official SpamAssassin `.tar.bz2` archives into `data/raw/`.
3. **Evaluation Philosophy**: Alignment that final metrics (Precision, Recall, FPR, F0.5) will be reported strictly as experimentally measured numbers on the 15% untouched test set without pre-baked accuracy targets.

---

## Final Recommendation

For a competition-grade project, the **SpamAssassin Public Corpus** provides the strongest scientifically defensible dataset. Because it retains authentic raw RFC 822 emails—including intact MIME boundaries, HTML body markup, headers, and real phishing/spam URLs—it allows us to build and demonstrate a true multi-signal classifier (Word TF-IDF + Char N-Grams + Security Indicators) rather than a simplified body-text-only keyword filter.
