# Feature Schema Specification Document

---

## Executive Overview
This document defines the complete feature schema for the Email Spam Classifier system. Features are categorized into four independent, modular feature families:
1. **Word TF-IDF (`word_tfidf`)**
2. **Character TF-IDF (`char_tfidf`)**
3. **Structural Features (`structural`)**
4. **URL Features (`url`)**

All feature extractors adhere strictly to the principle: **FIT ONLY on training partitions; TRANSFORM on validation and test partitions**.

---

## 1. Feature Families Summary

| Family Name | Type | Matrix Representation | Default Max Dimensions | Purpose |
|---|---|---|---|---|
| **Word TF-IDF** | Text | Sparse (`csr_matrix`) | 10,000 | Primary lexical vocabulary and 2-gram phrase patterns |
| **Character TF-IDF** | Text | Sparse (`csr_matrix`) | 15,000 | Sub-word obfuscation defense (`fr33`, `v!agra`, typos) |
| **Structural Features** | Numeric | Dense / Sparse | 19 | Metadata, length, casing, punctuation & HTML density |
| **URL Features** | Numeric | Dense / Sparse | 17 | Measurable URL structural properties (IPs, shorteners, ports) |

---

## 2. Detailed Structural Feature Schema (19 Features)

| Feature Name | Type | Source Field | Calculation Method | Imputation Rule | Inference Availability |
|---|---|---|---|---|---|
| `subject_length` | `float64` | `subject` | Character count of subject line | Default `0.0` | ✅ Yes |
| `body_length` | `float64` | `plain_body` | Character count of plain-text body | Default `0.0` | ✅ Yes |
| `total_length` | `float64` | `subject + body` | Total combined character count | Default `0.0` | ✅ Yes |
| `word_count` | `float64` | `subject + body` | Count of whitespace-separated tokens | Default `0.0` | ✅ Yes |
| `sentence_count` | `float64` | `subject + body` | Count of punctuation-split sentences | Default `0.0` | ✅ Yes |
| `uppercase_ratio` | `float64` | `subject + body` | `isupper()` character count / `total_length` | Default `0.0` | ✅ Yes |
| `digit_ratio` | `float64` | `subject + body` | `isdigit()` character count / `total_length` | Default `0.0` | ✅ Yes |
| `whitespace_ratio` | `float64` | `subject + body` | `isspace()` character count / `total_length` | Default `0.0` | ✅ Yes |
| `punctuation_ratio` | `float64` | `subject + body` | Punctuation character count / `total_length` | Default `0.0` | ✅ Yes |
| `exclamation_count` | `float64` | `subject + body` | Count of `!` characters | Default `0.0` | ✅ Yes |
| `question_count` | `float64` | `subject + body` | Count of `?` characters | Default `0.0` | ✅ Yes |
| `repeated_char_ratio` | `float64` | `subject + body` | Count of 3+ consecutive repeated characters / `total_length` | Default `0.0` | ✅ Yes |
| `html_present` | `float64` | `html_body` | `1.0` if HTML body present, else `0.0` | Default `0.0` | ✅ Yes |
| `html_tag_count` | `float64` | `html_body` | Regex count of `<...>` HTML tags | Default `0.0` | ✅ Yes |
| `url_count` | `float64` | `body text` | Count of extracted HTTP/HTTPS URLs | Default `0.0` | ✅ Yes |
| `unique_url_count` | `float64` | `body text` | Count of unique HTTP/HTTPS URLs | Default `0.0` | ✅ Yes |
| `email_address_count` | `float64` | `subject + body` | Regex count of email addresses found | Default `0.0` | ✅ Yes |
| `attachment_count` | `float64` | `MIME headers` | Count of MIME attachment parts | Default `0.0` | ✅ Yes |
| `mime_part_count` | `float64` | `MIME structure` | `2.0` if multipart email, `1.0` if singlepart | Default `1.0` | ✅ Yes |

---

## 3. Detailed URL Feature Schema (17 Features)

| Feature Name | Type | Source Field | Calculation Method | Imputation Rule | Inference Availability |
|---|---|---|---|---|---|
| `url_count` | `float64` | `urls` list | Count of total URLs in email | Default `0.0` | ✅ Yes |
| `unique_url_count` | `float64` | `urls` list | Count of distinct URLs in email | Default `0.0` | ✅ Yes |
| `http_count` | `float64` | `urls` list | Count of non-SSL `http://` links | Default `0.0` | ✅ Yes |
| `https_count` | `float64` | `urls` list | Count of SSL `https://` links | Default `0.0` | ✅ Yes |
| `avg_url_length` | `float64` | `urls` list | Average character length of URLs | Default `0.0` | ✅ Yes |
| `max_url_length` | `float64` | `urls` list | Maximum character length of URLs | Default `0.0` | ✅ Yes |
| `avg_hostname_length` | `float64` | `urls` list | Average hostname string length | Default `0.0` | ✅ Yes |
| `max_hostname_length` | `float64` | `urls` list | Maximum hostname string length | Default `0.0` | ✅ Yes |
| `avg_dot_count` | `float64` | `urls` list | Average count of `.` in hostnames | Default `0.0` | ✅ Yes |
| `max_dot_count` | `float64` | `urls` list | Maximum count of `.` in hostnames | Default `0.0` | ✅ Yes |
| `subdomain_count` | `float64` | `urls` list | Count of URLs with >2 dots in hostname | Default `0.0` | ✅ Yes |
| `ip_host_count` | `float64` | `urls` list | Count of URLs using raw IPv4/v6 host | Default `0.0` | ✅ Yes |
| `percent_encoding_count`| `float64` | `urls` list | Count of URLs with `%` character | Default `0.0` | ✅ Yes |
| `suspicious_port_count` | `float64` | `urls` list | Count of non-80/443 custom ports | Default `0.0` | ✅ Yes |
| `shortener_count` | `float64` | `urls` list | Count of known link shorteners | Default `0.0` | ✅ Yes |
| `query_string_count` | `float64` | `urls` list | Count of URLs with `?query=` parameters | Default `0.0` | ✅ Yes |
| `fragment_count` | `float64` | `urls` list | Count of URLs with `#anchor` fragments | Default `0.0` | ✅ Yes |

---

## 4. Leakage Prevention Rules

1. **Target Label Isolation**: Ground-truth target `label` (0=HAM, 1=SPAM) is strictly prohibited from feature input channels.
2. **Evaluation Header Isolation**: Headers matching `X-Spam-*`, `X-Pyzor-*`, `X-Razor-*` are stripped prior to feature extraction.
3. **Path & File Isolation**: File paths and folder names (e.g. `data/raw/spamassassin/spam/...`) are excluded.
4. **Vocabulary Fit Scope**: Vectorizers are `.fit()` strictly on training split data; `.transform()` is executed on validation/test sets without updating vectorizer vocabulary or IDF weights.
