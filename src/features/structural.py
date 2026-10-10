"""
Structural Feature Extractor Module.

Extracts deterministic, explainable numerical structural and metadata signals per email.
Fully handles missing/null attributes safely and outputs dense/sparse numeric matrix.
"""

import re
import string
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    "subject_length",
    "body_length",
    "total_length",
    "word_count",
    "sentence_count",
    "uppercase_ratio",
    "digit_ratio",
    "whitespace_ratio",
    "punctuation_ratio",
    "exclamation_count",
    "question_count",
    "repeated_char_ratio",
    "html_present",
    "html_tag_count",
    "url_count",
    "unique_url_count",
    "email_address_count",
    "attachment_count",
    "mime_part_count"
]

EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
HTML_TAG_REGEX = re.compile(r'<[^>]+>')


def compute_structural_vector(record: Dict[str, Any]) -> List[float]:
    """Compute structural numerical feature vector for a single email record."""
    subj = str(record.get("subject", "") or "")
    body = str(record.get("plain_body", "") or "")
    html = str(record.get("html_body", "") or "")

    combined_text = subj + " " + body
    total_len = len(combined_text)

    # 1. Length & Count Metrics
    subj_len = float(len(subj))
    body_len = float(len(body))
    tot_len = float(total_len)

    words = combined_text.split()
    word_cnt = float(len(words))

    # Sentences split by [.!?]
    sentences = [s for s in re.split(r'[.!?]+', combined_text) if s.strip()]
    sentence_cnt = float(len(sentences))

    # 2. Character Ratios
    if total_len > 0:
        uppercase_cnt = sum(1 for c in combined_text if c.isupper())
        digit_cnt = sum(1 for c in combined_text if c.isdigit())
        whitespace_cnt = sum(1 for c in combined_text if c.isspace())
        punct_cnt = sum(1 for c in combined_text if c in string.punctuation)

        uppercase_ratio = uppercase_cnt / total_len
        digit_ratio = digit_cnt / total_len
        whitespace_ratio = whitespace_cnt / total_len
        punct_ratio = punct_cnt / total_len
    else:
        uppercase_ratio = 0.0
        digit_ratio = 0.0
        whitespace_ratio = 0.0
        punct_ratio = 0.0

    excl_cnt = float(combined_text.count("!"))
    quest_cnt = float(combined_text.count("?"))

    # Repeated characters e.g. "freeeeee" (3+ same consecutive chars)
    repeated_chars = sum(1 for match in re.finditer(r'(.)\1{2,}', combined_text))
    repeated_char_ratio = (repeated_chars / total_len) if total_len > 0 else 0.0

    # 3. HTML & Attachment Indicators
    has_html_val = bool(record.get("has_html", False) or bool(html))
    html_present = 1.0 if has_html_val else 0.0
    html_tags = float(len(HTML_TAG_REGEX.findall(html)))

    # 4. URL & Email Counts
    urls = record.get("urls", [])
    if not isinstance(urls, list):
        urls = []
    url_cnt = float(record.get("url_count", len(urls)))
    unique_url_cnt = float(len(set(urls)))

    email_addrs = EMAIL_REGEX.findall(combined_text)
    email_addr_cnt = float(len(set(email_addrs)))

    attachment_cnt = float(record.get("attachment_count", 0))
    is_multipart = bool(record.get("is_multipart", False))
    mime_part_cnt = 2.0 if is_multipart else 1.0

    return [
        subj_len,
        body_len,
        tot_len,
        word_cnt,
        sentence_cnt,
        uppercase_ratio,
        digit_ratio,
        whitespace_ratio,
        punct_ratio,
        excl_cnt,
        quest_cnt,
        repeated_char_ratio,
        html_present,
        html_tags,
        url_cnt,
        unique_url_cnt,
        email_addr_cnt,
        attachment_cnt,
        mime_part_cnt
    ]


class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn Transformer for extracting numerical structural email features.
    Outputs a CSR sparse matrix or dense NumPy array.
    """

    def __init__(self, return_sparse: bool = False):
        self.return_sparse = return_sparse

    def fit(self, X: Union[List[Dict[str, Any]], pd.DataFrame], y=None):
        return self

    def transform(self, X: Union[List[Dict[str, Any]], pd.DataFrame]):
        if isinstance(X, pd.DataFrame):
            records = X.to_dict(orient="records")
        else:
            records = X

        feature_matrix = []
        for rec in records:
            if not isinstance(rec, dict):
                rec = {}
            vec = compute_structural_vector(rec)
            feature_matrix.append(vec)

        arr = np.array(feature_matrix, dtype=np.float64)
        # Handle any possible NaN / Inf safely
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if self.return_sparse:
            return csr_matrix(arr)
        return arr

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([f"structural__{name}" for name in FEATURE_NAMES])
