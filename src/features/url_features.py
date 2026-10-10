"""
URL Feature Extractor Module.

Extracts deterministic, measurable URL structural properties per email.
Separates raw URL structural indicators from ML classification.
Handles empty URL lists and malformed URLs gracefully.
"""

import re
from typing import List, Dict, Any, Union
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

KNOWN_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly",
    "goo.gl", "bit.do", "mcaf.ee", "su.pr", "tiny.cc"
}

IP_HOST_REGEX = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$|^\s*\[[0-9a-fA-F:]+\]\s*$')

URL_FEATURE_NAMES = [
    "url_count",
    "unique_url_count",
    "http_count",
    "https_count",
    "avg_url_length",
    "max_url_length",
    "avg_hostname_length",
    "max_hostname_length",
    "avg_dot_count",
    "max_dot_count",
    "subdomain_count",
    "ip_host_count",
    "percent_encoding_count",
    "suspicious_port_count",
    "shortener_count",
    "query_string_count",
    "fragment_count"
]


def extract_url_vector(urls: List[str]) -> List[float]:
    """Extract numerical structural properties from a list of URLs."""
    if not isinstance(urls, list) or not urls:
        return [0.0] * len(URL_FEATURE_NAMES)

    url_count = float(len(urls))
    unique_url_count = float(len(set(urls)))

    http_cnt = 0.0
    https_cnt = 0.0
    url_lengths = []
    hostname_lengths = []
    dot_counts = []
    subdomain_cnt = 0.0
    ip_host_cnt = 0.0
    percent_enc_cnt = 0.0
    suspicious_port_cnt = 0.0
    shortener_cnt = 0.0
    query_str_cnt = 0.0
    fragment_cnt = 0.0

    for url in urls:
        url_str = str(url).strip()
        if not url_str:
            continue

        url_lengths.append(len(url_str))

        if url_str.lower().startswith("https://"):
            https_cnt += 1.0
        elif url_str.lower().startswith("http://"):
            http_cnt += 1.0

        if "%" in url_str:
            percent_enc_cnt += 1.0

        try:
            parsed = urlparse(url_str)
            host = parsed.netloc or parsed.path.split("/")[0]

            # Strip port if present
            hostname = host.split(":")[0].lower()
            hostname_lengths.append(len(hostname))

            dots = hostname.count(".")
            dot_counts.append(dots)

            if dots > 2:
                subdomain_cnt += 1.0

            # IP host check
            if IP_HOST_REGEX.match(hostname):
                ip_host_cnt += 1.0

            # Known shortener check
            if any(shortener in hostname for shortener in KNOWN_URL_SHORTENERS):
                shortener_cnt += 1.0

            # Suspicious non-standard port check (e.g. :8080, :8000, :8888, :8443)
            if ":" in host:
                port_part = host.split(":")[-1]
                if port_part.isdigit() and port_part not in ["80", "443"]:
                    suspicious_port_cnt += 1.0

            if parsed.query:
                query_str_cnt += 1.0
            if parsed.fragment:
                fragment_cnt += 1.0

        except Exception:
            hostname_lengths.append(0)
            dot_counts.append(0)

    avg_url_len = float(np.mean(url_lengths)) if url_lengths else 0.0
    max_url_len = float(np.max(url_lengths)) if url_lengths else 0.0

    avg_host_len = float(np.mean(hostname_lengths)) if hostname_lengths else 0.0
    max_host_len = float(np.max(hostname_lengths)) if hostname_lengths else 0.0

    avg_dots = float(np.mean(dot_counts)) if dot_counts else 0.0
    max_dots = float(np.max(dot_counts)) if dot_counts else 0.0

    return [
        url_count,
        unique_url_count,
        http_cnt,
        https_cnt,
        avg_url_len,
        max_url_len,
        avg_host_len,
        max_host_len,
        avg_dots,
        max_dots,
        subdomain_cnt,
        ip_host_cnt,
        percent_enc_cnt,
        suspicious_port_cnt,
        shortener_cnt,
        query_str_cnt,
        fragment_cnt
    ]


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn Transformer for extracting numerical URL features.
    Outputs a CSR sparse matrix or dense NumPy array.
    """

    def __init__(self, return_sparse: bool = False):
        self.return_sparse = return_sparse

    def fit(self, X: Union[List[Dict[str, Any]], List[List[str]], pd.DataFrame], y=None):
        return self

    def transform(self, X: Union[List[Dict[str, Any]], List[List[str]], pd.DataFrame]):
        if isinstance(X, pd.DataFrame):
            records = X.to_dict(orient="records")
        else:
            records = X

        feature_matrix = []
        for rec in records:
            if isinstance(rec, dict):
                urls = rec.get("urls", [])
            elif isinstance(rec, list):
                urls = rec
            else:
                urls = []
            vec = extract_url_vector(urls)
            feature_matrix.append(vec)

        arr = np.array(feature_matrix, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if self.return_sparse:
            return csr_matrix(arr)
        return arr

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([f"url_feature__{name}" for name in URL_FEATURE_NAMES])
