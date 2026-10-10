"""
Character-Level TF-IDF Feature Extractor Module.

Extracts character n-grams (3 to 5-grams) to capture sub-word obfuscations,
letter substitutions (e.g. 'v!agra', 'fr33'), casing tricks, and typos.
Strictly fit ONLY on training data; transform on validation/test data.
"""

import unicodedata
import re
from typing import List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text_char(text: str) -> str:
    """Character n-gram preprocessor: Unicode NFKD, lowercasing, whitespace collapse."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text))
    lowered = normalized.lower()
    collapsed = re.sub(r'\s+', ' ', lowered).strip()
    return collapsed


class CharTfidfExtractor(BaseEstimator, TransformerMixin):
    """
    Character-Level TF-IDF Feature Extractor.
    Uses 'char_wb' analyzer (character n-grams within word boundaries) for obfuscation robustness.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        min_df: int = 3,
        max_df: float = 0.95,
        max_features: Optional[int] = 15000,
        sublinear_tf: bool = True
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vectorizer_ = None

    def fit(self, X: List[str], y=None):
        n_samples = len(X)
        eff_min_df = 1 if n_samples <= 10 else self.min_df
        eff_max_df = 1.0 if n_samples <= 10 else self.max_df

        self.vectorizer_ = TfidfVectorizer(
            preprocessor=preprocess_text_char,
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=eff_min_df,
            max_df=eff_max_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf
        )
        self.vectorizer_.fit(X)
        return self

    def transform(self, X: List[str]):
        if self.vectorizer_ is None:
            raise ValueError("CharTfidfExtractor is not fitted yet. Call fit() on training data first.")
        return self.vectorizer_.transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self.vectorizer_ is None:
            raise ValueError("CharTfidfExtractor is not fitted yet.")
        names = self.vectorizer_.get_feature_names_out()
        return np.array([f"char_tfidf__{n}" for n in names])
