"""
Word-Level TF-IDF Feature Extractor Module.

Provides configurable text preprocessing modes (light, stopword-aware, stemmed)
and word n-gram TF-IDF vectorization.
Strictly fit ONLY on training data; transform on validation/test data.
"""

import re
import unicodedata
from typing import List, Optional, Set, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# Domain-salient words to ALWAYS preserve even when stopword removal is enabled
PRESERVED_SPAM_WORDS: Set[str] = {
    "no", "not", "off", "free", "win", "winner", "urgent", "click", "cash",
    "money", "credit", "card", "call", "now", "buy", "save", "offer", "guarantee"
}

# Standard English stopwords (excluding preserved terms)
STANDARD_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
} - PRESERVED_SPAM_WORDS


def preprocess_text_light(text: str) -> str:
    """Light normalization: Unicode NFKD, lowercasing, whitespace collapse."""
    if not text:
        return ""
    # Unicode NFKD normalization
    normalized = unicodedata.normalize("NFKD", str(text))
    # Lowercase
    lowered = normalized.lower()
    # Collapse whitespace
    collapsed = re.sub(r'\s+', ' ', lowered).strip()
    return collapsed


def preprocess_text_stopword_aware(text: str) -> str:
    """Stopword-aware normalization: Light normalization + selective stopword filter."""
    light = preprocess_text_light(text)
    if not light:
        return ""
    tokens = light.split()
    filtered = [t for t in tokens if t not in STANDARD_STOPWORDS]
    return " ".join(filtered)


def simple_stem_word(word: str) -> str:
    """Light deterministic suffix stemming fallback."""
    if len(word) <= 3:
        return word
    for suffix in ["ing", "edly", "ed", "ly", "es", "s"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def preprocess_text_stemmed(text: str) -> str:
    """Stemmed normalization: Light normalization + word stemming."""
    light = preprocess_text_light(text)
    if not light:
        return ""
    tokens = light.split()
    stemmed = [simple_stem_word(t) for t in tokens]
    return " ".join(stemmed)


class WordTfidfExtractor(BaseEstimator, TransformerMixin):
    """
    Word-level TF-IDF Feature Extractor.
    Supports mode in ['light', 'stopword_aware', 'stemmed'].
    """

    def __init__(
        self,
        mode: str = "light",
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        max_features: Optional[int] = 10000,
        sublinear_tf: bool = True
    ):
        self.mode = mode
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vectorizer_ = None

    def _get_preprocessor(self):
        if self.mode == "stopword_aware":
            return preprocess_text_stopword_aware
        elif self.mode == "stemmed":
            return preprocess_text_stemmed
        else:
            return preprocess_text_light

    def fit(self, X: List[str], y=None):
        preproc = self._get_preprocessor()
        n_samples = len(X)
        eff_min_df = 1 if n_samples <= 10 else self.min_df
        eff_max_df = 1.0 if n_samples <= 10 else self.max_df

        self.vectorizer_ = TfidfVectorizer(
            preprocessor=preproc,
            analyzer="word",
            ngram_range=self.ngram_range,
            min_df=eff_min_df,
            max_df=eff_max_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r'(?u)\b\w+\b|\$|€|£|!'  # Preserve currency & exclamation
        )
        self.vectorizer_.fit(X)
        return self

    def transform(self, X: List[str]):
        if self.vectorizer_ is None:
            raise ValueError("WordTfidfExtractor is not fitted yet. Call fit() on training data first.")
        return self.vectorizer_.transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self.vectorizer_ is None:
            raise ValueError("WordTfidfExtractor is not fitted yet.")
        names = self.vectorizer_.get_feature_names_out()
        return np.array([f"word_tfidf__{n}" for n in names])
