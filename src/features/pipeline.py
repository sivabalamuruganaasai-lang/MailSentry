"""
Modular Feature Pipeline Builder.

Provides builder functions to instantiate independent feature families
(Word TF-IDF, Char TF-IDF, Structural, URL) or combine them using FeatureUnion.
Enforces FIT on training data only and TRANSFORM on validation/test data.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from scipy.sparse import issparse, hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

from src.features.word_tfidf import WordTfidfExtractor
from src.features.char_tfidf import CharTfidfExtractor
from src.features.structural import StructuralFeatureExtractor
from src.features.url_features import URLFeatureExtractor


class TextSelector(BaseEstimator, TransformerMixin):
    """Transformer to extract raw text strings from dicts / DataFrame records."""

    def __init__(self, key: str = "text"):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "to_dict"):
            records = X.to_dict(orient="records")
        else:
            records = X

        text_list = []
        for r in records:
            if isinstance(r, dict):
                subj = str(r.get("subject", "") or "")
                body = str(r.get("plain_body", "") or "")
                text_list.append(f"{subj} {body}".strip())
            elif isinstance(r, str):
                text_list.append(r)
            else:
                text_list.append("")
        return text_list


class RecordSelector(BaseEstimator, TransformerMixin):
    """Transformer to pass dict/DataFrame records directly to structural/url extractors."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "to_dict"):
            return X.to_dict(orient="records")
        return X


class CombinedFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Combined Feature Pipeline with independent feature family toggles.
    Allows testing feature families independently or combined.
    """

    def __init__(
        self,
        feature_families: Optional[List[str]] = None,
        word_mode: str = "light",
        word_max_features: int = 10000,
        char_max_features: int = 15000
    ):
        if feature_families is None:
            feature_families = ["word", "char", "structural", "url"]
        self.feature_families = feature_families
        self.word_mode = word_mode
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features

        self.word_extractor_ = None
        self.char_extractor_ = None
        self.structural_extractor_ = None
        self.url_extractor_ = None
        self.text_selector_ = TextSelector()
        self.record_selector_ = RecordSelector()

    def fit(self, X, y=None):
        text_data = self.text_selector_.transform(X)
        records_data = self.record_selector_.transform(X)

        if "word" in self.feature_families:
            self.word_extractor_ = WordTfidfExtractor(
                mode=self.word_mode,
                max_features=self.word_max_features
            )
            self.word_extractor_.fit(text_data)

        if "char" in self.feature_families:
            self.char_extractor_ = CharTfidfExtractor(
                max_features=self.char_max_features
            )
            self.char_extractor_.fit(text_data)

        if "structural" in self.feature_families:
            self.structural_extractor_ = StructuralFeatureExtractor(return_sparse=True)
            self.structural_extractor_.fit(records_data)

        if "url" in self.feature_families:
            self.url_extractor_ = URLFeatureExtractor(return_sparse=True)
            self.url_extractor_.fit(records_data)

        return self

    def transform(self, X):
        text_data = self.text_selector_.transform(X)
        records_data = self.record_selector_.transform(X)
        matrices = []

        if "word" in self.feature_families:
            if self.word_extractor_ is None:
                raise ValueError("Word extractor not fitted.")
            matrices.append(self.word_extractor_.transform(text_data))

        if "char" in self.feature_families:
            if self.char_extractor_ is None:
                raise ValueError("Char extractor not fitted.")
            matrices.append(self.char_extractor_.transform(text_data))

        if "structural" in self.feature_families:
            if self.structural_extractor_ is None:
                raise ValueError("Structural extractor not fitted.")
            s_mat = self.structural_extractor_.transform(records_data)
            matrices.append(s_mat if issparse(s_mat) else csr_matrix(s_mat))

        if "url" in self.feature_families:
            if self.url_extractor_ is None:
                raise ValueError("URL extractor not fitted.")
            u_mat = self.url_extractor_.transform(records_data)
            matrices.append(u_mat if issparse(u_mat) else csr_matrix(u_mat))

        if not matrices:
            raise ValueError("No feature families selected.")

        return hstack(matrices).tocsr()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        feature_names = []
        if "word" in self.feature_families and self.word_extractor_ is not None:
            feature_names.extend(self.word_extractor_.get_feature_names_out())
        if "char" in self.feature_families and self.char_extractor_ is not None:
            feature_names.extend(self.char_extractor_.get_feature_names_out())
        if "structural" in self.feature_families and self.structural_extractor_ is not None:
            feature_names.extend(self.structural_extractor_.get_feature_names_out())
        if "url" in self.feature_families and self.url_extractor_ is not None:
            feature_names.extend(self.url_extractor_.get_feature_names_out())
        return np.array(feature_names)
