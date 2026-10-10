"""
Linear Support Vector Machine (LinearSVC) Model Wrapper (Level 5).

Wraps sklearn.svm.LinearSVC for spam classification.
Does NOT assume calibrated probabilities out-of-the-box. Uses decision_function
scores for ranking and threshold evaluations.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC


class SpamLinearSVM(BaseEstimator, ClassifierMixin):
    """Linear Support Vector Classifier wrapper."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        random_state: int = 42,
        dual: Any = "auto"
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.dual = dual
        self.model_ = LinearSVC(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            dual=self.dual
        )
        self.classes_ = None

    def fit(self, X, y):
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(X)

    def decision_function(self, X) -> np.ndarray:
        return self.model_.decision_function(X)

    def predict_score(self, X) -> np.ndarray:
        """Return continuous decision score (distance to hyperplane) for SPAM class."""
        scores = self.decision_function(X)
        if scores.ndim == 2:
            return scores[:, 1]
        return scores
