"""
Logistic Regression Model Wrapper (Level 5).

Wraps sklearn.linear_model.LogisticRegression for spam classification.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class SpamLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic Regression classifier wrapper."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs"
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver
        )
        self.classes_ = None

    def fit(self, X, y):
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict_score(self, X) -> np.ndarray:
        """Return probability for positive class (SPAM = 1)."""
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]
