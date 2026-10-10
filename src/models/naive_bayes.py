"""
Multinomial Naive Bayes Model Wrapper (Level 5).

Wraps sklearn.naive_bayes.MultinomialNB for spam classification.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB


class SpamNaiveBayes(BaseEstimator, ClassifierMixin):
    """Multinomial Naive Bayes classifier wrapper."""

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.model_ = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)
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
        """Return posterior probability for positive class (SPAM = 1)."""
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]
