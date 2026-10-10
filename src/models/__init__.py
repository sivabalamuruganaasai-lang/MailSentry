"""
Models Package for Email Spam Classifier (Level 5).

Provides candidate model wrappers, pipeline integrators, evaluation metrics,
and benchmark execution runner.
"""

from src.models.naive_bayes import SpamNaiveBayes
from src.models.logistic_regression import SpamLogisticRegression
from src.models.linear_svm import SpamLinearSVM
from src.models.benchmark import SpamModelPipeline, BenchmarkRunner, compute_evaluation_metrics
from src.models.explainability import (
    get_feature_family,
    get_global_feature_importance,
    explain_individual_prediction
)

__all__ = [
    "SpamNaiveBayes",
    "SpamLogisticRegression",
    "SpamLinearSVM",
    "SpamModelPipeline",
    "BenchmarkRunner",
    "compute_evaluation_metrics",
    "get_feature_family",
    "get_global_feature_importance",
    "explain_individual_prediction"
]
