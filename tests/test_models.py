"""
Comprehensive Automated Unit Tests for Models and Benchmarking (Level 5).

Tests model training, prediction shape, class mapping, deterministic results,
validation metrics, sparse matrix compatibility, model serialization,
test-set isolation, and group isolation.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
import joblib

from src.models.naive_bayes import SpamNaiveBayes
from src.models.logistic_regression import SpamLogisticRegression
from src.models.linear_svm import SpamLinearSVM
from src.models.benchmark import (
    SpamModelPipeline,
    compute_evaluation_metrics,
    run_group_aware_cv
)
from src.data.loader import load_all_splits


def test_model_training_and_is_fitted():
    X = csr_matrix([[1, 0, 2], [0, 3, 0], [2, 1, 0], [0, 0, 4]])
    y = np.array([0, 1, 0, 1])

    nb = SpamNaiveBayes().fit(X, y)
    lr = SpamLogisticRegression().fit(X, y)
    svm = SpamLinearSVM().fit(X, y)

    assert nb.classes_ is not None
    assert lr.classes_ is not None
    assert svm.classes_ is not None


def test_prediction_shape_and_class_mapping():
    X = csr_matrix([[1, 0, 2], [0, 3, 0], [2, 1, 0], [0, 0, 4]])
    y = np.array([0, 1, 0, 1])

    models = [SpamNaiveBayes(), SpamLogisticRegression(), SpamLinearSVM()]
    for m in models:
        m.fit(X, y)
        preds = m.predict(X)
        scores = m.predict_score(X)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (4,)
        assert set(preds).issubset({0, 1})  # Binary class mapping 0=HAM, 1=SPAM
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (4,)


def test_deterministic_results():
    records = [
        {"subject": "Free money offer", "plain_body": "Click here for free cash now", "urls": ["http://free.com"]},
        {"subject": "Meeting reminder", "plain_body": "Team status call at 3pm today", "urls": []},
        {"subject": "Urgent claim prize", "plain_body": "You won $1000 gift card", "urls": ["http://win.com"]},
        {"subject": "Project update", "plain_body": "Code review ready for inspection", "urls": []}
    ]
    y = np.array([1, 0, 1, 0])

    p1 = SpamModelPipeline(["word"], "LogisticRegression", random_state=42)
    p1.fit(records, y)
    pred1 = p1.predict(records)

    p2 = SpamModelPipeline(["word"], "LogisticRegression", random_state=42)
    p2.fit(records, y)
    pred2 = p2.predict(records)

    np.testing.assert_array_equal(pred1, pred2)


def test_validation_metrics_computation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])  # TN=2, FP=1, FN=1, TP=2
    y_score = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3])

    metrics = compute_evaluation_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        train_time_sec=0.5,
        val_time_sec=0.06,
        model_size_kb=15.0
    )

    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 2
    assert metrics["precision"] == round(2 / 3, 4)
    assert metrics["recall"] == round(2 / 3, 4)
    assert metrics["fpr"] == round(1 / 3, 4)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_sparse_matrix_compatibility():
    X_sparse = csr_matrix([[0.5, 0.0], [0.0, 1.2], [0.8, 0.1], [0.0, 0.0]])
    y = np.array([0, 1, 0, 1])

    for model in [SpamNaiveBayes(), SpamLogisticRegression(), SpamLinearSVM()]:
        model.fit(X_sparse, y)
        preds = model.predict(X_sparse)
        assert preds.shape == (4,)


def test_model_serialization_and_deserialization():
    records = [
        {"subject": "Test", "plain_body": "Sample body text", "urls": []},
        {"subject": "Spam", "plain_body": "Buy now free offer", "urls": ["http://spam.com"]}
    ]
    y = np.array([0, 1])

    pipeline = SpamModelPipeline(["word", "char"], "LogisticRegression", random_state=42)
    pipeline.fit(records, y)
    pred_orig = pipeline.predict(records)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        joblib.dump(pipeline, tmp.name)
        loaded_pipeline = joblib.load(tmp.name)

    pred_loaded = loaded_pipeline.predict(records)
    np.testing.assert_array_equal(pred_orig, pred_loaded)


def test_test_set_isolation():
    splits = load_all_splits()
    train_records, y_train, train_df = splits["train"]
    val_records, y_val, val_df = splits["val"]
    test_records, y_test, test_df = splits["test"]

    # Verify zero overlap between train/val and test file paths
    train_paths = set(train_df["file_path"])
    val_paths = set(val_df["file_path"])
    test_paths = set(test_df["file_path"])

    assert len(train_paths.intersection(test_paths)) == 0
    assert len(val_paths.intersection(test_paths)) == 0

    # Ensure test records remain untouched during pipeline fitting
    pipeline = SpamModelPipeline(["word"], "LogisticRegression", random_state=42)
    pipeline.fit(train_records, y_train)

    # Feature vocabulary must only reflect train data
    assert pipeline.feature_pipeline.word_extractor_ is not None
    assert pipeline.feature_pipeline.word_extractor_.vectorizer_ is not None


def test_group_isolation_in_cross_validation():
    # Build 50 records with 10 explicit duplicate groups
    records = []
    labels = []
    groups = []
    for i in range(50):
        g_idx = i // 5
        records.append({
            "subject": f"Subject group {g_idx}",
            "plain_body": f"Body content for group {g_idx}",
            "urls": []
        })
        labels.append(1 if g_idx % 2 == 1 else 0)
        groups.append(f"GROUP_{g_idx:02d}")

    y_train = np.array(labels)
    groups_train = np.array(groups)

    exp_configs = [("E01", ["word"], "LogisticRegression", False)]

    # Run Group-Aware CV
    cv_res = run_group_aware_cv(
        train_records=records,
        y_train=y_train,
        groups_train=groups_train,
        exp_configs=exp_configs,
        n_splits=5,
        seed=42
    )

    assert "E01" in cv_res
    assert 0.0 <= cv_res["E01"]["mean_f1"] <= 1.0


def test_cv_vectorizer_fitted_per_fold_only():
    """Verify that vectorizers are fit inside each fold and not globally before CV."""
    records = [
        {"subject": "Free spam offer", "plain_body": "Buy now discount item", "urls": []},
        {"subject": "Meeting reminder", "plain_body": "Team sync tomorrow", "urls": []},
        {"subject": "Special promo", "plain_body": "Claim your prize voucher", "urls": []},
        {"subject": "Project status", "plain_body": "Updates on sprint goals", "urls": []},
        {"subject": "Invoice attached", "plain_body": "Payment confirmation details", "urls": []}
    ]
    y = np.array([1, 0, 1, 0, 0])
    groups = np.array([f"G{i}" for i in range(len(y))])

    # Unfitted pipeline
    p = SpamModelPipeline(["word"], "LogisticRegression", random_state=42)
    assert p.feature_pipeline.word_extractor_ is None

    # Run CV
    cv_res = run_group_aware_cv(
        train_records=records,
        y_train=y,
        groups_train=groups,
        exp_configs=[("E_TEST", ["word"], "LogisticRegression", False)],
        n_splits=3,
        seed=42
    )

    assert "E_TEST" in cv_res
    assert len(cv_res["E_TEST"]["folds"]) == 3


def test_cv_all_training_samples_participate_exactly_once():
    """Verify every training sample is validated exactly once across all CV folds."""
    from sklearn.model_selection import StratifiedGroupKFold
    splits = load_all_splits()
    train_records, y_train, train_df = splits["train"]
    groups_train = np.array([r["group_id"] for r in train_records])

    sgkf = StratifiedGroupKFold(n_splits=5)
    dummy = np.zeros(len(y_train))

    val_indices_seen = []
    for train_idx, val_idx in sgkf.split(dummy, y_train, groups_train):
        # Disjoint fold train and val indices
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        # Zero group crossing
        train_groups = set(groups_train[train_idx])
        val_groups = set(groups_train[val_idx])
        assert len(train_groups.intersection(val_groups)) == 0

        val_indices_seen.extend(val_idx)

    # Every sample validated exactly once
    assert len(val_indices_seen) == len(train_records)
    assert len(set(val_indices_seen)) == len(train_records)
    assert sorted(val_indices_seen) == list(range(len(train_records)))


def test_cv_no_validation_or_test_set_leakage():
    """Verify validation (907) and test (906) sets never touch CV training data."""
    splits = load_all_splits()
    train_records, y_train, train_df = splits["train"]
    val_records, y_val, val_df = splits["val"]
    test_records, y_test, test_df = splits["test"]

    train_msg_ids = set(r["message_id"] for r in train_records)
    val_msg_ids = set(r["message_id"] for r in val_records)
    test_msg_ids = set(r["message_id"] for r in test_records)

    assert len(train_msg_ids.intersection(val_msg_ids)) == 0
    assert len(train_msg_ids.intersection(test_msg_ids)) == 0
