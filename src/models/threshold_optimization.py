"""
Decision Threshold & False-Positive Optimization Module (Level 6).

Implements fine-grained threshold sweeps, low-FPR operating point detection,
Pareto efficiency analysis, 3-tier classification (HAM/REVIEW/SPAM),
controlled model calibration, and strict validation-only protection.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV

from src.models.benchmark import SpamModelPipeline

REPORTS_DIR = Path("d:/email-spam-classifier/reports").resolve()
FIGURES_DIR = REPORTS_DIR / "figures"


def assert_validation_only(split_name: str) -> None:
    """
    Enforce strict holdout protection rule.
    Threshold optimization MUST NOT be run on the test set.
    """
    if str(split_name).lower().strip() in ["test", "test_set", "testing"]:
        raise ValueError(
            "CRITICAL SECURITY VIOLATION: Test set accessed during threshold optimization! "
            "Threshold selection must be performed strictly on validation data."
        )


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Compute full classification metrics for a given decision score threshold.
    Positive class = SPAM (1), Negative class = HAM (0).
    Rule: predict SPAM if score >= threshold else HAM.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) != len(y_score):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_score ({len(y_score)})")

    if np.isnan(y_score).any() or np.isinf(y_score).any():
        raise ValueError("y_score contains NaN or Inf values!")

    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tn = int(tn)
    fp = int(fp)
    fn = int(fn)
    tp = int(tp)

    total = tn + fp + fn + tp
    acc = float((tp + tn) / total) if total > 0 else 0.0

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    if prec + rec > 0:
        f1 = float(2 * prec * rec / (prec + rec))
    else:
        f1 = 0.0

    beta_sq = 0.5 ** 2
    if (beta_sq * prec + rec) > 0:
        f0_5 = float((1 + beta_sq) * prec * rec / (beta_sq * prec + rec))
    else:
        f0_5 = 0.0

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    tpr = rec
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f0_5": round(f0_5, 4),
        "fpr": round(fpr, 4),
        "tpr": round(tpr, 4),
        "specificity": round(specificity, 4),
        "num_spam": tp + fp,
        "num_ham": tn + fn
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exp_id: str = "",
    model_name: str = "",
    num_points: int = 1000
) -> pd.DataFrame:
    """
    Perform fine-grained threshold sweep across validation scores.
    Uses uniform linear grid combined with unique score midpoints.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    min_s = float(np.min(y_score))
    max_s = float(np.max(y_score))

    # Create dense grid covering full score span
    grid_linear = np.linspace(min_s - 0.01, max_s + 0.01, num_points)

    # Also include exact unique score boundaries
    sorted_unique = np.sort(np.unique(y_score))
    midpoints = (sorted_unique[:-1] + sorted_unique[1:]) / 2.0

    grid = np.unique(np.concatenate([grid_linear, midpoints, [min_s - 0.05, max_s + 0.05]]))
    grid.sort()

    rows = []
    for t in grid:
        metrics = compute_metrics_at_threshold(y_true, y_score, t)
        metrics["exp_id"] = exp_id
        metrics["model_name"] = model_name
        rows.append(metrics)

    df_sweep = pd.DataFrame(rows)
    return df_sweep


def find_low_fpr_operating_points(
    df_sweep: pd.DataFrame,
    target_fprs: List[float] = [0.02, 0.015, 0.01, 0.005, 0.0]
) -> pd.DataFrame:
    """
    Identify optimal operating threshold for target FPR levels (FPR <= target_fpr).
    Breaks ties by highest Precision, then Recall, then F1, then lowest Threshold.
    """
    results = []

    for target in target_fprs:
        # Filter thresholds satisfying FPR constraint
        sub = df_sweep[df_sweep["fpr"] <= target].copy()
        if sub.empty:
            continue

        # Sort to find best operating point under constraint prioritizing F0.5 and F1
        sub_sorted = sub.sort_values(
            by=["f0_5", "f1", "precision", "recall", "threshold"],
            ascending=[False, False, False, False, True]
        )
        best_row = sub_sorted.iloc[0].to_dict()
        best_row["target_fpr_label"] = f"FPR <= {target*100:.1f}%"
        results.append(best_row)

    return pd.DataFrame(results)


def compute_pareto_front(df_sweep: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pareto-efficient threshold points maximizing Precision & Recall while minimizing FPR.
    """
    # Group by unique (fpr, precision, recall) to avoid redundant evaluations
    points = df_sweep.drop_duplicates(subset=["fpr", "precision", "recall"]).copy()

    pareto_mask = []
    for idx, row in points.iterrows():
        # A point is dominated if another point has (FPR <= row.FPR AND Recall >= row.Recall AND Precision >= row.Precision)
        # with at least one strict inequality
        dominated = False
        for _, other in points.iterrows():
            if (other["fpr"] <= row["fpr"] and
                other["precision"] >= row["precision"] and
                other["recall"] >= row["recall"]):
                if (other["fpr"] < row["fpr"] or
                    other["precision"] > row["precision"] or
                    other["recall"] > row["recall"]):
                    dominated = True
                    break
        pareto_mask.append(not dominated)

    pareto_df = points[pareto_mask].sort_values(by="fpr", ascending=True).reset_index(drop=True)
    return pareto_df


def evaluate_three_tier_policy(
    y_true: np.ndarray,
    y_score: np.ndarray,
    lower_threshold: float,
    upper_threshold: float
) -> Dict[str, Any]:
    """
    Evaluate 3-tier classification policy:
      - score < lower_threshold           => Auto HAM
      - lower_threshold <= score <= upper => REVIEW (Human Verification)
      - score > upper_threshold           => Auto SPAM
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if lower_threshold > upper_threshold:
        raise ValueError(f"lower_threshold ({lower_threshold}) cannot be greater than upper_threshold ({upper_threshold})")

    total = len(y_true)
    ham_mask = y_score < lower_threshold
    review_mask = (y_score >= lower_threshold) & (y_score <= upper_threshold)
    spam_mask = y_score > upper_threshold

    n_ham_auto = int(np.sum(ham_mask))
    n_review = int(np.sum(review_mask))
    n_spam_auto = int(np.sum(spam_mask))

    pct_review = round(float(n_review / total * 100.0), 2) if total > 0 else 0.0
    auto_coverage = round(float((total - n_review) / total * 100.0), 2) if total > 0 else 0.0

    # Auto SPAM statistics
    auto_spam_tp = int(np.sum((y_true == 1) & spam_mask))
    auto_spam_fp = int(np.sum((y_true == 0) & spam_mask))
    auto_spam_prec = float(auto_spam_tp / (auto_spam_tp + auto_spam_fp)) if (auto_spam_tp + auto_spam_fp) > 0 else 1.0

    n_total_ham = int(np.sum(y_true == 0))
    auto_spam_fpr = float(auto_spam_fp / n_total_ham) if n_total_ham > 0 else 0.0

    # Review bucket composition
    review_spam_count = int(np.sum((y_true == 1) & review_mask))
    review_ham_count = int(np.sum((y_true == 0) & review_mask))

    # Auto HAM statistics
    auto_ham_tn = int(np.sum((y_true == 0) & ham_mask))
    auto_ham_fn = int(np.sum((y_true == 1) & ham_mask))

    total_spam = int(np.sum(y_true == 1))
    overall_spam_recall = float((auto_spam_tp + review_spam_count) / total_spam) if total_spam > 0 else 0.0
    auto_spam_recall = float(auto_spam_tp / total_spam) if total_spam > 0 else 0.0

    return {
        "lower_threshold": round(float(lower_threshold), 6),
        "upper_threshold": round(float(upper_threshold), 6),
        "total_emails": total,
        "auto_ham_count": n_ham_auto,
        "review_count": n_review,
        "auto_spam_count": n_spam_auto,
        "pct_review": pct_review,
        "auto_coverage_pct": auto_coverage,
        "auto_spam_tp": auto_spam_tp,
        "auto_spam_fp": auto_spam_fp,
        "auto_spam_precision": round(auto_spam_prec, 4),
        "auto_spam_fpr": round(auto_spam_fpr, 4),
        "auto_spam_recall": round(auto_spam_recall, 4),
        "review_spam_count": review_spam_count,
        "review_ham_count": review_ham_count,
        "auto_ham_tn": auto_ham_tn,
        "auto_ham_fn": auto_ham_fn,
        "overall_spam_recall": round(overall_spam_recall, 4)
    }


def run_calibration_experiment(
    pipeline: SpamModelPipeline,
    train_records: List[Dict[str, Any]],
    y_train: np.ndarray,
    val_records: List[Dict[str, Any]],
    y_val: np.ndarray,
    method: str = "sigmoid"
) -> Dict[str, Any]:
    """
    Perform a controlled Probability Calibration experiment using CalibratedClassifierCV.
    Evaluates calibrated probabilities against uncalibrated decision scores strictly on validation set.
    """
    t0 = float(pd.Timestamp.now().timestamp())

    # Extract features using fitted/or fitting pipeline
    X_train = pipeline.transform_features(train_records) if pipeline.is_fitted_ else pipeline.feature_pipeline.fit_transform(train_records)
    if pipeline.scaler is not None and not pipeline.is_fitted_:
        X_train = pipeline.scaler.fit_transform(X_train)
    elif pipeline.scaler is not None:
        X_train = pipeline.scaler.transform(X_train)

    X_val = pipeline.transform_features(val_records)

    # Base uncalibrated scores
    if not pipeline.is_fitted_:
        pipeline.model.fit(X_train, y_train)

    uncalib_scores = pipeline.model.predict_score(X_val)

    # Calibrate using pre-fit estimator or CV on train set only
    calibrated_clf = CalibratedClassifierCV(estimator=pipeline.model, method=method, cv=5)
    calibrated_clf.fit(X_train, y_train)
    t_train = float(pd.Timestamp.now().timestamp()) - t0

    t1 = float(pd.Timestamp.now().timestamp())
    calib_probas = calibrated_clf.predict_proba(X_val)[:, 1]
    t_val = float(pd.Timestamp.now().timestamp()) - t1

    # Uncalibrated ROC/PR AUC
    uncalib_roc = float(roc_auc_score(y_val, uncalib_scores))
    uncalib_pr = float(average_precision_score(y_val, uncalib_scores))

    # Calibrated ROC/PR AUC
    calib_roc = float(roc_auc_score(y_val, calib_probas))
    calib_pr = float(average_precision_score(y_val, calib_probas))

    return {
        "calibration_method": method,
        "train_time_sec": round(t_train, 4),
        "val_time_sec": round(t_val, 4),
        "uncalib_roc_auc": round(uncalib_roc, 4),
        "calib_roc_auc": round(calib_roc, 4),
        "uncalib_pr_auc": round(uncalib_pr, 4),
        "calib_pr_auc": round(calib_pr, 4),
        "calib_probas": calib_probas,
        "uncalib_scores": uncalib_scores
    }
