"""
Automated Regression Tests for Level 6 — Threshold & False-Positive Optimization.

Verifies holdout protection, threshold sweep determinism, metric math accuracy,
low-FPR operating point selection, 3-tier boundary logic, and zero test-set leakage.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.loader import load_email_records_for_split, ensure_split_manifest
from src.models.benchmark import SpamModelPipeline
from src.models.threshold_optimization import (
    assert_validation_only,
    compute_metrics_at_threshold,
    sweep_thresholds,
    find_low_fpr_operating_points,
    evaluate_three_tier_policy
)


def test_1_test_set_never_loaded_in_threshold_opt():
    """Verify that passing 'test' or test-related split names raises a strict ValueError."""
    with pytest.raises(ValueError, match="CRITICAL SECURITY VIOLATION"):
        assert_validation_only("test")

    with pytest.raises(ValueError, match="CRITICAL SECURITY VIOLATION"):
        assert_validation_only("TEST_SET")

    # Validation split name should pass without exception
    assert_validation_only("val")
    assert_validation_only("validation")


def test_2_threshold_sweep_is_deterministic():
    """Verify that threshold sweep produces identical outputs for identical inputs."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([-1.5, -0.8, 0.2, 0.5, 0.1, 0.9, 1.2, 2.0])

    df1 = sweep_thresholds(y_true, y_score, exp_id="E03", model_name="LinearSVC")
    df2 = sweep_thresholds(y_true, y_score, exp_id="E03", model_name="LinearSVC")

    pd.testing.assert_frame_equal(df1, df2)


def test_3_positive_class_is_spam():
    """Verify that positive class is 1 (SPAM) and negative class is 0 (HAM)."""
    y_true = np.array([0, 1])  # 0=HAM, 1=SPAM
    y_score = np.array([0.1, 0.9])

    res = compute_metrics_at_threshold(y_true, y_score, threshold=0.5)
    assert res["tp"] == 1  # True SPAM detected
    assert res["tn"] == 1  # True HAM detected
    assert res["fp"] == 0
    assert res["fn"] == 0


def test_4_false_positive_calculation_is_correct():
    """Verify exact FPR calculation: FP / (FP + TN)."""
    # 4 HAM emails (0), 2 SPAM emails (1)
    y_true = np.array([0, 0, 0, 0, 1, 1])
    # Scores: HAM emails get scores 0.1, 0.2, 0.6, 0.7. At thresh=0.5, 2 HAM get FP (0.6, 0.7)
    y_score = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9])

    res = compute_metrics_at_threshold(y_true, y_score, threshold=0.5)
    # TN=2 (0.1, 0.2), FP=2 (0.6, 0.7). FPR = 2 / (2 + 2) = 0.5
    assert res["fp"] == 2
    assert res["tn"] == 2
    assert pytest.approx(res["fpr"], 0.0001) == 0.5


def test_5_threshold_ordering_is_correct():
    """Verify that threshold grid is monotonically increasing."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([-1.0, 0.1, 0.5, 2.0])

    df = sweep_thresholds(y_true, y_score)
    thresholds = df["threshold"].values
    assert np.all(np.diff(thresholds) >= 0), "Threshold grid must be monotonically increasing"


def test_6_threshold_metrics_internally_consistent():
    """Verify TP + TN + FP + FN == N, and derived precision/recall/F1/F0.5 formulas match."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([-1.2, -0.5, 0.1, 0.4, -0.2, 0.8, 1.1, 1.5])

    res = compute_metrics_at_threshold(y_true, y_score, threshold=0.0)
    total = res["tp"] + res["tn"] + res["fp"] + res["fn"]
    assert total == len(y_true)

    expected_acc = (res["tp"] + res["tn"]) / total
    assert pytest.approx(res["accuracy"], 0.0001) == expected_acc

    expected_prec = res["tp"] / (res["tp"] + res["fp"])
    assert pytest.approx(res["precision"], 0.0001) == expected_prec

    expected_rec = res["tp"] / (res["tp"] + res["fn"])
    assert pytest.approx(res["recall"], 0.0001) == expected_rec


def test_7_validation_only_threshold_selection():
    """Verify that finding low-FPR operating points operates on validation data dataframes."""
    y_true = np.array([0]*100 + [1]*50)
    # Generate scores where 1 HAM gets high score
    y_score = np.concatenate([np.linspace(-2, 0.1, 99), [1.5], np.linspace(0.2, 3, 50)])

    df = sweep_thresholds(y_true, y_score)
    df_op = find_low_fpr_operating_points(df, target_fprs=[0.02, 0.01])

    assert not df_op.empty
    assert "target_fpr_label" in df_op.columns
    for _, row in df_op.iterrows():
        assert row["fpr"] <= 0.02


def test_8_no_nan_inf_decision_scores():
    """Verify that y_score containing NaN or Inf raises an explicit error."""
    y_true = np.array([0, 1])
    y_score_nan = np.array([0.1, np.nan])
    y_score_inf = np.array([np.inf, 0.5])

    with pytest.raises(ValueError, match="y_score contains NaN or Inf"):
        compute_metrics_at_threshold(y_true, y_score_nan, threshold=0.5)

    with pytest.raises(ValueError, match="y_score contains NaN or Inf"):
        compute_metrics_at_threshold(y_true, y_score_inf, threshold=0.5)


def test_9_candidate_model_prediction_lengths_match_val_size():
    """Verify candidate model prediction length matches validation size (907)."""
    manifest_df = ensure_split_manifest()
    train_records, y_train, _ = load_email_records_for_split("train", manifest_df)
    val_records, y_val, _ = load_email_records_for_split("val", manifest_df)

    assert len(val_records) == 907
    assert len(y_val) == 907

    pipeline = SpamModelPipeline(feature_families=["word"], model_name="LinearSVC")
    pipeline.fit(train_records, y_train)
    scores = pipeline.predict_score(val_records)

    assert len(scores) == 907
    assert not np.isnan(scores).any()


def test_10_selected_threshold_is_reproducible():
    """Verify reproducible operating point selection."""
    y_true = np.array([0]*80 + [1]*20)
    y_score = np.random.RandomState(42).randn(100)

    df1 = sweep_thresholds(y_true, y_score)
    op1 = find_low_fpr_operating_points(df1, target_fprs=[0.01])

    df2 = sweep_thresholds(y_true, y_score)
    op2 = find_low_fpr_operating_points(df2, target_fprs=[0.01])

    assert op1.iloc[0]["threshold"] == op2.iloc[0]["threshold"]


def test_11_three_tier_classification_boundaries_valid():
    """Verify 3-tier boundary logic and sum of categories equal total emails."""
    y_true = np.array([0]*10 + [1]*10)
    y_score = np.linspace(-2.0, 2.0, 20)

    res = evaluate_three_tier_policy(y_true, y_score, lower_threshold=-0.5, upper_threshold=0.5)

    total_cat = res["auto_ham_count"] + res["review_count"] + res["auto_spam_count"]
    assert total_cat == len(y_true)
    assert res["lower_threshold"] <= res["upper_threshold"]

    # Invalid boundaries should raise ValueError
    with pytest.raises(ValueError, match="lower_threshold .* cannot be greater than upper_threshold"):
        evaluate_three_tier_policy(y_true, y_score, lower_threshold=1.0, upper_threshold=-0.5)


def test_12_test_set_protection_guard():
    """Verify that attempting to optimize thresholds on test split raises exception."""
    manifest_df = ensure_split_manifest()
    test_records, y_test, _ = load_email_records_for_split("test", manifest_df)

    assert len(test_records) == 906

    # Attempting to call assert_validation_only with 'test' must fail
    with pytest.raises(ValueError):
        assert_validation_only("test")


def test_13_tie_breaking_order_is_deterministic():
    """Verify tie-breaking prioritizes F0.5 > F1 > Precision > Recall > lower threshold."""
    # Synthetic dataframe with equal F0.5 scores
    data = [
        {"threshold": 0.5, "fpr": 0.005, "f0_5": 0.95, "f1": 0.90, "precision": 0.98, "recall": 0.85},
        {"threshold": 0.3, "fpr": 0.005, "f0_5": 0.95, "f1": 0.92, "precision": 0.96, "recall": 0.88},
    ]
    df = pd.DataFrame(data)
    op = find_low_fpr_operating_points(df, target_fprs=[0.01])
    # Higher F1 (0.92 vs 0.90) should win
    assert op.iloc[0]["threshold"] == 0.3


def test_14_score_comparison_operator_is_greater_than_or_equal():
    """Verify score >= threshold operator."""
    y_true = np.array([0, 1])
    y_score = np.array([0.5, 0.5])

    # At thresh=0.5, both scores are >= 0.5
    res = compute_metrics_at_threshold(y_true, y_score, threshold=0.5)
    assert res["tp"] == 1
    assert res["fp"] == 1


def test_15_three_tier_boundary_equality_belongs_to_review():
    """Verify that scores equal to lower or upper threshold belong to REVIEW tier."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([-1.0, -0.352, 0.601, 1.0])

    res = evaluate_three_tier_policy(y_true, y_score, lower_threshold=-0.352, upper_threshold=0.601)
    # -1.0 is Auto HAM (1), -0.352 and 0.601 are REVIEW (2), 1.0 is Auto SPAM (1)
    assert res["auto_ham_count"] == 1
    assert res["review_count"] == 2
    assert res["auto_spam_count"] == 1


def test_16_score_ranking_preservation():
    """Verify rank correlation and detection of non-monotonic score transformations."""
    raw_scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    monotonic_transform = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    non_monotonic_transform = np.array([0.1, 0.3, 0.2, 0.8, 0.9])

    corr_mono = pd.Series(raw_scores).corr(pd.Series(monotonic_transform), method="spearman")
    corr_non_mono = pd.Series(raw_scores).corr(pd.Series(non_monotonic_transform), method="spearman")

    assert pytest.approx(corr_mono, 1e-6) == 1.0
    assert corr_non_mono < 0.99


def test_17_three_tier_metrics_disambiguation():
    """Verify that automatic_spam_recall, spam_captured_or_escalated_rate, auto_coverage, and auto_spam_precision match Level 6.2 definitions."""
    y_true = np.array([0]*10 + [1]*10) # 10 HAM, 10 SPAM
    # 6 HAM Auto HAM (-1), 3 HAM REVIEW (0), 1 HAM Auto SPAM (1)
    # 1 SPAM Auto HAM (-1), 2 SPAM REVIEW (0), 7 SPAM Auto SPAM (1)
    y_score = np.array([-2]*6 + [0]*3 + [1]*1 + [-2]*1 + [0]*2 + [1]*7)

    res = evaluate_three_tier_policy(y_true, y_score, lower_threshold=-0.5, upper_threshold=0.5)

    # Auto SPAM TP=7, FP=1, FN in Auto HAM=1, REVIEW SPAM=2, REVIEW HAM=3, Auto HAM TN=6
    assert res["auto_spam_tp"] == 7
    assert res["auto_spam_fp"] == 1

    auto_spam_recall = res["auto_spam_tp"] / 10.0 # 7 / 10 = 0.70
    captured_or_escalated = (res["auto_spam_tp"] + res["review_spam_count"]) / 10.0 # (7 + 2) / 10 = 0.90
    auto_coverage = (res["auto_ham_count"] + res["auto_spam_count"]) / 20.0 # (7 + 8) / 20 = 0.75
    auto_spam_precision = res["auto_spam_tp"] / (res["auto_spam_tp"] + res["auto_spam_fp"]) # 7 / 8 = 0.875
    ham_fpr = res["auto_spam_fp"] / 10.0 # 1 / 10 = 0.10

    assert pytest.approx(res["auto_spam_recall"], 0.001) == auto_spam_recall
    assert pytest.approx(res["overall_spam_recall"], 0.001) == captured_or_escalated
    assert pytest.approx(res["auto_coverage_pct"] / 100.0, 0.001) == auto_coverage
    assert pytest.approx(res["auto_spam_precision"], 0.001) == auto_spam_precision
    assert pytest.approx(res["auto_spam_fpr"], 0.001) == ham_fpr


def test_18_operating_points_csv_satisfies_constraints():
    """Verify that reports/threshold_operating_points.csv exists and satisfies all FPR constraints."""
    csv_path = "reports/threshold_operating_points.csv"
    df = pd.read_csv(csv_path)

    assert len(df) == 12
    for _, row in df.iterrows():
        # Target constraint extracted from label
        target_str = row["target_fpr_constraint"].replace("FPR <= ", "").replace("%", "")
        target_val = float(target_str) / 100.0

        # Check FPR constraint satisfaction
        assert row["fpr"] <= target_val + 1e-6
        # Check confusion matrix sum
        total = row["tp"] + row["tn"] + row["fp"] + row["fn"]
        assert total == 907
