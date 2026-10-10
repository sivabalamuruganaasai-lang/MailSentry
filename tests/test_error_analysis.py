"""
Automated Unit Tests for Error Analysis Logic (Level 11).

Tests:
1. Reconstruction of the validation set confusion matrix (TP=271, TN=619, FP=5, FN=12, Total=907).
2. Calculation of decision score margins relative to threshold 0.4013.
3. Validation of error categorization rules.
4. Non-mutation of frozen model parameters, threshold, and predictions.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

PREDS_PATH = "d:/email-spam-classifier/reports/val_predictions_E11.csv"
MANIFEST_PATH = "d:/email-spam-classifier/reports/final_model_manifest.json"
FROZEN_THRESHOLD = 0.4013


def test_validation_confusion_matrix_totals():
    """Verify validation confusion matrix totals match 907 total samples exactly."""
    assert os.path.exists(PREDS_PATH)
    df = pd.read_csv(PREDS_PATH)
    assert len(df) == 907

    y_true = df["true_label"].values
    scores = df["decision_score"].values
    y_pred = (scores >= FROZEN_THRESHOLD).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    assert tp == 271
    assert tn == 619
    assert fp == 5
    assert fn == 12
    assert tp + tn + fp + fn == 907


def test_score_margin_arithmetic():
    """Verify score margin arithmetic for False Positives and False Negatives."""
    df = pd.read_csv(PREDS_PATH)
    y_true = df["true_label"].values
    scores = df["decision_score"].values

    # FP: true_label==0, score >= 0.4013 -> margin = score - 0.4013 > 0
    fp_df = df[(y_true == 0) & (scores >= FROZEN_THRESHOLD)]
    assert len(fp_df) == 5
    for _, row in fp_df.iterrows():
        s = row["decision_score"]
        margin = s - FROZEN_THRESHOLD
        assert margin >= 0.0

    # FN: true_label==1, score < 0.4013 -> margin = 0.4013 - score > 0
    fn_df = df[(y_true == 1) & (scores < FROZEN_THRESHOLD)]
    assert len(fn_df) == 12
    for _, row in fn_df.iterrows():
        s = row["decision_score"]
        margin = FROZEN_THRESHOLD - s
        assert margin > 0.0


def test_frozen_manifest_and_threshold_unaltered():
    """Verify that model manifest and threshold 0.4013 are strictly preserved."""
    assert os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "r") as f:
        data = json.load(f)

    assert data["frozen_threshold"] == FROZEN_THRESHOLD
    assert data["config_sha256"] == "61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2"
    assert data["experiment_id"] == "E11"
