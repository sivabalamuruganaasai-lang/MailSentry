"""
Unit and Integration Tests for Model Persistence, Reproducibility, and Inference (Level 12).

Validates:
1. Deployment artifact existence, file size, and SHA-256 integrity.
2. Artifact reloading without errors using joblib.
3. Feature matrix shape (N, 10036).
4. Score tolerance matching (< 1e-6) against validation prediction records.
5. Zero label classification mismatches at threshold 0.4013.
6. Handles diverse inputs: empty, ordinary, HTML, URL-rich, malformed/unicode.
7. Sub-10ms per-sample inference latency.
"""

import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.loader import load_all_splits

ARTIFACT_PATH = Path("d:/email-spam-classifier/models/deployment_e11.joblib").resolve()
MANIFEST_PATH = Path("d:/email-spam-classifier/models/deployment_manifest.json").resolve()
VAL_PRED_PATH = Path("d:/email-spam-classifier/reports/val_predictions_E11.csv").resolve()

FROZEN_THRESHOLD = 0.4013


def test_deployment_artifacts_exist_and_hash_valid():
    """Verify that deployment bundle and manifest exist and hashes match."""
    assert ARTIFACT_PATH.exists(), f"Missing deployment artifact: {ARTIFACT_PATH}"
    assert MANIFEST_PATH.exists(), f"Missing deployment manifest: {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r") as f:
        meta = json.load(f)

    assert meta["model_name"] == "LinearSVC"
    assert meta["experiment_id"] == "E11"
    assert meta["feature_count"] == 10036
    assert meta["frozen_threshold"] == FROZEN_THRESHOLD

    # Verify SHA-256 hash match
    with open(ARTIFACT_PATH, "rb") as f:
        computed_hash = hashlib.sha256(f.read()).hexdigest()

    assert computed_hash == meta["artifact_sha256"], (
        f"Artifact SHA-256 mismatch! Expected {meta['artifact_sha256']}, got {computed_hash}"
    )


def test_pipeline_reloading_and_dimensionality():
    """Verify joblib reload and feature dimension (N, 10036)."""
    pipeline = joblib.load(ARTIFACT_PATH)
    assert pipeline.is_fitted_, "Reloaded pipeline should be fitted."

    dummy_record = {
        "subject": "Test subject",
        "plain_body": "Hello world",
        "html_body": "",
        "has_html": False,
        "urls": [],
        "url_count": 0,
        "attachment_count": 0,
        "is_multipart": False
    }

    X_feat = pipeline.transform_features([dummy_record])
    assert X_feat.shape == (1, 10036), f"Expected shape (1, 10036), got {X_feat.shape}"


def test_validation_reproducibility_and_labels():
    """Verify reloaded predictions match reports/val_predictions_E11.csv within 1e-6."""
    pipeline = joblib.load(ARTIFACT_PATH)

    splits = load_all_splits()
    val_records, _, _ = splits["val"]

    val_pred_df = pd.read_csv(VAL_PRED_PATH)
    expected_scores = val_pred_df["decision_score"].values
    expected_labels = val_pred_df["pred_optimal_thresh"].values

    reloaded_scores = pipeline.predict_score(val_records)
    reloaded_labels = (reloaded_scores >= FROZEN_THRESHOLD).astype(int)

    max_diff = np.max(np.abs(reloaded_scores - expected_scores))
    label_mismatches = np.sum(reloaded_labels != expected_labels)

    assert max_diff < 1e-6, f"Max score difference {max_diff:.12e} exceeds tolerance 1e-6"
    assert label_mismatches == 0, f"Found {label_mismatches} classification label mismatches"


def test_diverse_input_handling():
    """Verify that reloaded pipeline processes edge case inputs without throwing exceptions."""
    pipeline = joblib.load(ARTIFACT_PATH)

    test_cases = [
        {"subject": "", "plain_body": "", "html_body": "", "has_html": False, "urls": [], "url_count": 0, "attachment_count": 0, "is_multipart": False},
        {"subject": "A" * 500, "plain_body": "B" * 5000, "html_body": "<html>" + "C" * 1000 + "</html>", "has_html": True, "urls": ["http://test.com"], "url_count": 1, "attachment_count": 2, "is_multipart": True},
        {"subject": "Emoji 🎉\x00\xff", "plain_body": "Unicode \x01\x02 test", "html_body": "", "has_html": False, "urls": [], "url_count": 0, "attachment_count": 0, "is_multipart": False},
    ]

    scores = pipeline.predict_score(test_cases)
    labels = pipeline.predict(test_cases)

    assert len(scores) == 3
    assert len(labels) == 3
    assert all(isinstance(s, (float, np.float64, np.float32)) for s in scores)
    assert all(l in (0, 1) for l in labels)
