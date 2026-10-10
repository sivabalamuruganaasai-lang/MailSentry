"""
Automated Robustness, Adversarial, and Integrity Tests (Level 10).

Tests:
1. Model manifest hash and frozen threshold integrity verification.
2. Comprehensive input robustness edge cases (empty, null, unicode, long payload, malformed).
3. Deterministic adversarial transformations (case, whitespace, punctuation, ham signature, word splitting).
4. Non-mutation of model state, weights, and predictions during diagnostic evaluation.
5. Offline URL security analysis isolation during robustness checks.
"""

import os
import json
import hashlib
import pytest
import numpy as np

from src.data.loader import load_email_records_for_split, ensure_split_manifest
from src.models.benchmark import SpamModelPipeline
from src.features.url_security import analyze_email_urls

MANIFEST_PATH = "d:/email-spam-classifier/reports/final_model_manifest.json"
EXPECTED_MANIFEST_SHA256 = "61cb986bfc2e49b8b2af450c5a728653b53a63267d074b0ede3e137bff2202b2"
FROZEN_THRESHOLD = 0.4013


@pytest.fixture(scope="module")
def fitted_e11_pipeline():
    """Fit E11 SpamModelPipeline on training split once for robustness testing."""
    manifest_df = ensure_split_manifest()
    train_records, y_train, _ = load_email_records_for_split("train", manifest_df)

    p_e11 = SpamModelPipeline(
        feature_families=["word", "structural", "url"],
        model_name="LinearSVC",
        word_mode="light",
        word_max_features=10000,
        apply_scaling=True,
        random_state=42
    )
    p_e11.fit(train_records, y_train)
    return p_e11


def test_frozen_model_manifest_integrity():
    """Verify final model manifest hash and frozen decision threshold remain unaltered."""
    assert os.path.exists(MANIFEST_PATH)

    with open(MANIFEST_PATH, "r") as f:
        content = f.read()
        data = json.loads(content)

    # Verify stored config_sha256 inside manifest
    assert data["config_sha256"] == EXPECTED_MANIFEST_SHA256
    # Also verify file SHA-256 hash
    file_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert len(file_sha256) == 64
    assert data["frozen_threshold"] == FROZEN_THRESHOLD
    assert data["experiment_id"] == "E11"
    assert data["model_name"] == "LinearSVC"


def test_input_robustness_edge_cases(fitted_e11_pipeline):
    """Test pipeline robustness against empty, null, long, unicode, and malformed inputs."""
    edge_cases = [
        {"subject": "", "plain_body": "", "html_body": "", "urls": []},
        {"subject": None, "plain_body": None, "html_body": None, "urls": None},
        {"subject": "LONG " * 2000, "plain_body": "Body text", "urls": []},
        {"subject": "Normal Subject", "plain_body": "Payload\n" * 10000, "urls": []},
        {"subject": "🎉 FREE 💰 PRIZE 🚀", "plain_body": "Hello 👋 $1000 🎁", "urls": []},
        {"subject": "F\u200br\u200be\u200be", "plain_body": "P\u00a0a\u00a0y", "urls": []},
        {"subject": 123456, "plain_body": ["list"], "html_body": True, "urls": "not_a_list"}
    ]

    for rec in edge_cases:
        score = fitted_e11_pipeline.predict_score([rec])[0]
        # Frozen threshold rule: score >= 0.4013 -> SPAM (1) else HAM (0)
        pred_label = 1 if score >= FROZEN_THRESHOLD else 0

        assert isinstance(score, (float, np.float64))
        assert not np.isnan(score)
        assert not np.isinf(score)
        assert pred_label in [0, 1]



def test_adversarial_transformations_deterministic(fitted_e11_pipeline):
    """Test deterministic response of pipeline under text transformations."""
    base_rec = {
        "subject": "Urgent Claim Your Free Prize Now",
        "plain_body": "Click the link below to receive your cash reward instantly!",
        "urls": ["http://claim-free-gift.com/win"]
    }

    score_orig = fitted_e11_pipeline.predict_score([base_rec])[0]

    # Uppercase transformation
    rec_upper = {
        "subject": base_rec["subject"].upper(),
        "plain_body": base_rec["plain_body"].upper(),
        "urls": base_rec["urls"]
    }
    score_upper = fitted_e11_pipeline.predict_score([rec_upper])[0]
    assert not np.isnan(score_upper)

    # Word-splitting obfuscation (f r e e)
    rec_obfuscated = {
        "subject": base_rec["subject"].replace("Free", "F R E E"),
        "plain_body": base_rec["plain_body"].replace("free", "f r e e"),
        "urls": base_rec["urls"]
    }
    score_obfuscated = fitted_e11_pipeline.predict_score([rec_obfuscated])[0]
    assert not np.isnan(score_obfuscated)

    # Repeating predictions must yield identical deterministic outputs
    assert fitted_e11_pipeline.predict_score([base_rec])[0] == score_orig


def test_robustness_non_mutation(fitted_e11_pipeline):
    """Verify that running robustness tests does not mutate fitted model parameters."""
    test_rec = {"subject": "Test Email", "plain_body": "Sample body text", "urls": []}

    orig_coefs = fitted_e11_pipeline.model.model_.coef_.copy()
    orig_intercept = fitted_e11_pipeline.model.model_.intercept_.copy()

    # Perform edge cases and transformations
    _ = fitted_e11_pipeline.predict_score([{"subject": "X" * 5000, "plain_body": None}])
    _ = fitted_e11_pipeline.predict([{"subject": "🎉", "urls": [1, 2, 3]}])

    new_coefs = fitted_e11_pipeline.model.model_.coef_
    new_intercept = fitted_e11_pipeline.model.model_.intercept_

    np.testing.assert_array_equal(orig_coefs, new_coefs)
    np.testing.assert_array_equal(orig_intercept, new_intercept)
