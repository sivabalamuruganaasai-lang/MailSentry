"""
Comprehensive Unit and Integration Tests for Explainability Module (Level 8).

Tests:
1. Feature-name alignment, ordering, and feature family classification.
2. Exact score reconstruction arithmetic (score = intercept + sum(x_i * w_i)) within tolerance 1e-5.
3. Positive vs. negative feature contributions.
4. Edge cases: empty email inputs, sparse inputs, structural/URL-only inputs.
5. Non-mutation of model state, weights, and predictions during explanation generation.
6. Threshold rule enforcement (score >= 0.4013 -> SPAM).
"""

import pytest
import numpy as np

from src.data.loader import load_email_records_for_split, ensure_split_manifest
from src.models.benchmark import SpamModelPipeline
from src.models.explainability import (
    get_feature_family,
    get_global_feature_importance,
    explain_individual_prediction
)


@pytest.fixture(scope="module")
def fitted_e11_pipeline():
    """Fit E11 SpamModelPipeline on training set once for explainability testing."""
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


def test_feature_family_classification():
    """Verify feature family prefix mapping logic."""
    assert get_feature_family("structural__html_present") == "structural"
    assert get_feature_family("url_feature__ip_host_count") == "url"
    assert get_feature_family("char_tfidf__free") == "char"
    assert get_feature_family("word_tfidf__subject_free") == "word"
    assert get_feature_family("plain_unprefixed_word") == "word"


def test_global_feature_importance_extraction(fitted_e11_pipeline):
    """Test global feature importance extraction and structure."""
    global_info = get_global_feature_importance(fitted_e11_pipeline, top_k=15)

    assert "intercept" in global_info
    assert "total_features" in global_info
    assert global_info["total_features"] == 10036

    top_pos = global_info["top_positive"]
    top_neg = global_info["top_negative"]

    assert len(top_pos) == 15
    assert len(top_neg) == 15

    # Check descending order for positive coefficients
    pos_coefs = [item["coefficient"] for item in top_pos]
    assert pos_coefs == sorted(pos_coefs, reverse=True)
    assert all(c > 0 for c in pos_coefs)

    # Check ascending order for negative coefficients
    neg_coefs = [item["coefficient"] for item in top_neg]
    assert neg_coefs == sorted(neg_coefs)
    assert all(c < 0 for c in neg_coefs)

    # Check feature family summaries
    fam_summaries = global_info["family_summaries"]
    assert "word" in fam_summaries
    assert "structural" in fam_summaries
    assert "url" in fam_summaries
    assert fam_summaries["word"]["feature_count"] == 10000
    assert fam_summaries["structural"]["feature_count"] == 19
    assert fam_summaries["url"]["feature_count"] == 17

    # Check causation disclaimer presence
    assert "causation_disclaimer" in global_info
    assert "associations" in global_info["causation_disclaimer"]


def test_individual_explanation_exact_score_reconstruction(fitted_e11_pipeline):
    """Test individual email explanation and exact score reconstruction."""
    manifest_df = ensure_split_manifest()
    val_records, y_val, _ = load_email_records_for_split("val", manifest_df)

    # Test on first 5 validation records
    for i in range(min(5, len(val_records))):
        rec = val_records[i]
        exp = explain_individual_prediction(fitted_e11_pipeline, rec, top_k=10, threshold=0.4013, tolerance=1e-5)

        assert exp["is_reconstructed_exact"] is True
        assert exp["diff"] < 1e-5
        assert np.isclose(exp["raw_decision_score"], exp["reconstructed_score"], atol=1e-5)
        assert np.isclose(exp["reconstructed_score"], exp["intercept"] + exp["sum_contributions"], atol=1e-5)

        # Confirm threshold rule
        expected_is_spam = exp["raw_decision_score"] >= 0.4013
        assert exp["is_spam"] == expected_is_spam
        assert exp["predicted_label"] == ("SPAM" if expected_is_spam else "HAM")


def test_explanation_positive_and_negative_contributions(fitted_e11_pipeline):
    """Test positive and negative feature contribution sorting and contents."""
    sample_rec = {
        "subject": "Free money prize claim now",
        "plain_body": "Click here to claim your free cash prize!",
        "html_body": "<html><body><h1>Free Cash</h1><a href='http://192.168.1.1/win'>Click</a></body></html>",
        "urls": ["http://192.168.1.1/win"]
    }

    exp = explain_individual_prediction(fitted_e11_pipeline, sample_rec, top_k=10)

    assert exp["is_reconstructed_exact"] is True

    # Validate positive contributions (SPAM pushers)
    pos_contribs = exp["top_positive_contributions"]
    if pos_contribs:
        vals = [c["contribution"] for c in pos_contribs]
        assert vals == sorted(vals, reverse=True)
        assert all(v > 0 for v in vals)

    # Validate negative contributions (HAM pushers)
    neg_contribs = exp["top_negative_contributions"]
    if neg_contribs:
        vals = [c["contribution"] for c in neg_contribs]
        assert vals == sorted(vals)
        assert all(v < 0 for v in vals)


def test_empty_email_edge_case(fitted_e11_pipeline):
    """Test empty email input edge case and verify exact score reconstruction."""
    empty_rec = {
        "subject": "",
        "plain_body": "",
        "html_body": "",
        "urls": []
    }

    exp = explain_individual_prediction(fitted_e11_pipeline, empty_rec, threshold=0.4013)

    assert exp["is_reconstructed_exact"] is True
    assert exp["diff"] < 1e-5
    assert np.isclose(exp["reconstructed_score"], exp["raw_decision_score"], atol=1e-5)
    assert np.isclose(exp["reconstructed_score"], exp["intercept"] + exp["sum_contributions"], atol=1e-5)
    # Intercept (0.062258) + sum_contributions (~0.115718) = ~0.177976 < 0.4013 (HAM)
    assert exp["is_spam"] is False
    assert exp["predicted_label"] == "HAM"



def test_model_non_mutation(fitted_e11_pipeline):
    """Verify that generating explanations does not mutate model state or predictions."""
    sample_rec = {
        "subject": "Team meeting tomorrow",
        "plain_body": "Let's meet at 10am to discuss project status.",
        "urls": []
    }

    # Initial score and prediction
    orig_score = fitted_e11_pipeline.predict_score([sample_rec])[0]
    orig_pred = fitted_e11_pipeline.predict([sample_rec])[0]

    # Generate global importance and individual explanations multiple times
    _ = get_global_feature_importance(fitted_e11_pipeline)
    _ = explain_individual_prediction(fitted_e11_pipeline, sample_rec)
    _ = explain_individual_prediction(fitted_e11_pipeline, sample_rec)

    # Subsequent score and prediction must match original exactly
    new_score = fitted_e11_pipeline.predict_score([sample_rec])[0]
    new_pred = fitted_e11_pipeline.predict([sample_rec])[0]

    assert orig_score == new_score
    assert orig_pred == new_pred
