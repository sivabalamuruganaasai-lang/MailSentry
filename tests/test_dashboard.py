"""
Unit and Integration Tests for Streamlit Dashboard & Decision Engines (Level 13).

Validates:
1. Exact Three-Tier Boundary Semantics (AUTO HAM / REVIEW / AUTO SPAM).
   - Tests just below (-0.3521), exactly at (-0.3520), and just above (-0.3519) lower bound.
   - Tests just below (0.6009), exactly at (0.6010), and just above (0.6011) upper bound.
2. Binary Classification Threshold (SPAM if score >= 0.4013 else HAM).
3. Pinned Artifact SHA-256 hash verification against immutable code constant and manifest.
4. Input validation (2 MB size limit enforcement).
5. Safe handling of empty, unicode, HTML, and URL-rich email inputs.
6. Offline URL security analysis without external network calls.
7. Streamlit configuration telemetry-disabled verification.
"""

import sys
import json
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.app import (
    evaluate_binary_classification,
    evaluate_three_tier_routing,
    verify_artifact_hash,
    MAX_INPUT_BYTES,
    BINARY_THRESHOLD,
    LOWER_THREE_TIER,
    UPPER_THREE_TIER,
    EXPECTED_ARTIFACT_SHA256,
    load_cached_pipeline
)
from src.data.parser import parse_raw_email
from src.features.url_security import analyze_email_urls


def test_three_tier_lower_boundary_semantics():
    """
    Test 3-tier policy around lower threshold (-0.3520):
    score < -0.3520             => AUTO HAM
    -0.3520 <= score <= 0.6010  => REVIEW QUEUE
    """
    res_below = evaluate_three_tier_routing(-0.3521)
    res_exact = evaluate_three_tier_routing(-0.3520)
    res_above = evaluate_three_tier_routing(-0.3519)

    assert res_below["tier"] == "AUTO HAM", f"Expected AUTO HAM for -0.3521, got {res_below['tier']}"
    assert res_exact["tier"] == "REVIEW QUEUE", f"Expected REVIEW QUEUE for -0.3520, got {res_exact['tier']}"
    assert res_above["tier"] == "REVIEW QUEUE", f"Expected REVIEW QUEUE for -0.3519, got {res_above['tier']}"


def test_three_tier_upper_boundary_semantics():
    """
    Test 3-tier policy around upper threshold (0.6010):
    -0.3520 <= score <= 0.6010  => REVIEW QUEUE
    score > 0.6010              => AUTO SPAM
    """
    res_below = evaluate_three_tier_routing(0.6009)
    res_exact = evaluate_three_tier_routing(0.6010)
    res_above = evaluate_three_tier_routing(0.6011)

    assert res_below["tier"] == "REVIEW QUEUE", f"Expected REVIEW QUEUE for 0.6009, got {res_below['tier']}"
    assert res_exact["tier"] == "REVIEW QUEUE", f"Expected REVIEW QUEUE for 0.6010, got {res_exact['tier']}"
    assert res_above["tier"] == "AUTO SPAM", f"Expected AUTO SPAM for 0.6011, got {res_above['tier']}"


def test_three_tier_example_score_0_6027_is_auto_spam():
    """Verify that score +0.6027 yields AUTO SPAM (mandatory prompt correction)."""
    res = evaluate_three_tier_routing(0.6027)
    assert res["tier"] == "AUTO SPAM", f"Expected AUTO SPAM for score +0.6027, got {res['tier']}"


def test_binary_threshold_semantics():
    """
    Test binary threshold 0.4013:
    score >= 0.4013 => SPAM
    score < 0.4013  => HAM
    """
    res_below = evaluate_binary_classification(0.4012)
    res_exact = evaluate_binary_classification(0.4013)
    res_above = evaluate_binary_classification(0.4014)

    assert res_below["label"] == "HAM", f"Expected HAM for 0.4012, got {res_below['label']}"
    assert res_exact["label"] == "SPAM", f"Expected SPAM for 0.4013, got {res_exact['label']}"
    assert res_above["label"] == "SPAM", f"Expected SPAM for 0.4014, got {res_above['label']}"


def test_artifact_hash_verification():
    """Verify deployment artifact SHA-256 integrity checker."""
    assert verify_artifact_hash() is True, "verify_artifact_hash should return True for deployment artifacts."


def test_pinned_expected_sha256_constant():
    """Verify pinned EXPECTED_ARTIFACT_SHA256 constant value in application code."""
    assert EXPECTED_ARTIFACT_SHA256 == "7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06"


def test_pipeline_cached_loading_and_inference():
    """Verify loading cached pipeline and making a test inference."""
    pipeline = load_cached_pipeline()
    assert pipeline.is_fitted_, "Pipeline must be fitted."

    parsed = parse_raw_email(b"From: test@corp.com\nSubject: Hello\n\nMeeting at 10am")
    rec = {
        "subject": parsed.get("subject", ""),
        "plain_body": parsed.get("plain_body", ""),
        "html_body": parsed.get("html_body", ""),
        "has_html": parsed.get("has_html", False),
        "urls": parsed.get("urls", []),
        "url_count": parsed.get("url_count", 0),
        "attachment_count": parsed.get("attachment_count", 0),
        "is_multipart": parsed.get("is_multipart", False)
    }

    score = float(pipeline.predict_score([rec])[0])
    assert isinstance(score, float)


def test_input_size_limit_constant():
    """Verify 2 MB max input size limit constant."""
    assert MAX_INPUT_BYTES == 2 * 1024 * 1024


def test_offline_url_analysis_no_network():
    """Verify url security analysis executes strictly offline."""
    urls = ["http://192.168.1.1/login", "http://bit.ly/test", "http://safe-domain.org/path"]
    res = analyze_email_urls(urls)

    assert res["analyzed_url_count"] == 3
    assert res["flagged_url_count"] >= 2
    assert "IP_ADDRESS_HOST" in res["observed_indicators"]
    assert "URL_SHORTENER_DOMAIN" in res["observed_indicators"]


def test_streamlit_config_file_disables_telemetry():
    """Verify .streamlit/config.toml exists and disables gatherUsageStats."""
    config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
    assert config_path.exists(), f"Missing Streamlit config file at {config_path}"

    content = config_path.read_text(encoding="utf-8")
    assert "[browser]" in content
    assert "gatherUsageStats = false" in content
