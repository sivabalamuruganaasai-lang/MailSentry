"""
Canonical Headless Inference and Decision Policy Engine.

Contains canonical implementations of:
- Deployment artifact SHA-256 trust anchor verification
- Model pipeline deserialization and loading
- Binary classification decision evaluation (tau = 0.4013)
- Three-tier operational routing evaluation (bounds [-0.3520, 0.6010])
- Input byte limit constants (2 MiB limit)
- Standardized feature record formatting

This module has ZERO dependencies on Streamlit and serves as the single source of truth
for both CLI engines (batch_predict, healthcheck) and the Streamlit web dashboard.
"""

import sys
import hashlib
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List

# Configuration Paths and Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOYMENT_ARTIFACT = PROJECT_ROOT / "models" / "deployment_e11.joblib"
DEPLOYMENT_MANIFEST = PROJECT_ROOT / "models" / "deployment_manifest.json"

EXPECTED_ARTIFACT_SHA256 = "7752ac87665a2d98c033a77e39688cb854d486ae0224ce458c6020627644ea06"
MAX_INPUT_BYTES = 2 * 1024 * 1024  # 2 MiB strict limit
BINARY_THRESHOLD = 0.4013
LOWER_THREE_TIER = -0.3520
UPPER_THREE_TIER = 0.6010


def compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hexadecimal digest of a file in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact_hash(
    artifact_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None
) -> bool:
    """
    Verify deployment artifact SHA-256 against pinned code constant and manifest BEFORE loading.
    Must return True before any joblib.load() is invoked.
    """
    target_artifact = artifact_path or DEPLOYMENT_ARTIFACT
    target_manifest = manifest_path or DEPLOYMENT_MANIFEST

    if not target_artifact.exists():
        return False

    try:
        computed_hash = compute_file_sha256(target_artifact)

        # Primary trust anchor: Pinned code constant
        if computed_hash != EXPECTED_ARTIFACT_SHA256:
            return False

        # Secondary consistency check against manifest if present
        if target_manifest.exists():
            with open(target_manifest, "r", encoding="utf-8") as f:
                manifest_meta = json.load(f)
            manifest_hash = manifest_meta.get("artifact_sha256", "")
            if manifest_hash and manifest_hash != computed_hash:
                return False

        return True
    except Exception:
        return False


def load_deployment_pipeline(artifact_path: Optional[Path] = None) -> Any:
    """
    Load the verified fitted SpamModelPipeline deployment artifact.
    Guarantees SHA-256 pre-verification before deserialization.
    """
    target_artifact = artifact_path or DEPLOYMENT_ARTIFACT
    if not verify_artifact_hash(target_artifact):
        raise RuntimeError(
            f"Deployment artifact SHA-256 verification failed for {target_artifact}. "
            "Refusing to deserialize unverified joblib bundle."
        )
    return joblib.load(target_artifact)


def evaluate_binary_classification(decision_score: float) -> Dict[str, Any]:
    """
    Evaluate binary classification at frozen threshold tau = 0.4013:
      score >= 0.4013 => SPAM
      score <  0.4013 => HAM
    """
    is_spam = decision_score >= BINARY_THRESHOLD
    margin = abs(decision_score - BINARY_THRESHOLD)
    return {
        "label": "SPAM" if is_spam else "HAM",
        "is_spam": is_spam,
        "score": decision_score,
        "threshold": BINARY_THRESHOLD,
        "margin": margin,
        "explanation": f"Decision score {decision_score:+.4f} is {'>=' if is_spam else '<'} threshold {BINARY_THRESHOLD:.4f}."
    }


def evaluate_three_tier_routing(decision_score: float) -> Dict[str, Any]:
    """
    Evaluate three-tier classification policy:
      score < -0.3520               => AUTO HAM (Direct Delivery)
      -0.3520 <= score <= 0.6010    => REVIEW QUEUE (Human Moderation)
      score > 0.6010                => AUTO SPAM (Automatically Blocked)
    """
    if decision_score < LOWER_THREE_TIER:
        tier = "AUTO HAM"
        action = "Direct Delivery to Inbox"
        color = "green"
        desc = f"Score {decision_score:+.4f} < {LOWER_THREE_TIER:.4f}. High confidence legitimate email."
    elif decision_score <= UPPER_THREE_TIER:
        tier = "REVIEW QUEUE"
        action = "Routed to Human Moderation"
        color = "orange"
        desc = f"Score {decision_score:+.4f} is between [{LOWER_THREE_TIER:.4f}, {UPPER_THREE_TIER:.4f}]. Moderate confidence, human review recommended."
    else:
        tier = "AUTO SPAM"
        action = "Automatically Blocked"
        color = "red"
        desc = f"Score {decision_score:+.4f} > {UPPER_THREE_TIER:.4f}. High confidence spam message."

    return {
        "tier": tier,
        "action": action,
        "color": color,
        "description": desc,
        "lower_bound": LOWER_THREE_TIER,
        "upper_bound": UPPER_THREE_TIER
    }


def build_feature_record(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format parsed email dictionary or CSV record for feature extraction and model inference.
    Normalizes missing fields to empty strings/zeroes.
    """
    return {
        "subject": str(parsed_data.get("subject", "") or ""),
        "plain_body": str(parsed_data.get("plain_body", "") or parsed_data.get("body", "") or parsed_data.get("text", "") or ""),
        "html_body": str(parsed_data.get("html_body", "") or ""),
        "has_html": bool(parsed_data.get("has_html", False) or parsed_data.get("html_body", "")),
        "urls": list(parsed_data.get("urls", []) or []),
        "url_count": int(parsed_data.get("url_count", 0) or len(parsed_data.get("urls", []) or [])),
        "attachment_count": int(parsed_data.get("attachment_count", 0) or 0),
        "is_multipart": bool(parsed_data.get("is_multipart", False))
    }
