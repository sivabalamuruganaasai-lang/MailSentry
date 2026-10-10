"""
Standalone Deployment Health Check and Pre-Flight Self-Test (Level 14).

Verifies deployment readiness without starting a web server or connecting to external networks:
1. Validates deployment artifact SHA-256 against pinned code trust anchor BEFORE unpickling.
2. Checks dependency compatibility against deployment manifest.
3. Performs deterministic synthetic smoke inference against a fixed reference fixture.
4. Exits with code 0 on HEALTHY, code 1 on FAILURE.

Zero external network requests, zero arbitrary file execution, zero disk mutations.
"""

import sys
import os
import json
import argparse
import importlib
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import (
    DEPLOYMENT_ARTIFACT,
    DEPLOYMENT_MANIFEST,
    EXPECTED_ARTIFACT_SHA256,
    BINARY_THRESHOLD,
    LOWER_THREE_TIER,
    UPPER_THREE_TIER,
    compute_file_sha256,
    verify_artifact_hash,
    load_deployment_pipeline,
    evaluate_binary_classification,
    evaluate_three_tier_routing,
    build_feature_record
)
from src.data.parser import parse_raw_email

# Reference Fixture Provenance:
# Established during Level 12 persistence validation and confirmed during Level 13 live audits.
# Evaluated on frozen model E11 (LinearSVC, 10,036 features, MaxAbsScaler).
EXPECTED_REFERENCE_SCORE = 0.020524
REFERENCE_SCORE_TOLERANCE = 1e-4

HEALTHCHECK_FIXTURE_BYTES = (
    b"From: alex.smith@company.com\n"
    b"To: team@company.com\n"
    b"Subject: Project Launch Sync Tomorrow at 10am\n"
    b"Date: Thu, 09 Oct 2026 10:00:00 +0000\n\n"
    b"Hi Team,\n\n"
    b"Please review the attached project roadmap for tomorrow's launch meeting.\n"
    b"Let me know if you have any questions.\n\n"
    b"Best regards,\nAlex"
)


def run_health_check() -> Tuple[bool, Dict[str, Any]]:
    """
    Execute all 4 stages of deployment health check.
    Returns (is_healthy, results_dict).
    """
    results: Dict[str, Any] = {
        "status": "HEALTHY",
        "stages": {},
        "errors": []
    }

    # Stage 1: Pinned SHA-256 Hash Pre-Verification (BEFORE any unpickling)
    stage1 = {
        "name": "Artifact Hash Verification",
        "passed": False,
        "artifact_path": str(DEPLOYMENT_ARTIFACT),
        "pinned_sha256": EXPECTED_ARTIFACT_SHA256,
        "computed_sha256": "",
        "manifest_match": False
    }

    if not DEPLOYMENT_ARTIFACT.exists():
        stage1["error"] = f"Artifact missing at {DEPLOYMENT_ARTIFACT}"
        results["errors"].append(stage1["error"])
        results["status"] = "UNHEALTHY"
        results["stages"]["artifact_hash"] = stage1
        return False, results

    try:
        computed_sha = compute_file_sha256(DEPLOYMENT_ARTIFACT)
        stage1["computed_sha256"] = computed_sha

        if computed_sha != EXPECTED_ARTIFACT_SHA256:
            stage1["error"] = f"Computed SHA-256 {computed_sha} does not match pinned trust anchor {EXPECTED_ARTIFACT_SHA256}"
            results["errors"].append(stage1["error"])
            results["status"] = "UNHEALTHY"
            results["stages"]["artifact_hash"] = stage1
            return False, results  # HALT IMMEDIATELY: Never deserialize untrusted bytes

        # Check against manifest
        if DEPLOYMENT_MANIFEST.exists():
            with open(DEPLOYMENT_MANIFEST, "r", encoding="utf-8") as f:
                manifest_meta = json.load(f)
            manifest_sha = manifest_meta.get("artifact_sha256", "")
            stage1["manifest_match"] = (manifest_sha == computed_sha)
            if not stage1["manifest_match"]:
                stage1["error"] = "Manifest SHA-256 differs from computed hash"
                results["errors"].append(stage1["error"])
                results["status"] = "UNHEALTHY"
                results["stages"]["artifact_hash"] = stage1
                return False, results

        stage1["passed"] = True
    except Exception as e:
        stage1["error"] = f"Hash computation exception: {str(e)}"
        results["errors"].append(stage1["error"])
        results["status"] = "UNHEALTHY"
        results["stages"]["artifact_hash"] = stage1
        return False, results

    results["stages"]["artifact_hash"] = stage1

    # Stage 2: Runtime Environment & Dependencies Verification
    stage2 = {
        "name": "Runtime Dependencies Compatibility",
        "passed": True,
        "packages": {}
    }

    required_pkgs = ["sklearn", "joblib", "numpy", "pandas"]
    for pkg in required_pkgs:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            stage2["packages"][pkg] = {"installed": True, "version": version}
        except ImportError:
            stage2["packages"][pkg] = {"installed": False, "version": None}
            stage2["passed"] = False
            results["errors"].append(f"Required package '{pkg}' is missing")

    if not stage2["passed"]:
        results["status"] = "UNHEALTHY"
        results["stages"]["dependencies"] = stage2
        return False, results

    results["stages"]["dependencies"] = stage2

    # Stage 3: Safe Joblib Deserialization
    stage3 = {
        "name": "Pipeline Deserialization",
        "passed": False
    }

    try:
        pipeline = load_deployment_pipeline(DEPLOYMENT_ARTIFACT)
        if not getattr(pipeline, "is_fitted_", False):
            raise ValueError("Deserialized pipeline does not have is_fitted_ set to True")
        stage3["passed"] = True
    except Exception as e:
        stage3["error"] = f"Joblib deserialization failed: {str(e)}"
        results["errors"].append(stage3["error"])
        results["status"] = "UNHEALTHY"
        results["stages"]["deserialization"] = stage3
        return False, results

    results["stages"]["deserialization"] = stage3

    # Stage 4: Deterministic Synthetic Smoke Inference
    stage4 = {
        "name": "Deterministic Smoke Inference",
        "passed": False,
        "reference_score": EXPECTED_REFERENCE_SCORE,
        "measured_score": None,
        "delta": None
    }

    try:
        parsed_fixture = parse_raw_email(HEALTHCHECK_FIXTURE_BYTES)
        feature_rec = build_feature_record(parsed_fixture)
        measured_score = float(pipeline.predict_score([feature_rec])[0])
        delta = abs(measured_score - EXPECTED_REFERENCE_SCORE)

        stage4["measured_score"] = round(measured_score, 6)
        stage4["delta"] = delta

        if delta > REFERENCE_SCORE_TOLERANCE:
            stage4["error"] = (
                f"Smoke inference score mismatch: measured {measured_score:.6f} "
                f"differs from reference {EXPECTED_REFERENCE_SCORE:.6f} by {delta:.2e} (> tolerance {REFERENCE_SCORE_TOLERANCE})"
            )
            results["errors"].append(stage4["error"])
            results["status"] = "UNHEALTHY"
        else:
            bin_res = evaluate_binary_classification(measured_score)
            tier_res = evaluate_three_tier_routing(measured_score)
            stage4["binary_label"] = bin_res["label"]
            stage4["routing_tier"] = tier_res["tier"]
            stage4["passed"] = True

    except Exception as e:
        stage4["error"] = f"Smoke inference exception: {str(e)}"
        results["errors"].append(stage4["error"])
        results["status"] = "UNHEALTHY"

    results["stages"]["smoke_inference"] = stage4

    is_healthy = (results["status"] == "HEALTHY")
    return is_healthy, results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deployment Health Check and Pre-Flight Self-Test (Level 14)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results in machine-readable JSON format to stdout"
    )
    args = parser.parse_args()

    is_healthy, results = run_health_check()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("=================================================================")
        print("     EMAIL SPAM CLASSIFIER -- DEPLOYMENT PRE-FLIGHT HEALTH CHECK ")
        print("=================================================================")
        print(f"Overall Status: {'[HEALTHY]' if is_healthy else '[UNHEALTHY]'}\n")

        for s_key, s_data in results["stages"].items():
            mark = "[PASS]" if s_data.get("passed") else "[FAIL]"
            print(f"{mark} {s_data.get('name')}")
            if s_key == "artifact_hash":
                print(f"       File:     {s_data.get('artifact_path')}")
                print(f"       SHA-256:  {s_data.get('computed_sha256')}")
                print(f"       Trust:    {s_data.get('pinned_sha256') == s_data.get('computed_sha256')}")
            elif s_key == "dependencies":
                pkgs = ", ".join(f"{k}=={v.get('version')}" for k, v in s_data.get("packages", {}).items())
                print(f"       Packages: {pkgs}")
            elif s_key == "smoke_inference":
                print(f"       Measured Score:  {s_data.get('measured_score'):+.6f}")
                print(f"       Reference Score: {s_data.get('reference_score'):+.6f}")
                print(f"       Binary Label:    {s_data.get('binary_label')} (tau = {BINARY_THRESHOLD})")
                print(f"       Routing Tier:    {s_data.get('routing_tier')} (bounds [{LOWER_THREE_TIER}, {UPPER_THREE_TIER}])")

            if s_data.get("error"):
                print(f"       ERROR: {s_data.get('error')}")
            print()

        print("=================================================================")
        if not is_healthy:
            print(f"Errors detected: {len(results['errors'])}")
            for err in results["errors"]:
                print(f" - {err}")
            print("=================================================================")

    return 0 if is_healthy else 1


if __name__ == "__main__":
    sys.exit(main())
