"""
Unit and Integration Tests for Level 14 CLI Engine and Deployment Health Check.

Validates:
1. Health-check standalone execution (healthy exit 0, tampered exit 1, JSON schema).
2. Deterministic reference score reproduction against frozen E11 model.
3. Batch directory mode with .eml and .txt files.
4. Structured CSV mode with full schema and minimal schema.
5. 2 MiB input size limit enforcement before in-memory reads.
6. 1-to-1 record preservation (corrupt MIME, oversized files, partial parses).
7. Exit code contracts (0 for clean/non-strict, 1 for fatal errors, 2 for --strict partials).
8. JSONL and CSV output formats.
9. Stdout stream purity (diagnostics on stderr only).
10. Chunked streaming accuracy.
"""

import sys
import os
import io
import csv
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import (
    EXPECTED_ARTIFACT_SHA256,
    BINARY_THRESHOLD,
    LOWER_THREE_TIER,
    UPPER_THREE_TIER,
    MAX_INPUT_BYTES
)
from src.cli.healthcheck import run_health_check, EXPECTED_REFERENCE_SCORE
from src.cli.batch_predict import run_batch_pipeline, OUTPUT_FIELDS


@pytest.fixture
def temp_batch_dir():
    """Create a temporary directory with sample .eml and .txt files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir)

        # 1. Clean HAM email
        ham_eml = dir_path / "01_ham.eml"
        ham_eml.write_bytes(
            b"From: colleague@company.com\n"
            b"To: team@company.com\n"
            b"Subject: Team Sync Tomorrow\n\n"
            b"Please review the slide deck before tomorrow's meeting."
        )

        # 2. Phishing SPAM email with IP host
        spam_txt = dir_path / "02_spam.txt"
        spam_txt.write_bytes(
            b"From: alert@security-bank.com\n"
            b"Subject: URGENT: Verify your account\n\n"
            b"Click here immediately: http://192.168.1.1/login"
        )

        # 3. Empty file (0 bytes)
        empty_eml = dir_path / "03_empty.eml"
        empty_eml.write_bytes(b"")

        # 4. Non-matching extension (should be ignored)
        ignored_pdf = dir_path / "04_notes.pdf"
        ignored_pdf.write_bytes(b"PDF dummy content")

        yield dir_path


@pytest.fixture
def temp_oversized_file(temp_batch_dir):
    """Add a file exceeding 2 MiB limit to temporary directory."""
    oversized = temp_batch_dir / "05_oversized.eml"
    with open(oversized, "wb") as f:
        f.write(b"Subject: Big\n\n" + b"X" * (MAX_INPUT_BYTES + 1024))
    return oversized


# =========================================================================
# 1. HEALTH-CHECK ENGINE TESTS
# =========================================================================

def test_healthcheck_valid_deployment():
    """Verify health-check succeeds with code 0 on unmodified deployment."""
    is_healthy, results = run_health_check()
    assert is_healthy is True
    assert results["status"] == "HEALTHY"
    assert results["stages"]["artifact_hash"]["passed"] is True
    assert results["stages"]["dependencies"]["passed"] is True
    assert results["stages"]["deserialization"]["passed"] is True
    assert results["stages"]["smoke_inference"]["passed"] is True
    assert results["stages"]["smoke_inference"]["reference_score"] == EXPECTED_REFERENCE_SCORE
    assert abs(results["stages"]["smoke_inference"]["measured_score"] - EXPECTED_REFERENCE_SCORE) < 1e-4


def test_healthcheck_tampered_hash_fails_before_load():
    """Verify that tampering with pinned hash constant aborts before deserialization."""
    with patch("src.cli.healthcheck.EXPECTED_ARTIFACT_SHA256", "0000000000000000000000000000000000000000000000000000000000000000"):
        is_healthy, results = run_health_check()
        assert is_healthy is False
        assert results["status"] == "UNHEALTHY"
        assert results["stages"]["artifact_hash"]["passed"] is False
        # Assert Stage 3 (deserialization) was NEVER executed
        assert "deserialization" not in results["stages"]


# =========================================================================
# 2. BATCH PREDICT: DIRECTORY INPUT MODE TESTS
# =========================================================================

def test_batch_directory_mode_csv_output(temp_batch_dir):
    """Verify processing directory containing .eml, .txt, and empty file."""
    output_csv = temp_batch_dir / "out.csv"

    exit_code, summary = run_batch_pipeline(
        input_dir=temp_batch_dir,
        output_file=output_csv,
        output_format="csv",
        quiet=True
    )

    assert exit_code == 0
    assert summary["total_inputs"] == 3  # 01_ham, 02_spam, 03_empty (04_notes.pdf ignored)
    assert summary["success_count"] == 2
    assert summary["error_count"] == 1  # 03_empty is recorded as error

    assert output_csv.exists()
    with open(output_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 3
    # Check 01_ham record
    r_ham = next(r for r in reader if "01_ham.eml" in r["input_id"])
    assert r_ham["status"] == "success"
    assert r_ham["binary_label"] in ("HAM", "SPAM")
    assert float(r_ham["decision_score"]) < BINARY_THRESHOLD

    # Check 02_spam record
    r_spam = next(r for r in reader if "02_spam.txt" in r["input_id"])
    assert r_spam["status"] == "success"
    assert r_spam["url_risk_level"] == "HIGH"
    assert int(r_spam["flagged_url_count"]) >= 1

    # Check 03_empty record
    r_empty = next(r for r in reader if "03_empty.eml" in r["input_id"])
    assert r_empty["status"] == "error"
    assert "Empty email file" in r_empty["error_message"]
    assert r_empty["binary_label"] == "ERROR"


def test_batch_oversized_file_enforced_before_read(temp_batch_dir, temp_oversized_file):
    """Verify that oversized file (>2 MiB) is marked as error without silently dropping it."""
    output_csv = temp_batch_dir / "out_oversized.csv"

    exit_code, summary = run_batch_pipeline(
        input_dir=temp_batch_dir,
        output_file=output_csv,
        output_format="csv",
        quiet=True
    )

    assert exit_code == 0
    assert summary["total_inputs"] == 4
    assert summary["error_count"] == 2  # 03_empty and 05_oversized

    with open(output_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 4
    r_over = next(r for r in reader if "05_oversized.eml" in r["input_id"])
    assert r_over["status"] == "error"
    assert "exceeds 2 MiB limit" in r_over["error_message"]
    assert r_over["parse_status"] == "skipped"


# =========================================================================
# 3. BATCH PREDICT: STRUCTURED CSV INPUT MODE TESTS
# =========================================================================

def test_batch_csv_mode_with_full_and_minimal_rows(tmp_path):
    """Verify CSV ingestion with full schema, minimal schema, and duplicate IDs."""
    input_csv = tmp_path / "input.csv"
    output_jsonl = tmp_path / "out.jsonl"

    rows = [
        {"id": "msg_1", "subject": "Quarterly Report", "body": "Attached please find the quarterly report.", "urls": ""},
        {"id": "msg_2", "subject": "Exclusive Discount", "body": "Claim 50% discount at http://bit.ly/deal", "urls": "http://bit.ly/deal"},
        {"id": "msg_2", "subject": "Duplicate ID Row", "body": "Same id as row above to test duplicate ID preservation.", "urls": ""},
        {"id": "msg_3", "subject": "", "body": "Minimal row without subject line", "urls": ""},
        {"id": "msg_4", "subject": "", "body": "", "urls": ""}  # Empty row
    ]

    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "subject", "body", "urls"])
        writer.writeheader()
        writer.writerows(rows)

    exit_code, summary = run_batch_pipeline(
        input_csv=input_csv,
        output_file=output_jsonl,
        output_format="jsonl",
        quiet=True
    )

    assert exit_code == 0
    assert summary["total_inputs"] == 5
    assert summary["success_count"] == 4
    assert summary["error_count"] == 1  # msg_4 is blank

    with open(output_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # 1-to-1 input to output mapping check
    assert len(records) == 5

    # Duplicate IDs check: both msg_2 rows must be preserved in output
    msg_2_records = [r for r in records if r["input_id"] == "msg_2"]
    assert len(msg_2_records) == 2

    # Blank row check
    msg_4 = next(r for r in records if r["input_id"] == "msg_4")
    assert msg_4["status"] == "error"
    assert "Empty row content" in msg_4["error_message"]


# =========================================================================
# 4. CHUNKING, STRICT EXIT CODES & STDOUT TESTS
# =========================================================================

def test_chunked_streaming_consistency(tmp_path):
    """Verify small chunk size produces identical results in correct order."""
    input_csv = tmp_path / "chunk_test.csv"
    output_csv = tmp_path / "chunk_out.csv"

    rows = [{"id": f"mail_{i}", "subject": f"Subj {i}", "body": f"Hello body {i}"} for i in range(12)]
    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "subject", "body"])
        writer.writeheader()
        writer.writerows(rows)

    exit_code, summary = run_batch_pipeline(
        input_csv=input_csv,
        output_file=output_csv,
        output_format="csv",
        chunk_size=3,  # 12 records in chunks of 3
        quiet=True
    )

    assert exit_code == 0
    assert summary["total_inputs"] == 12
    assert summary["success_count"] == 12

    with open(output_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 12
    for idx, r in enumerate(reader):
        assert r["input_id"] == f"mail_{idx}"


def test_empty_input_directory(tmp_path):
    """Verify empty input directory outputs valid empty CSV structure cleanly."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    out_csv = tmp_path / "empty_out.csv"

    exit_code, summary = run_batch_pipeline(
        input_dir=empty_dir,
        output_file=out_csv,
        output_format="csv",
        quiet=True
    )

    assert exit_code == 0
    assert summary["total_inputs"] == 0

    with open(out_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 0  # Header present, 0 rows


def test_strict_exit_code_on_partial_failure(temp_batch_dir):
    """Verify that --strict flag causes CLI main() to exit with code 2 on record errors."""
    from src.cli.batch_predict import main

    out_file = temp_batch_dir / "strict_out.csv"
    test_args = [
        "batch_predict.py",
        "--input-dir", str(temp_batch_dir),
        "--output-file", str(out_file),
        "--strict",
        "--quiet"
    ]

    with patch.object(sys, "argv", test_args):
        # 03_empty.eml in temp_batch_dir will produce an error record, triggering code 2 under --strict
        exit_code = main()
        assert exit_code == 2


def test_stdout_stream_purity(tmp_path, monkeypatch):
    """Verify that when writing to stdout, stdout receives pure CSV without log contamination."""
    test_dir = tmp_path / "stdout_dir"
    test_dir.mkdir()
    (test_dir / "mail.txt").write_bytes(b"From: test@corp.com\nSubject: Hi\n\nTesting body")

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_stdout)
    monkeypatch.setattr(sys, "stderr", captured_stderr)

    exit_code, summary = run_batch_pipeline(
        input_dir=test_dir,
        output_file=None,  # stdout
        output_format="csv",
        quiet=False
    )

    assert exit_code == 0
    stdout_content = captured_stdout.getvalue()
    stderr_content = captured_stderr.getvalue()

    # stdout must be valid CSV
    reader = list(csv.DictReader(io.StringIO(stdout_content)))
    assert len(reader) == 1
    assert reader[0]["input_id"] == "mail.txt"
    assert reader[0]["status"] == "success"

    # stderr must contain progress / summary logs
    assert "BATCH INFERENCE SUMMARY" in stderr_content
    assert "Discovered 1 email files" in stderr_content


def test_partial_parse_status_preserved(tmp_path):
    """Verify that emails with subject only are marked with parse_status='partial' and status='partial'."""
    test_csv = tmp_path / "partial.csv"
    out_csv = tmp_path / "partial_out.csv"

    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "subject", "body"])
        writer.writeheader()
        writer.writerow({"id": "p_1", "subject": "Subject Only Meeting", "body": ""})

    exit_code, summary = run_batch_pipeline(
        input_csv=test_csv,
        output_file=out_csv,
        output_format="csv",
        quiet=True
    )

    assert exit_code == 0
    assert summary["partial_count"] == 1

    with open(out_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 1
    assert reader[0]["status"] == "partial"
    assert reader[0]["parse_status"] == "partial"
    assert "Missing email body" in reader[0]["error_message"]
    # Model score must still be calculated
    assert reader[0]["decision_score"] != ""
    assert reader[0]["binary_label"] in ("HAM", "SPAM")
