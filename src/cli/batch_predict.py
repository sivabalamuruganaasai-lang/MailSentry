"""
Batch Inference CLI Engine for Email Spam Classifier (Level 14).

Provides headless, high-throughput, memory-bounded batch email classification:
- Ingests raw .eml and .txt files from a directory, or structured rows from CSV.
- Enforces strict 2 MiB per-email limit BEFORE reading files into memory.
- Explicitly handles parser statuses: 'success', 'partial', and 'failed'.
- Strictly maintains 1-to-1 input-to-output record correspondence (no dropped records).
- Employs bounded chunk streaming (default chunk_size=250) for constant RAM usage.
- Emits structured CSV or JSONL outputs to file or stdout.
- Directs all logs, progress indicators, and diagnostics exclusively to stderr.
- Follows strict exit code contract:
    0: Success (or partial record errors under default non-strict mode)
    1: Fatal invocation, missing inputs, integrity failure, or unwriteable output
    2: Partial record errors when --strict is enabled
"""

import sys
import os
import csv
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import (
    MAX_INPUT_BYTES,
    BINARY_THRESHOLD,
    LOWER_THREE_TIER,
    UPPER_THREE_TIER,
    verify_artifact_hash,
    load_deployment_pipeline,
    evaluate_binary_classification,
    evaluate_three_tier_routing,
    build_feature_record
)
from src.data.parser import parse_raw_email, extract_urls
from src.features.url_security import analyze_email_urls

OUTPUT_FIELDS = [
    "input_id",
    "decision_score",
    "binary_label",
    "routing_tier",
    "url_risk_level",
    "flagged_url_count",
    "parse_status",
    "status",
    "error_message"
]


def log_stderr(message: str) -> None:
    """Write log messages exclusively to sys.stderr to preserve stdout stream cleanliness."""
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def scan_directory_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    """Scan directory for .eml and .txt files."""
    valid_extensions = {".eml", ".txt"}
    collected: List[Path] = []

    if recursive:
        for root, _, files in os.walk(input_dir):
            for f in sorted(files):
                p = Path(root) / f
                if p.suffix.lower() in valid_extensions:
                    collected.append(p)
    else:
        for entry in sorted(os.scandir(input_dir), key=lambda e: e.name):
            if entry.is_file():
                p = Path(entry.path)
                if p.suffix.lower() in valid_extensions:
                    collected.append(p)

    return collected


def process_single_file_record(file_path: Path, base_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Validate size and parse a single email file.
    Returns: (output_row_dict, feature_record_or_None)
    """
    input_id = str(file_path.relative_to(base_dir)) if base_dir else str(file_path.name)

    # 1. Enforce 2 MiB limit BEFORE reading file into memory
    try:
        file_size = os.path.getsize(file_path)
    except Exception as e:
        return {
            "input_id": input_id,
            "decision_score": "",
            "binary_label": "ERROR",
            "routing_tier": "ERROR",
            "url_risk_level": "N/A",
            "flagged_url_count": "",
            "parse_status": "failed",
            "status": "error",
            "error_message": f"Filesystem access error: {str(e)}"
        }, None

    if file_size > MAX_INPUT_BYTES:
        return {
            "input_id": input_id,
            "decision_score": "",
            "binary_label": "ERROR",
            "routing_tier": "ERROR",
            "url_risk_level": "N/A",
            "flagged_url_count": "",
            "parse_status": "skipped",
            "status": "error",
            "error_message": f"File size ({file_size / (1024 * 1024):.2f} MiB) exceeds 2 MiB limit"
        }, None

    if file_size == 0:
        return {
            "input_id": input_id,
            "decision_score": "",
            "binary_label": "ERROR",
            "routing_tier": "ERROR",
            "url_risk_level": "N/A",
            "flagged_url_count": "",
            "parse_status": "failed",
            "status": "error",
            "error_message": "Empty email file (0 bytes)"
        }, None

    # 2. Read bytes safely
    try:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        return {
            "input_id": input_id,
            "decision_score": "",
            "binary_label": "ERROR",
            "routing_tier": "ERROR",
            "url_risk_level": "N/A",
            "flagged_url_count": "",
            "parse_status": "failed",
            "status": "error",
            "error_message": f"File read error: {str(e)}"
        }, None

    # 3. Parse Email
    parsed = parse_raw_email(raw_bytes)
    p_status = parsed.get("parse_status", "failed")

    if p_status == "failed":
        return {
            "input_id": input_id,
            "decision_score": "",
            "binary_label": "ERROR",
            "routing_tier": "ERROR",
            "url_risk_level": "N/A",
            "flagged_url_count": "",
            "parse_status": "failed",
            "status": "error",
            "error_message": f"MIME parse failure: {parsed.get('parse_error', 'unspecified error')}"
        }, None

    # Prepare partial or success record
    feat_record = build_feature_record(parsed)
    feat_record["_input_id"] = input_id
    feat_record["_parse_status"] = p_status
    feat_record["_parse_error"] = parsed.get("parse_error", "")
    feat_record["_urls"] = parsed.get("urls", [])

    return {}, feat_record


def process_single_csv_row(row: Dict[str, Any], row_idx: int) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Validate and extract features from a structured CSV row.
    Returns: (output_row_dict, feature_record_or_None)
    """
    input_id = str(row.get("id") or row.get("email_id") or row.get("message_id") or f"row_{row_idx}")

    subject = str(row.get("subject", "") or "")
    body = str(row.get("body", "") or row.get("plain_body", "") or row.get("text", "") or "")
    html_body = str(row.get("html_body", "") or "")

    # Size check on relevant string content
    content_bytes_len = len(subject.encode("utf-8", errors="replace")) + \
                        len(body.encode("utf-8", errors="replace")) + \
                        len(html_body.encode("utf-8", errors="replace"))

    if content_bytes_len > MAX_INPUT_BYTES:
        return {
            "input_id": input_id,
            "decision_score": "",
            "binary_label": "ERROR",
            "routing_tier": "ERROR",
            "url_risk_level": "N/A",
            "flagged_url_count": "",
            "parse_status": "skipped",
            "status": "error",
            "error_message": f"Row content size ({content_bytes_len / (1024 * 1024):.2f} MiB) exceeds 2 MiB limit"
        }, None

    if not body.strip() and not html_body.strip() and not subject.strip():
        return {
            "input_id": input_id,
            "decision_score": "",
            "binary_label": "ERROR",
            "routing_tier": "ERROR",
            "url_risk_level": "N/A",
            "flagged_url_count": "",
            "parse_status": "failed",
            "status": "error",
            "error_message": "Empty row content: subject and body are both blank"
        }, None

    # Handle URLs
    raw_urls_field = row.get("urls", "")
    if raw_urls_field and isinstance(raw_urls_field, str):
        urls = [u.strip() for u in raw_urls_field.replace(";", ",").split(",") if u.strip()]
    elif isinstance(raw_urls_field, list):
        urls = raw_urls_field
    else:
        urls = extract_urls(f"{subject}\n{body}\n{html_body}")

    p_status = "success"
    p_error = ""
    if not body.strip() and not html_body.strip():
        p_status = "partial"
        p_error = "Missing email body (subject only)"

    parsed_dict = {
        "subject": subject,
        "plain_body": body,
        "html_body": html_body,
        "has_html": bool(html_body.strip()),
        "urls": urls,
        "url_count": len(urls),
        "attachment_count": int(row.get("attachment_count", 0) or 0),
        "is_multipart": bool(row.get("is_multipart", False))
    }

    feat_record = build_feature_record(parsed_dict)
    feat_record["_input_id"] = input_id
    feat_record["_parse_status"] = p_status
    feat_record["_parse_error"] = p_error
    feat_record["_urls"] = urls

    return {}, feat_record


def execute_batch_inference_chunk(
    pipeline: Any,
    chunk_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Score a chunk of validated records and evaluate decision policies.
    """
    if not chunk_records:
        return []

    # Extract model scores for the batch
    scores = pipeline.predict_score(chunk_records)

    results: List[Dict[str, Any]] = []
    for rec, score_val in zip(chunk_records, scores):
        s_float = float(score_val)
        b_res = evaluate_binary_classification(s_float)
        t_res = evaluate_three_tier_routing(s_float)
        u_res = analyze_email_urls(rec.get("_urls", []))

        p_status = rec.get("_parse_status", "success")
        is_partial = (p_status == "partial")

        results.append({
            "input_id": rec["_input_id"],
            "decision_score": f"{s_float:+.4f}",
            "binary_label": b_res["label"],
            "routing_tier": t_res["tier"],
            "url_risk_level": u_res["heuristic_risk_level"],
            "flagged_url_count": u_res["flagged_url_count"],
            "parse_status": p_status,
            "status": "partial" if is_partial else "success",
            "error_message": f"Notice: {rec['_parse_error']}" if is_partial and rec.get("_parse_error") else ""
        })

    return results


class BatchOutputWriter:
    """Streaming writer supporting CSV and JSONL destinations (file or stdout)."""

    def __init__(self, output_path: Optional[Path], output_format: str):
        self.output_path = output_path
        self.output_format = output_format.lower()
        self.handle = None
        self.csv_writer = None
        self.records_written = 0

    def open(self):
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(self.output_path, "w", newline="", encoding="utf-8")
        else:
            self.handle = sys.stdout

        if self.output_format == "csv":
            self.csv_writer = csv.DictWriter(self.handle, fieldnames=OUTPUT_FIELDS)
            self.csv_writer.writeheader()
            self.handle.flush()

    def write_rows(self, rows: List[Dict[str, Any]]):
        if not rows:
            return

        for r in rows:
            clean_row = {k: r.get(k, "") for k in OUTPUT_FIELDS}
            if self.output_format == "csv":
                self.csv_writer.writerow(clean_row)
            else:
                self.handle.write(json.dumps(clean_row) + "\n")
            self.records_written += 1

        self.handle.flush()

    def close(self):
        if self.handle and self.output_path:
            self.handle.close()


def run_batch_pipeline(
    input_dir: Optional[Path] = None,
    input_csv: Optional[Path] = None,
    output_file: Optional[Path] = None,
    output_format: str = "csv",
    chunk_size: int = 250,
    recursive: bool = False,
    quiet: bool = False
) -> Tuple[int, Dict[str, Any]]:
    """
    Execute streaming batch pipeline.
    Returns: (exit_code, summary_dict)
    """
    t_start = time.perf_counter()

    summary = {
        "total_inputs": 0,
        "success_count": 0,
        "partial_count": 0,
        "error_count": 0,
        "elapsed_sec": 0.0,
        "throughput_per_sec": 0.0
    }

    # Verify and load pipeline once
    if not quiet:
        log_stderr("Validating deployment artifact trust anchor...")

    try:
        pipeline = load_deployment_pipeline()
    except Exception as e:
        log_stderr(f"FATAL: Pipeline load failure: {str(e)}")
        return 1, summary

    writer = BatchOutputWriter(output_file, output_format)
    writer.open()

    pending_records_to_infer: List[Dict[str, Any]] = []

    def flush_inference_chunk():
        nonlocal pending_records_to_infer
        if pending_records_to_infer:
            scored_rows = execute_batch_inference_chunk(pipeline, pending_records_to_infer)
            writer.write_rows(scored_rows)
            for sr in scored_rows:
                if sr["status"] == "success":
                    summary["success_count"] += 1
                elif sr["status"] == "partial":
                    summary["partial_count"] += 1
            pending_records_to_infer = []

    try:
        if input_dir:
            file_paths = scan_directory_files(input_dir, recursive=recursive)
            summary["total_inputs"] = len(file_paths)

            if not quiet:
                log_stderr(f"Discovered {len(file_paths)} email files in {input_dir}. Streaming in chunks of {chunk_size}...")

            for idx, f_path in enumerate(file_paths, 1):
                err_row, feat_rec = process_single_file_record(f_path, base_dir=input_dir)
                if err_row:
                    flush_inference_chunk()  # flush pending before writing error to preserve ordering
                    writer.write_rows([err_row])
                    summary["error_count"] += 1
                elif feat_rec:
                    pending_records_to_infer.append(feat_rec)
                    if len(pending_records_to_infer) >= chunk_size:
                        flush_inference_chunk()

                if not quiet and idx % chunk_size == 0:
                    log_stderr(f"  Processed {idx}/{len(file_paths)} files...")

            flush_inference_chunk()

        elif input_csv:
            with open(input_csv, "r", encoding="utf-8", errors="replace") as f_csv:
                reader = csv.DictReader(f_csv)
                for idx, row in enumerate(reader, 1):
                    summary["total_inputs"] += 1
                    err_row, feat_rec = process_single_csv_row(row, idx)
                    if err_row:
                        flush_inference_chunk()
                        writer.write_rows([err_row])
                        summary["error_count"] += 1
                    elif feat_rec:
                        pending_records_to_infer.append(feat_rec)
                        if len(pending_records_to_infer) >= chunk_size:
                            flush_inference_chunk()

                    if not quiet and idx % chunk_size == 0:
                        log_stderr(f"  Processed {idx} CSV rows...")

            flush_inference_chunk()

    except Exception as e:
        log_stderr(f"FATAL: Batch processing exception: {str(e)}")
        writer.close()
        return 1, summary

    writer.close()

    elapsed = time.perf_counter() - t_start
    summary["elapsed_sec"] = round(elapsed, 4)
    summary["throughput_per_sec"] = round(summary["total_inputs"] / elapsed, 2) if elapsed > 0 else 0.0

    if not quiet:
        log_stderr("\n================== BATCH INFERENCE SUMMARY ==================")
        log_stderr(f"Total Inputs Scanned: {summary['total_inputs']}")
        log_stderr(f"Success Records:     {summary['success_count']}")
        log_stderr(f"Partial Warnings:    {summary['partial_count']}")
        log_stderr(f"Error Records:       {summary['error_count']}")
        log_stderr(f"Elapsed Time:        {summary['elapsed_sec']:.2f}s")
        log_stderr(f"Throughput:          {summary['throughput_per_sec']:.1f} records/sec")
        log_stderr("=============================================================")

    return 0, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Headless Batch Inference CLI for Email Spam Classifier (Level 14)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Path to directory containing .eml and .txt email files"
    )
    input_group.add_argument(
        "--input-csv",
        type=Path,
        help="Path to structured CSV file containing email records"
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Destination path for output predictions (defaults to stdout if omitted)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default=None,
        help="Output format: 'csv' or 'jsonl' (inferred from output-file extension if omitted)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Batch chunk size for bounded RAM usage (default: 250)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories in directory mode"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if one or more individual records encounter errors"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress and summary messages on stderr"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.input_dir and not args.input_dir.is_dir():
        log_stderr(f"FATAL: Specified input directory does not exist or is not a directory: {args.input_dir}")
        return 1

    if args.input_csv and not args.input_csv.is_file():
        log_stderr(f"FATAL: Specified input CSV does not exist or is not a file: {args.input_csv}")
        return 1

    # Incur format
    out_format = "csv"
    if args.format:
        out_format = args.format.lower()
    elif args.output_file and args.output_file.suffix.lower() == ".jsonl":
        out_format = "jsonl"

    exit_code, summary = run_batch_pipeline(
        input_dir=args.input_dir,
        input_csv=args.input_csv,
        output_file=args.output_file,
        output_format=out_format,
        chunk_size=args.chunk_size,
        recursive=args.recursive,
        quiet=args.quiet
    )

    if exit_code != 0:
        return exit_code

    if args.strict and summary["error_count"] > 0:
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
