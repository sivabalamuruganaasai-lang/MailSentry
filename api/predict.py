"""
Vercel Serverless Function: Inference & URL Security Endpoint (/api/predict).

Accepts pasted email text or raw .eml/.txt file uploads up to 2.0 MiB, executes
the frozen E11 LinearSVC model and offline URL heuristic analysis, and returns
structured JSON with threat verdict, signed margins, routing dispositions, and explainability.

PRIVACY & SECURITY GUARANTEES:
- Strictly in-memory processing (zero temporary files written to disk).
- Zero logging of email bodies, subjects, headers, or extracted URLs.
- Client IP rate-limiting (30 requests/minute ceiling).
- Strict 2.0 MiB server-side input size ceiling enforced before body ingestion.
- Safe error handling (zero Python tracebacks or internal paths leaked to client).
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from typing import Optional

# Ensure project root is on sys.path for headless imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import (
    BINARY_THRESHOLD,
    LOWER_THREE_TIER,
    UPPER_THREE_TIER,
    MAX_INPUT_BYTES,
    EXPECTED_ARTIFACT_SHA256,
    verify_artifact_hash,
    load_deployment_pipeline,
    evaluate_binary_classification,
    evaluate_three_tier_routing,
    build_feature_record
)
from src.data.parser import parse_raw_email
from src.data.email_validator import validate_email_input
from src.features.url_security import analyze_email_urls
from src.models.explainability import explain_individual_prediction


# Pinned Demo Email Fixtures for /api/predict GET endpoint
DEMO_PRESETS = {
    "Legitimate Team Sync (HAM)": (
        "From: alex.smith@company.com\n"
        "To: team@company.com\n"
        "Subject: Project Launch Sync Tomorrow at 10am\n"
        "Date: Thu, 09 Oct 2026 10:00:00 +0000\n\n"
        "Hi Team,\n\n"
        "Please review the attached project roadmap for tomorrow's launch meeting.\n"
        "Let me know if you have any questions.\n\n"
        "Best regards,\nAlex"
    ),
    "Promotional HTML Newsletter (HTML FP Case)": (
        "From: news@techdigest.org\n"
        "To: subscriber@domain.com\n"
        "Subject: Weekly Tech Digest & Member Deals\n"
        "Content-Type: text/html\n\n"
        "<html><body>"
        "<h1>Weekly Tech Digest</h1>"
        "<p>Check out our latest articles and exclusive member offers!</p>"
        "<a href='http://techdigest.org/read'>Read Full Issue</a>"
        "</body></html>"
    ),
    "Phishing Scam with IP Host Link (SPAM)": (
        "From: security-alert@bank-update-service.net\n"
        "To: victim@user.com\n"
        "Subject: URGENT: Your Account Has Been Locked\n\n"
        "Dear Customer,\n\n"
        "We detected unauthorized access to your account. You must verify your credentials immediately.\n"
        "Click the link below to restore access:\n"
        "http://192.168.1.100/verify_login\n\n"
        "Failure to verify within 24 hours will result in permanent suspension."
    )
}

# Module-level warm-start cache for the deserialized pipeline
_CACHED_PIPELINE = None

# In-memory sliding-window rate limiter per client IP: max 30 requests per 60-second window
_RATE_LIMIT_STORE = defaultdict(list)
_RATE_LIMIT_MAX_REQUESTS = 30
_RATE_LIMIT_WINDOW_SECONDS = 60.0


def get_cached_pipeline():
    """Retrieve or lazily initialize the verified model pipeline in container memory."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        if not verify_artifact_hash():
            raise RuntimeError("Cryptographic integrity verification failed for deployment artifact.")
        _CACHED_PIPELINE = load_deployment_pipeline()
    return _CACHED_PIPELINE


def check_rate_limit(client_ip: str) -> bool:
    """Return True if request is within rate limit; False if rate limit exceeded."""
    if not client_ip:
        return True
    now = time.time()
    window_start = now - _RATE_LIMIT_WINDOW_SECONDS

    # Clean timestamps older than window
    timestamps = [t for t in _RATE_LIMIT_STORE[client_ip] if t > window_start]
    if len(timestamps) >= _RATE_LIMIT_MAX_REQUESTS:
        _RATE_LIMIT_STORE[client_ip] = timestamps
        return False

    timestamps.append(now)
    _RATE_LIMIT_STORE[client_ip] = timestamps
    return True


def extract_payload_bytes(content_type: str, body_bytes: bytes) -> bytes:
    """Extract raw email bytes from JSON, multipart/form-data, or raw text payloads."""
    if not body_bytes:
        return b""

    c_type_lower = content_type.lower() if content_type else ""

    # Branch 1: JSON payload (e.g. {"email_text": "..."})
    if "application/json" in c_type_lower:
        try:
            data = json.loads(body_bytes.decode("utf-8", errors="replace"))
            if isinstance(data, dict):
                text_val = (
                    data.get("email_text")
                    or data.get("text")
                    or data.get("raw_email")
                    or data.get("payload")
                    or ""
                )
                return str(text_val).encode("utf-8")
            elif isinstance(data, str):
                return data.encode("utf-8")
        except Exception:
            return body_bytes

    # Branch 2: Multipart form-data upload
    if "multipart/form-data" in c_type_lower and "boundary=" in c_type_lower:
        try:
            # Extract boundary token preserving original case
            idx = c_type_lower.find("boundary=")
            boundary_part = content_type[idx + len("boundary="):].strip().split(";")[0].strip(' \t\r\n\'"')
            boundary = boundary_part.encode()
            delimiter = b"--" + boundary

            parts = body_bytes.split(delimiter)
            for part in parts:
                if b'filename="' in part:
                    # Found file part, extract payload after header separator (\r\n\r\n or \n\n)
                    if b"\r\n\r\n" in part:
                        file_data = part.split(b"\r\n\r\n", 1)[1]
                    elif b"\n\n" in part:
                        file_data = part.split(b"\n\n", 1)[1]
                    else:
                        continue
                    # Strip trailing CRLF
                    return file_data.rstrip(b"\r\n-")
                elif b'name="email_text"' in part or b'name="text"' in part:
                    if b"\r\n\r\n" in part:
                        field_data = part.split(b"\r\n\r\n", 1)[1]
                    elif b"\n\n" in part:
                        field_data = part.split(b"\n\n", 1)[1]
                    else:
                        continue
                    return field_data.rstrip(b"\r\n-")
        except Exception:
            return body_bytes

    # Branch 3: Raw text/plain or octet-stream bytes
    return body_bytes


def extract_uploaded_filename(content_type: str, body_bytes: bytes) -> Optional[str]:
    """Extract uploaded filename from multipart/form-data payload if present."""
    if not body_bytes or not content_type:
        return None
    c_type_lower = content_type.lower()
    if "multipart/form-data" in c_type_lower and "boundary=" in c_type_lower:
        try:
            idx = c_type_lower.find("boundary=")
            boundary_part = content_type[idx + len("boundary="):].strip().split(";")[0].strip(' \t\r\n\'"')
            boundary = boundary_part.encode()
            delimiter = b"--" + boundary

            parts = body_bytes.split(delimiter)
            for part in parts:
                if b'filename="' in part:
                    fn_start = part.find(b'filename="') + len(b'filename="')
                    fn_end = part.find(b'"', fn_start)
                    if fn_end != -1:
                        return part[fn_start:fn_end].decode("utf-8", errors="replace").strip()
        except Exception:
            return None
    return None



class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Function HTTP request handler for /api/predict."""

    def _send_json(self, status_code: int, data: dict):
        """Helper to send standardized JSON response with security headers."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests safely."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self):
        """Return API documentation and verified benchmark presets."""
        self._send_json(200, {
            "endpoint": "/api/predict",
            "description": "MailSentry E11 Email Spam & Offline URL Security Analysis Engine",
            "supported_methods": ["POST"],
            "max_input_bytes": MAX_INPUT_BYTES,
            "binary_threshold": BINARY_THRESHOLD,
            "routing_boundaries": {
                "lower_auto_ham": LOWER_THREE_TIER,
                "upper_review_queue": UPPER_THREE_TIER
            },
            "presets": list(DEMO_PRESETS.keys()),
            "preset_payloads": DEMO_PRESETS
        })

    def do_POST(self):
        """Execute full headless threat analysis on submitted payload."""
        # Step 1: Identify client IP for rate limiting
        forwarded_for = self.headers.get("X-Forwarded-For", "")
        client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (self.client_address[0] if self.client_address else "unknown")

        if not check_rate_limit(client_ip):
            self._send_json(429, {
                "error": "Rate limit exceeded. Maximum 30 requests per minute allowed.",
                "code": "RATE_LIMIT_EXCEEDED"
            })
            return

        # Step 2: Enforce strict 2.0 MiB input size limit via Content-Length header
        content_length_header = self.headers.get("Content-Length")
        if content_length_header:
            try:
                content_len = int(content_length_header)
                if content_len > MAX_INPUT_BYTES:
                    self._send_json(413, {
                        "error": f"Payload size ({content_len} bytes) exceeds maximum limit of 2.0 MiB ({MAX_INPUT_BYTES} bytes).",
                        "code": "PAYLOAD_TOO_LARGE"
                    })
                    return
            except ValueError:
                self._send_json(400, {
                    "error": "Invalid Content-Length header format.",
                    "code": "INVALID_CONTENT_LENGTH"
                })
                return
        else:
            content_len = None

        # Step 3: Stream ingestion with size guard
        try:
            if content_len is not None:
                body_bytes = self.rfile.read(content_len)
            else:
                body_bytes = self.rfile.read(MAX_INPUT_BYTES + 1)
                if len(body_bytes) > MAX_INPUT_BYTES:
                    self._send_json(413, {
                        "error": f"Payload exceeds maximum limit of 2.0 MiB ({MAX_INPUT_BYTES} bytes).",
                        "code": "PAYLOAD_TOO_LARGE"
                    })
                    return
        except Exception:
            self._send_json(400, {
                "error": "Failed to read request body.",
                "code": "BODY_READ_ERROR"
            })
            return

        # Step 4: Extract payload bytes and metadata
        content_type = self.headers.get("Content-Type", "")
        uploaded_filename = extract_uploaded_filename(content_type, body_bytes)
        if uploaded_filename:
            file_ext = uploaded_filename.lower().rsplit(".", 1)[-1] if "." in uploaded_filename else ""
            if file_ext and file_ext not in ["eml", "txt"]:
                self._send_json(422, {
                    "error": f"Unsupported file extension (.{file_ext}). Only .eml and .txt files are accepted for email analysis.",
                    "code": "UNSUPPORTED_FILE_TYPE",
                    "message": f"Unsupported file extension (.{file_ext}). Please upload a valid .eml or .txt file."
                })
                return

        active_payload = extract_payload_bytes(content_type, body_bytes)

        # Step 5: Validate empty / whitespace payload
        if not active_payload or not active_payload.strip():
            self._send_json(400, {
                "error": "Email payload is empty or contains only whitespace.",
                "code": "EMPTY_INPUT",
                "message": "Email payload is empty or contains only whitespace."
            })
            return

        # Step 6: Parse RFC 822 email bytes
        try:
            parsed_email = parse_raw_email(active_payload)
            if not isinstance(parsed_email, dict) or parsed_email.get("parse_status") == "failed":
                err_msg = parsed_email.get("parse_error", "Structure could not be parsed.") if isinstance(parsed_email, dict) else "Parser failure."
                self._send_json(422, {
                    "error": f"Email structure could not be parsed: {err_msg}",
                    "code": "PARSE_FAILURE",
                    "message": "Email structure could not be parsed."
                })
                return
        except Exception:
            self._send_json(422, {
                "error": "Email parser encountered an unrecoverable format error.",
                "code": "PARSE_EXCEPTION",
                "message": "Email parser encountered an unrecoverable format error."
            })
            return

        # Step 7: Pre-Inference Email Input Validation
        # Validates email plausibility and rejects non-email / random text before model execution
        validation = validate_email_input(active_payload, parsed_email)
        if not validation["is_valid"]:
            self._send_json(422, {
                "error": validation["message"],
                "code": validation["error_code"],
                "message": validation["message"],
                "details": validation.get("details", {})
            })
            return

        # Step 8: Load verified model pipeline (only reached for valid email input)
        try:
            pipeline = get_cached_pipeline()

        except Exception:
            self._send_json(500, {
                "error": "Model deployment bundle failed cryptographic verification or loading.",
                "code": "INTEGRITY_FAILURE"
            })
            return

        # Step 8: Build feature record and execute inference
        try:
            record_for_model = build_feature_record(parsed_email)
            explanation = explain_individual_prediction(
                pipeline, record_for_model, top_k=10, threshold=BINARY_THRESHOLD
            )
            score = float(explanation["raw_decision_score"])
            binary_res = evaluate_binary_classification(score)
            three_tier_res = evaluate_three_tier_routing(score)
        except Exception:
            self._send_json(500, {
                "error": "Model feature transformation or inference execution failed.",
                "code": "INFERENCE_ERROR"
            })
            return

        # Step 9: Offline URL security heuristic evaluation
        try:
            url_sec_res = analyze_email_urls(parsed_email.get("urls", []))
        except Exception:
            url_sec_res = {
                "analyzed_url_count": 0,
                "flagged_url_count": 0,
                "heuristic_risk_level": "UNKNOWN",
                "observed_indicators": [],
                "indicator_descriptions": {},
                "url_details": [],
                "disclaimer": "URL analysis failed."
            }

        # Step 10: Assemble clean, structured JSON response
        delta_val = score - BINARY_THRESHOLD
        abs_threshold_dist = abs(delta_val)
        svm_hyperplane_dist = abs(score)
        unique_urls = len(set(u.get("url", "") for u in url_sec_res.get("url_details", [])))

        response_payload = {
            "success": True,
            "decision": {
                "raw_decision_score": score,
                "binary_threshold": BINARY_THRESHOLD,
                "signed_margin": delta_val,
                "signed_margin_formatted": f"{delta_val:+.4f}",
                "absolute_distance_to_threshold": abs_threshold_dist,
                "svm_hyperplane_distance": svm_hyperplane_dist,
                "label": binary_res["label"],
                "is_spam": binary_res["is_spam"],
                "routing_tier": three_tier_res["tier"],
                "routing_action": three_tier_res["action"],
                "routing_description": three_tier_res["description"],
                "routing_bounds": {
                    "auto_ham_ceiling": LOWER_THREE_TIER,
                    "review_queue_ceiling": UPPER_THREE_TIER
                }
            },
            "url_security": {
                "analyzed_count": url_sec_res.get("analyzed_url_count", 0),
                "unique_count": unique_urls,
                "flagged_count": url_sec_res.get("flagged_url_count", 0),
                "heuristic_risk_level": url_sec_res.get("heuristic_risk_level", "LOW"),
                "observed_indicators": url_sec_res.get("observed_indicators", []),
                "indicator_descriptions": url_sec_res.get("indicator_descriptions", {}),
                "url_details": url_sec_res.get("url_details", []),
                "disclaimer": (
                    "Offline heuristics only: Syntactic pattern analysis without live DNS lookups, "
                    "reputation API calls, or verifying destination safety. A benign status does not "
                    "guarantee destination safety."
                )
            },
            "explainability": {
                "top_spam_features": explanation.get("top_spam_features", []),
                "top_ham_features": explanation.get("top_ham_features", []),
                "family_breakdown": explanation.get("family_breakdown", {}),
                "intercept": explanation.get("intercept", 0.0),
                "is_reconstructed_exact": explanation.get("is_reconstructed_exact", True)
            },
            "telemetry": {
                "subject": parsed_email.get("subject", ""),
                "from": parsed_email.get("from", ""),
                "to": parsed_email.get("to", ""),
                "date": parsed_email.get("date", ""),
                "message_id": parsed_email.get("message_id", ""),
                "content_type": parsed_email.get("content_type", ""),
                "is_multipart": parsed_email.get("is_multipart", False),
                "encoding_info": parsed_email.get("encoding_info", ""),
                "attachment_count": parsed_email.get("attachment_count", 0),
                "raw_byte_length": len(active_payload),
                "pinned_artifact_sha256": EXPECTED_ARTIFACT_SHA256
            }
        }

        self._send_json(200, response_payload)

    def log_message(self, format, *args):
        """Suppress default stderr request logging to ensure email content is never logged."""
        pass
