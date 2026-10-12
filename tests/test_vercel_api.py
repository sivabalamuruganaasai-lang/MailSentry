"""
Comprehensive Tests for Vercel Serverless API Endpoints.

Tests:
1. GET /api/health: Cryptographic trust verification, config metadata, zero local path leakage.
2. GET /api/predict: Preset demo emails and endpoint documentation.
3. POST /api/predict (JSON HAM): Clear HAM classification, signed margin, routing disposition.
4. POST /api/predict (JSON SPAM with IP URL): SPAM detection, IP_ADDRESS_HOST heuristic.
5. POST /api/predict (Multipart file upload): EML file upload parsing and feature evaluation.
6. POST /api/predict (Input validation): Rejection of empty/whitespace input (HTTP 400).
7. POST /api/predict (Oversized payload): Strict 2.0 MiB ceiling enforced (HTTP 413).
8. POST /api/predict (Rate limiting): Client IP sliding-window throttling (HTTP 429).
9. POST /api/predict (HAM with Flagged URLs): Preserves HAM decision while reporting flagged links.
10. Privacy & Safe Error Trapping: Zero traceback leakage on malformed input.
"""

import io
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from api.health import handler as HealthHandler
from api.predict import handler as PredictHandler, _RATE_LIMIT_STORE, DEMO_PRESETS
from src.models.inference import (
    EXPECTED_ARTIFACT_SHA256,
    BINARY_THRESHOLD,
    LOWER_THREE_TIER,
    UPPER_THREE_TIER,
    MAX_INPUT_BYTES
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MockServerlessRequest:
    """Simulates HTTP request/response buffers for BaseHTTPRequestHandler testing."""

    def __init__(self, method="GET", path="/", headers=None, body=b"", client_ip="127.0.0.1"):
        self.method = method
        self.path = path
        self.headers = headers or {}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.client_address = (client_ip, 12345)
        self.response_code = None
        self.response_headers = {}

    def send_response(self, code, message=None):
        self.response_code = code

    def send_header(self, keyword, value):
        self.response_headers[keyword.lower()] = value

    def end_headers(self):
        pass

    def get_response_json(self):
        self.wfile.seek(0)
        content = self.wfile.read()
        return json.loads(content.decode("utf-8"))


def test_health_endpoint_success():
    """Verify GET /api/health returns HTTP 200, verified SHA-256, and zero local filesystem paths."""
    req = MockServerlessRequest(method="GET", path="/api/health")
    handler = HealthHandler.__new__(HealthHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_GET()

    assert req.response_code == 200
    data = req.get_response_json()
    assert data["status"] == "healthy"
    assert data["sha256_verified"] is True
    assert data["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256
    assert data["binary_threshold"] == BINARY_THRESHOLD
    assert data["max_input_bytes"] == MAX_INPUT_BYTES

    # Crucial security check: Ensure NO local filesystem paths are exposed
    raw_response_str = json.dumps(data)
    assert "d:" not in raw_response_str.lower()
    assert "c:" not in raw_response_str.lower()
    assert "users" not in raw_response_str.lower()
    assert "antigravity" not in raw_response_str.lower()


def test_predict_endpoint_get_documentation_and_presets():
    """Verify GET /api/predict returns presets and endpoint documentation."""
    req = MockServerlessRequest(method="GET", path="/api/predict")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_GET()

    assert req.response_code == 200
    data = req.get_response_json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    assert "Legitimate Team Sync (HAM)" in data["presets"]


def test_predict_endpoint_post_json_ham():
    """Verify POST /api/predict with JSON HAM payload returns correct decision."""
    ham_text = DEMO_PRESETS["Legitimate Team Sync (HAM)"]
    payload_bytes = json.dumps({"email_text": ham_text}).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload_bytes)),
        "X-Forwarded-For": "10.0.0.1"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=payload_bytes, client_ip="10.0.0.1")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 200
    res = req.get_response_json()
    assert res["success"] is True

    dec = res["decision"]
    assert dec["is_spam"] is False
    assert dec["label"] == "HAM"
    assert dec["raw_decision_score"] < BINARY_THRESHOLD
    # Score is +0.020524, which falls between [-0.3520, 0.6010], routing to REVIEW QUEUE
    assert dec["routing_tier"] == "REVIEW QUEUE"
    assert dec["signed_margin"] < 0.0
    assert abs(dec["absolute_distance_to_threshold"] - abs(dec["signed_margin"])) < 1e-5


def test_predict_endpoint_post_json_auto_ham():
    """Verify POST /api/predict routes emails with score < -0.3520 to AUTO HAM."""
    auto_ham_text = (
        "From: digest@lockergnome.com\n"
        "To: subscriber@domain.com\n"
        "Subject: Lockergnome newsletter update\n\n"
        "Here is your daily Lockergnome tech newsletter with software updates and tips.\n"
    )
    payload_bytes = json.dumps({"email_text": auto_ham_text}).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload_bytes)),
        "X-Forwarded-For": "10.0.0.99"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=payload_bytes, client_ip="10.0.0.99")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 200
    res = req.get_response_json()
    assert res["success"] is True

    dec = res["decision"]
    assert dec["is_spam"] is False
    assert dec["label"] == "HAM"
    assert dec["raw_decision_score"] < LOWER_THREE_TIER
    assert dec["routing_tier"] == "AUTO HAM"



def test_predict_endpoint_post_json_spam_with_ip_url():
    """Verify POST /api/predict with phishing email detects SPAM and flags IP URL."""
    spam_text = DEMO_PRESETS["Phishing Scam with IP Host Link (SPAM)"]
    payload_bytes = json.dumps({"email_text": spam_text}).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload_bytes)),
        "X-Forwarded-For": "10.0.0.2"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=payload_bytes, client_ip="10.0.0.2")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 200
    res = req.get_response_json()

    dec = res["decision"]
    assert dec["is_spam"] is True
    assert dec["label"] == "SPAM"
    assert dec["raw_decision_score"] >= BINARY_THRESHOLD
    assert dec["routing_tier"] == "AUTO SPAM"
    assert dec["signed_margin"] > 0.0

    url_sec = res["url_security"]
    assert url_sec["analyzed_count"] >= 1
    assert url_sec["flagged_count"] >= 1
    assert "IP_ADDRESS_HOST" in url_sec["observed_indicators"]


def test_predict_endpoint_post_multipart_file_upload():
    """Verify POST /api/predict handles multipart/form-data file uploads cleanly."""
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    raw_email = (
        b"From: partner@trusted.com\r\n"
        b"Subject: Quarterly Contract\r\n"
        b"\r\n"
        b"Please find our signed quarterly agreement."
    )

    multipart_body = (
        f"--{boundary}\r\n".encode() +
        b'Content-Disposition: form-data; name="email_file"; filename="contract.eml"\r\n' +
        b"Content-Type: message/rfc822\r\n\r\n" +
        raw_email +
        f"\r\n--{boundary}--\r\n".encode()
    )

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(multipart_body)),
        "X-Forwarded-For": "10.0.0.3"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=multipart_body, client_ip="10.0.0.3")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 200
    res = req.get_response_json()
    assert res["success"] is True
    assert res["telemetry"]["subject"] == "Quarterly Contract"


def test_predict_endpoint_empty_input_rejected():
    """Verify empty or whitespace-only input returns HTTP 400."""
    payload_bytes = json.dumps({"email_text": "   \n\t  "}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload_bytes)),
        "X-Forwarded-For": "10.0.0.4"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=payload_bytes, client_ip="10.0.0.4")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 400
    res = req.get_response_json()
    assert res["code"] == "EMPTY_INPUT"


def test_predict_endpoint_oversized_payload_rejected():
    """Verify payloads exceeding 2.0 MiB are rejected with HTTP 413."""
    oversized_len = MAX_INPUT_BYTES + 1024
    headers = {
        "Content-Type": "text/plain",
        "Content-Length": str(oversized_len),
        "X-Forwarded-For": "10.0.0.5"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=b"", client_ip="10.0.0.5")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 413
    res = req.get_response_json()
    assert res["code"] == "PAYLOAD_TOO_LARGE"


def test_predict_endpoint_rate_limiting():
    """Verify IP-based sliding window rate limiter triggers HTTP 429 after 30 requests."""
    test_ip = "198.51.100.99"
    _RATE_LIMIT_STORE.pop(test_ip, None)

    payload_bytes = json.dumps({"email_text": "From: test@corp.com\nSubject: Hi\n\nMeeting"}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload_bytes)),
        "X-Forwarded-For": test_ip
    }

    last_code = None
    # Fire 31 requests
    for i in range(32):
        req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=payload_bytes, client_ip=test_ip)
        handler = PredictHandler.__new__(PredictHandler)
        handler.headers = req.headers
        handler.rfile = req.rfile
        handler.wfile = req.wfile
        handler.client_address = req.client_address
        handler.send_response = req.send_response
        handler.send_header = req.send_header
        handler.end_headers = req.end_headers

        handler.do_POST()
        last_code = req.response_code

    assert last_code == 429
    res = req.get_response_json()
    assert res["code"] == "RATE_LIMIT_EXCEEDED"


def test_predict_endpoint_ham_with_flagged_urls():
    """Verify benchmark 00118 email (163 URLs, HAM score, 5 flagged URLs) returns accurate telemetry."""
    eml_path = PROJECT_ROOT / "data" / "raw" / "spamassassin" / "hard_ham" / "00118.afcf98c4f10a53e86d033ddf95660b5d"
    raw_bytes = eml_path.read_bytes()

    headers = {
        "Content-Type": "message/rfc822",
        "Content-Length": str(len(raw_bytes)),
        "X-Forwarded-For": "10.0.0.6"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=raw_bytes, client_ip="10.0.0.6")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 200
    res = req.get_response_json()

    # Classification remains HAM
    assert res["decision"]["is_spam"] is False
    assert res["decision"]["label"] == "HAM"

    # URL metrics correctly report 163 occurrences and 91 unique URLs
    urls = res["url_security"]
    assert urls["analyzed_count"] == 163
    assert urls["unique_count"] == 91
    assert urls["flagged_count"] == 5
    assert urls["heuristic_risk_level"] == "MEDIUM"


def test_predict_endpoint_zero_traceback_on_malformed_input():
    """Verify malformed input does not leak stack traces or system filepaths."""
    malformed_bytes = b"\x00\xff\xfe\x00\x01\x02\x03\x04"
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(malformed_bytes)),
        "X-Forwarded-For": "10.0.0.7"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=malformed_bytes, client_ip="10.0.0.7")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    raw_output = req.wfile.getvalue().decode("utf-8", errors="replace")
    assert "Traceback" not in raw_output
    assert "File \"" not in raw_output
    assert "d:\\email-spam-classifier" not in raw_output.lower()


def test_vercel_json_configuration_schema():
    """Verify vercel.json contains required deployment keys, functions config, and security headers."""
    vj_path = PROJECT_ROOT / "vercel.json"
    assert vj_path.exists(), "vercel.json must exist in project root"

    with open(vj_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 1. Output directory must be set to 'public' to prevent exposing internal repo files
    assert config.get("outputDirectory") == "public"

    # 2. Functions must include models/** and exclude data/reports/tests
    funcs = config.get("functions", {})
    assert "api/*.py" in funcs
    assert "models/**" in funcs["api/*.py"]["includeFiles"]
    assert "data/**" in funcs["api/*.py"]["excludeFiles"]

    # 3. Security headers must be present
    headers = config.get("headers", [])
    assert len(headers) >= 1
    header_keys = [h["key"] for h_block in headers for h in h_block.get("headers", [])]
    assert "X-Content-Type-Options" in header_keys
    assert "X-Frame-Options" in header_keys
    assert "Referrer-Policy" in header_keys


def test_vercelignore_does_not_exclude_src_packages():
    """Verify .vercelignore anchors root directories and does not exclude src/data/parser.py."""
    vi_path = PROJECT_ROOT / ".vercelignore"
    assert vi_path.exists(), ".vercelignore must exist in project root"

    with open(vi_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Critical check: ensure 'data/' is not unanchored (which would match any '**/data/' including 'src/data/')
    assert "data/" not in raw_lines, "Unanchored 'data/' in .vercelignore unintentionally excludes 'src/data/'"
    assert any(line in ["/data/", "/data"] for line in raw_lines), ".vercelignore must anchor root /data/ directory"
    assert any(line in ["/reports/", "/reports"] for line in raw_lines), ".vercelignore must anchor root /reports/ directory"


def test_predict_endpoint_missing_content_length_oversized():
    """Verify oversized request (> 2.0 MiB) is rejected with HTTP 413 when Content-Length header is omitted."""
    oversized_body = b"Subject: NoHeaderBig\n\n" + b"A" * (MAX_INPUT_BYTES + 2048)
    headers = {
        "Content-Type": "text/plain",
        "X-Forwarded-For": "10.0.0.10"
        # Content-Length explicitly omitted
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=oversized_body, client_ip="10.0.0.10")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 413
    res = req.get_response_json()
    assert res["code"] == "PAYLOAD_TOO_LARGE"


def test_predict_endpoint_inaccurate_invalid_content_length():
    """Verify non-integer Content-Length header returns HTTP 400."""
    payload_bytes = b"Hello world"
    headers = {
        "Content-Type": "text/plain",
        "Content-Length": "not-a-number",
        "X-Forwarded-For": "10.0.0.11"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=payload_bytes, client_ip="10.0.0.11")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 400
    res = req.get_response_json()
    assert res["code"] == "INVALID_CONTENT_LENGTH"


def test_predict_endpoint_multipart_with_quoted_mixed_case_boundary():
    """Verify multipart parser extracts payload when boundary contains quotes and mixed-case tokens."""
    boundary = "----WebKitFormBoundaryAbC123XyZ"
    raw_email = (
        b"From: secure@bank.com\r\n"
        b"Subject: Case Sensitive Boundary Test\r\n\r\n"
        b"Content body inside quoted boundary."
    )

    multipart_body = (
        f"--{boundary}\r\n".encode() +
        b'Content-Disposition: form-data; name="email_file"; filename="test.eml"\r\n' +
        b"Content-Type: message/rfc822\r\n\r\n" +
        raw_email +
        f"\r\n--{boundary}--\r\n".encode()
    )

    # Header with quotes around boundary
    headers = {
        "Content-Type": f'multipart/form-data; boundary="{boundary}"',
        "Content-Length": str(len(multipart_body)),
        "X-Forwarded-For": "10.0.0.12"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=multipart_body, client_ip="10.0.0.12")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 200
    res = req.get_response_json()
    assert res["success"] is True
    assert res["telemetry"]["subject"] == "Case Sensitive Boundary Test"


def test_predict_endpoint_binary_unsupported_file_graceful():
    """Verify uploading a binary executable file is gracefully rejected with HTTP 422 without model execution."""
    # Fake Windows PE executable header
    fake_exe_bytes = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00\xb8\x00\x00\x00"
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(fake_exe_bytes)),
        "X-Forwarded-For": "10.0.0.13"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=fake_exe_bytes, client_ip="10.0.0.13")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 422
    res = req.get_response_json()
    assert res["code"] == "INVALID_EMAIL_INPUT"
    assert "decision" not in res


def test_predict_endpoint_invalid_generic_paragraph_rejected():
    """Verify arbitrary generic prose (e.g. encyclopedia paragraph) returns HTTP 422 without invoking model."""
    generic_text = (
        "Photosynthesis is a process used by plants and other organisms to convert light "
        "energy into chemical energy that, through cellular respiration, can later be released "
        "to fuel the organism's metabolic activities. This chemical energy is stored in "
        "carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water."
    )
    payload_bytes = json.dumps({"email_text": generic_text}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload_bytes)),
        "X-Forwarded-For": "10.0.0.20"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=payload_bytes, client_ip="10.0.0.20")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 422
    res = req.get_response_json()
    assert res["code"] == "INVALID_EMAIL_INPUT"
    assert "decision" not in res
    assert "is_spam" not in res


def test_predict_endpoint_random_text_hello_world_rejected():
    """Verify greeting-only / random text like 'hello world' returns HTTP 422."""
    payload_bytes = json.dumps({"email_text": "hello world"}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload_bytes)),
        "X-Forwarded-For": "10.0.0.21"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=payload_bytes, client_ip="10.0.0.21")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 422
    res = req.get_response_json()
    assert res["code"] == "INSUFFICIENT_CONTENT"
    assert "decision" not in res


def test_predict_endpoint_multipart_unsupported_extension_rejected():
    """Verify uploading an unsupported file format (.pdf) returns HTTP 422 UNSUPPORTED_FILE_TYPE."""
    boundary = "----WebKitFormBoundaryPdfUpload123"
    pdf_content = b"%PDF-1.4 header fake content"
    multipart_body = (
        f"--{boundary}\r\n".encode() +
        b'Content-Disposition: form-data; name="email_file"; filename="invoice.pdf"\r\n' +
        b"Content-Type: application/pdf\r\n\r\n" +
        pdf_content +
        f"\r\n--{boundary}--\r\n".encode()
    )

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(multipart_body)),
        "X-Forwarded-For": "10.0.0.22"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=multipart_body, client_ip="10.0.0.22")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 422
    res = req.get_response_json()
    assert res["code"] == "UNSUPPORTED_FILE_TYPE"
    assert ".pdf" in res["error"]
    assert "decision" not in res


def test_predict_endpoint_multipart_txt_file_accepted():
    """Verify uploading a valid .txt email file is processed and classified by the model."""
    boundary = "----WebKitFormBoundaryTxtUpload456"
    txt_content = (
        b"From: billing@cloudprovider.com\r\n"
        b"To: client@domain.com\r\n"
        b"Subject: Monthly Cloud Invoice #4092\r\n\r\n"
        b"Please review your monthly cloud services invoice attached.\r\n"
        b"Best regards,\r\nBilling Team"
    )
    multipart_body = (
        f"--{boundary}\r\n".encode() +
        b'Content-Disposition: form-data; name="email_file"; filename="invoice_notice.txt"\r\n' +
        b"Content-Type: text/plain\r\n\r\n" +
        txt_content +
        f"\r\n--{boundary}--\r\n".encode()
    )

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(multipart_body)),
        "X-Forwarded-For": "10.0.0.23"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=multipart_body, client_ip="10.0.0.23")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 200
    res = req.get_response_json()
    assert res["success"] is True
    assert "decision" in res
    assert res["decision"]["label"] in ["HAM", "SPAM"]


def test_predict_endpoint_multipart_empty_file_rejected():
    """Verify uploading an empty file via multipart returns HTTP 400 EMPTY_INPUT."""
    boundary = "----WebKitFormBoundaryEmpty789"
    multipart_body = (
        f"--{boundary}\r\n".encode() +
        b'Content-Disposition: form-data; name="email_file"; filename="empty.eml"\r\n' +
        b"Content-Type: message/rfc822\r\n\r\n" +
        b"" +
        f"\r\n--{boundary}--\r\n".encode()
    )

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(multipart_body)),
        "X-Forwarded-For": "10.0.0.24"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=multipart_body, client_ip="10.0.0.24")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 400
    res = req.get_response_json()
    assert res["code"] == "EMPTY_INPUT"
    assert "decision" not in res


def test_client_xss_injection_payload_sanitization():
    """Verify email payloads with embedded script/HTML tags return safely without executing."""
    xss_payload = (
        "From: attacker@evil.com\n"
        "Subject: <script>alert('XSS')</script>\n"
        "Content-Type: text/html\n\n"
        "<html><body>"
        "<p>Hello <img src=x onerror=alert('img_xss')></p>"
        "<a href='http://evil.com/<script>alert(1)</script>'>Click here</a>"
        "</body></html>"
    )
    payload_bytes = json.dumps({"email_text": xss_payload}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload_bytes)),
        "X-Forwarded-For": "10.0.0.14"
    }

    req = MockServerlessRequest(method="POST", path="/api/predict", headers=headers, body=payload_bytes, client_ip="10.0.0.14")
    handler = PredictHandler.__new__(PredictHandler)
    handler.headers = req.headers
    handler.rfile = req.rfile
    handler.wfile = req.wfile
    handler.client_address = req.client_address
    handler.send_response = req.send_response
    handler.send_header = req.send_header
    handler.end_headers = req.end_headers

    handler.do_POST()

    assert req.response_code == 200
    res = req.get_response_json()
    assert res["success"] is True
    # The JSON response contains the parsed subject as raw string data; it must not be rendered into HTML unescaped
    assert res["telemetry"]["subject"] == "<script>alert('XSS')</script>"
