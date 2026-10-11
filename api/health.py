"""
Vercel Serverless Function: Health Check Endpoint (/api/health).

Provides pre-flight cryptographic verification and operational configuration metadata
without exposing server filesystem paths, internal stack traces, or environment secrets.
"""

import json
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler

# Ensure project root is on sys.path for headless imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import (
    EXPECTED_ARTIFACT_SHA256,
    BINARY_THRESHOLD,
    LOWER_THREE_TIER,
    UPPER_THREE_TIER,
    MAX_INPUT_BYTES,
    verify_artifact_hash
)


class handler(BaseHTTPRequestHandler):
    """Vercel Serverless Function HTTP request handler for /api/health."""

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests safely."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self):
        """Execute pre-flight cryptographic trust verification and return public metadata."""
        try:
            # Verify SHA-256 hash against pinned constant before confirming health
            is_verified = verify_artifact_hash()
            status_str = "healthy" if is_verified else "degraded"
            http_code = 200 if is_verified else 500

            response_data = {
                "status": status_str,
                "deployment_id": "E11_deploy_v1",
                "model_name": "LinearSVC",
                "sha256_verified": is_verified,
                "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
                "binary_threshold": BINARY_THRESHOLD,
                "routing_boundaries": {
                    "auto_ham_ceiling": LOWER_THREE_TIER,
                    "review_queue_ceiling": UPPER_THREE_TIER
                },
                "max_input_bytes": MAX_INPUT_BYTES,
                "offline_security_guarantee": "air_gapped_heuristics_only"
            }
        except Exception:
            http_code = 500
            response_data = {
                "status": "error",
                "error": "Health check diagnostic failed.",
                "code": "HEALTH_CHECK_FAILED"
            }

        body = json.dumps(response_data).encode("utf-8")
        self.send_response(http_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Suppress default stderr request logging to protect client privacy."""
        pass
