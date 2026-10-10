"""
Safe, Robust Email Parser for raw RFC 822/5322 emails.

Uses Python standard library `email` module to extract structural headers,
plain-text body, HTML body, attachments, URLs, and metadata.
Handles encoding issues, malformed headers, and missing parts gracefully.
"""

import re
import email
from email import policy
from email.header import decode_header
from typing import Dict, Any, List, Tuple
from urllib.parse import urlparse

# Regular expression to extract URLs cleanly
URL_REGEX = re.compile(
    r'https?://[^\s<>"\']+',
    re.IGNORECASE
)

# Suspicious leakage headers to audit
LEAKAGE_HEADERS = [
    "X-Spam-Status",
    "X-Spam-Level",
    "X-Spam-Flag",
    "X-Spam-Checker-Version",
    "X-Spam-Report",
    "X-Spam-Pyzor-Response",
    "X-Razor-id",
]


def safe_decode_header(header_value: str) -> str:
    """Safely decode email header strings with arbitrary encodings."""
    if not header_value:
        return ""
    try:
        decoded_parts = decode_header(header_value)
        header_str = ""
        for content, encoding in decoded_parts:
            if isinstance(content, bytes):
                enc = encoding or "utf-8"
                try:
                    header_str += content.decode(enc, errors="replace")
                except Exception:
                    header_str += content.decode("latin-1", errors="replace")
            else:
                header_str += str(content)
        return header_str.strip()
    except Exception:
        return str(header_value).strip()


def extract_sender_domain(from_header: str) -> str:
    """Extract domain from From header e.g., 'User <user@example.com>' -> 'example.com'."""
    if not from_header:
        return ""
    match = re.search(r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', from_header)
    if match:
        return match.group(1).lower()
    return ""


def extract_urls(text: str) -> List[str]:
    """Extract all HTTP/HTTPS URLs from text cleanly stripping trailing punctuation."""
    if not text:
        return []
    raw_urls = URL_REGEX.findall(text)
    cleaned_urls = []
    for u in raw_urls:
        cleaned = u.rstrip(".,;:!)]>'\"")
        if cleaned:
            cleaned_urls.append(cleaned)
    return cleaned_urls


def decode_payload_bytes(payload: bytes, charset: str) -> str:
    """Decode raw bytes with fallback encodings."""
    if not payload:
        return ""
    encodings_to_try = []
    if charset:
        encodings_to_try.append(charset)
    encodings_to_try.extend(["utf-8", "iso-8859-1", "windows-1252", "ascii"])

    for enc in encodings_to_try:
        try:
            return payload.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Fallback replace
    return payload.decode("latin-1", errors="replace")


def parse_raw_email(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Parse a raw email byte string into structured metadata and content.
    Returns a dictionary with parse status, headers, body, and security indicators.
    """
    result = {
        "parse_status": "success",
        "parse_error": "",
        "message_id": "",
        "from": "",
        "to": "",
        "reply_to": "",
        "date": "",
        "subject": "",
        "sender_domain": "",
        "content_type": "",
        "is_multipart": False,
        "plain_body": "",
        "html_body": "",
        "has_plain": False,
        "has_html": False,
        "url_count": 0,
        "urls": [],
        "attachment_count": 0,
        "encoding_info": "",
        "leakage_headers_present": {},
        "raw_byte_length": len(raw_bytes)
    }

    if not raw_bytes or not raw_bytes.strip():
        result["parse_status"] = "failed"
        result["parse_error"] = "Empty raw email bytes"
        return result

    try:
        msg = email.message_from_bytes(raw_bytes, policy=policy.compat32)
    except Exception as e:
        result["parse_status"] = "failed"
        result["parse_error"] = f"Email message_from_bytes failed: {str(e)}"
        return result

    # Extract Standard Headers
    try:
        result["message_id"] = safe_decode_header(msg.get("Message-ID", ""))
        result["from"] = safe_decode_header(msg.get("From", ""))
        result["to"] = safe_decode_header(msg.get("To", ""))
        result["reply_to"] = safe_decode_header(msg.get("Reply-To", ""))
        result["date"] = safe_decode_header(msg.get("Date", ""))
        result["subject"] = safe_decode_header(msg.get("Subject", ""))
        result["sender_domain"] = extract_sender_domain(result["from"])
        result["content_type"] = msg.get_content_type()
        result["is_multipart"] = msg.is_multipart()
    except Exception as e:
        result["parse_status"] = "partial"
        result["parse_error"] += f"Header extraction warning: {str(e)}; "

    # Check for X-Spam-* Leakage Headers
    for lk_hdr in LEAKAGE_HEADERS:
        val = msg.get(lk_hdr)
        if val is not None:
            result["leakage_headers_present"][lk_hdr] = safe_decode_header(val)

    # Extract Body Content & Attachments
    plain_parts = []
    html_parts = []
    encodings_found = set()

    try:
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in content_disposition:
                    result["attachment_count"] += 1
                    continue

                charset = part.get_content_charset()
                if charset:
                    encodings_found.add(charset)

                payload_bytes = part.get_payload(decode=True)
                if payload_bytes:
                    decoded_text = decode_payload_bytes(payload_bytes, charset)
                    if content_type == "text/plain":
                        plain_parts.append(decoded_text)
                    elif content_type == "text/html":
                        html_parts.append(decoded_text)
        else:
            charset = msg.get_content_charset()
            if charset:
                encodings_found.add(charset)
            payload_bytes = msg.get_payload(decode=True)
            if payload_bytes:
                decoded_text = decode_payload_bytes(payload_bytes, charset)
                content_type = msg.get_content_type()
                if content_type == "text/html":
                    html_parts.append(decoded_text)
                else:
                    plain_parts.append(decoded_text)

        result["plain_body"] = "\n".join(plain_parts).strip()
        result["html_body"] = "\n".join(html_parts).strip()
        result["has_plain"] = bool(result["plain_body"])
        result["has_html"] = bool(result["html_body"])
        result["encoding_info"] = ", ".join(sorted(list(encodings_found)))

        # Extract URLs
        combined_text = result["plain_body"] + "\n" + result["html_body"]
        extracted_urls = extract_urls(combined_text)
        result["urls"] = extracted_urls
        result["url_count"] = len(extracted_urls)

        if not result["has_plain"] and not result["has_html"]:
            if result["parse_status"] == "success":
                result["parse_status"] = "partial"
                result["parse_error"] += "Empty body extracted; "

    except Exception as e:
        result["parse_status"] = "partial" if result["parse_status"] == "success" else "failed"
        result["parse_error"] += f"Body parsing error: {str(e)}"

    return result
