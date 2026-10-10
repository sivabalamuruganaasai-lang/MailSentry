"""
Unit tests for email parser module (src/data/parser.py).
"""

import pytest
from src.data.parser import parse_raw_email, extract_urls, extract_sender_domain, safe_decode_header


def test_parse_plain_text_email():
    raw_email = (
        b"From: sender@example.com\n"
        b"To: recipient@example.com\n"
        b"Subject: Test Plain Text\n"
        b"Message-ID: <12345@example.com>\n"
        b"Content-Type: text/plain; charset=utf-8\n\n"
        b"Hello, this is a plain text email with a link http://example.com/test.\n"
    )
    parsed = parse_raw_email(raw_email)

    assert parsed["parse_status"] in ["success", "partial"]
    assert parsed["subject"] == "Test Plain Text"
    assert parsed["from"] == "sender@example.com"
    assert parsed["sender_domain"] == "example.com"
    assert parsed["has_plain"] is True
    assert "Hello, this is a plain text email" in parsed["plain_body"]
    assert parsed["url_count"] == 1
    assert "http://example.com/test" in parsed["urls"]


def test_parse_html_email():
    raw_email = (
        b"From: promo@offers.org\n"
        b"Subject: Exclusive Offer!\n"
        b"Content-Type: text/html; charset=utf-8\n\n"
        b"<html><body><h1>Special Deal</h1><a href='https://offers.org/claim'>Claim Now</a></body></html>"
    )
    parsed = parse_raw_email(raw_email)

    assert parsed["subject"] == "Exclusive Offer!"
    assert parsed["has_html"] is True
    assert "Claim Now" in parsed["html_body"]
    assert parsed["url_count"] == 1
    assert "https://offers.org/claim" in parsed["urls"]


def test_parse_multipart_email():
    raw_email = (
        b"From: Multi <multi@domain.com>\n"
        b"Subject: Multipart Test\n"
        b"MIME-Version: 1.0\n"
        b"Content-Type: multipart/alternative; boundary=\"BOUNDARY\"\n\n"
        b"--BOUNDARY\n"
        b"Content-Type: text/plain; charset=utf-8\n\n"
        b"Plain text part.\n\n"
        b"--BOUNDARY\n"
        b"Content-Type: text/html; charset=utf-8\n\n"
        b"<p>HTML part with link https://domain.com/link</p>\n\n"
        b"--BOUNDARY--"
    )
    parsed = parse_raw_email(raw_email)

    assert parsed["is_multipart"] is True
    assert parsed["has_plain"] is True
    assert parsed["has_html"] is True
    assert "Plain text part" in parsed["plain_body"]
    assert "HTML part" in parsed["html_body"]
    assert parsed["url_count"] == 1


def test_parse_malformed_email():
    # Corrupted / binary noise bytes
    raw_email = b"\xFF\xFE\xFD\xFC Malformed email data without proper headers \x00\x01\x02"
    parsed = parse_raw_email(raw_email)

    assert parsed["parse_status"] in ["success", "partial", "failed"]
    assert isinstance(parsed["subject"], str)
    assert isinstance(parsed["plain_body"], str)


def test_leakage_header_extraction():
    raw_email = (
        b"From: spammer@bad.com\n"
        b"Subject: Buy Now\n"
        b"X-Spam-Status: Yes, score=15.2 required=5.0\n"
        b"X-Spam-Flag: YES\n\n"
        b"Buy cheap items now!"
    )
    parsed = parse_raw_email(raw_email)

    assert "X-Spam-Status" in parsed["leakage_headers_present"]
    assert "Yes" in parsed["leakage_headers_present"]["X-Spam-Status"]
    assert "X-Spam-Flag" in parsed["leakage_headers_present"]


def test_url_extraction():
    text = "Visit http://test.com/a and https://secure.org/b?id=10 for details."
    urls = extract_urls(text)
    assert len(urls) == 2
    assert "http://test.com/a" in urls
    assert "https://secure.org/b?id=10" in urls


def test_extract_sender_domain():
    assert extract_sender_domain("John Doe <john@company.co.uk>") == "company.co.uk"
    assert extract_sender_domain("spammer@domain.net") == "domain.net"
    assert extract_sender_domain("No Email Address") == ""
