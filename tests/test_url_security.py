"""
Unit and Integration Tests for Offline URL and Email Security Analysis (Level 9).

Tests:
1. Benign vs. suspicious URL structural indicator detection.
2. IP address hosts, user-info @ symbols, non-standard ports, shorteners, Punycode IDN.
3. Excessive subdomains, double slashes, encoded control characters, executable extensions.
4. Malformed, empty, None, and extreme-length (10,000+ chars) URL inputs.
5. Strict network isolation assurance (verifies ZERO network sockets or HTTP calls).
"""

import pytest
import socket
from unittest.mock import patch

from src.features.url_security import (
    analyze_single_url,
    analyze_email_urls,
    get_indicator_definitions,
    INDICATOR_CATALOG
)


def test_indicator_catalog_definitions():
    """Verify indicator catalog contains all required definitions and plain-language descriptions."""
    catalog = get_indicator_definitions()
    assert isinstance(catalog, dict)
    assert len(catalog) >= 10

    for key, info in catalog.items():
        assert "severity" in info
        assert "name" in info
        assert "description" in info
        assert info["severity"] in ["HIGH", "MEDIUM", "LOW"]


def test_benign_urls():
    """Test benign URLs yield no high risk indicators and low risk level."""
    benign_list = [
        "https://www.google.com",
        "https://en.wikipedia.org/wiki/Main_Page",
        "http://python.org/downloads"
    ]

    for url in benign_list:
        res = analyze_single_url(url)
        assert res["is_valid"] is True
        assert res["indicators"] == []
        assert res["flagged"] is False

    summary = analyze_email_urls(benign_list)
    assert summary["analyzed_url_count"] == 3
    assert summary["flagged_url_count"] == 0
    assert summary["heuristic_risk_level"] == "LOW"
    assert summary["observed_indicators"] == []


def test_ip_address_host_detection():
    """Test detection of raw IP address hostnames (IPv4 and IPv6)."""
    res1 = analyze_single_url("http://192.168.1.1/login.php")
    assert "IP_ADDRESS_HOST" in res1["indicators"]

    res2 = analyze_single_url("http://[2001:db8::1]:8080/index.html")
    assert "IP_ADDRESS_HOST" in res2["indicators"]
    assert "NON_STANDARD_PORT" in res2["indicators"]


def test_userinfo_in_authority_detection():
    """Test detection of user-info @ symbol in authority component."""
    res = analyze_single_url("http://paypal.com@evil-phishing-site.com/account")
    assert "USERINFO_IN_AUTHORITY" in res["indicators"]
    assert res["flagged"] is True


def test_non_standard_port_detection():
    """Test detection of non-standard ports (outside 80/443)."""
    res_8080 = analyze_single_url("http://example.com:8080/api")
    assert "NON_STANDARD_PORT" in res_8080["indicators"]

    res_80 = analyze_single_url("http://example.com:80/index.html")
    assert "NON_STANDARD_PORT" not in res_80["indicators"]

    res_443 = analyze_single_url("https://example.com:443/secure")
    assert "NON_STANDARD_PORT" not in res_443["indicators"]


def test_url_shortener_detection():
    """Test detection of known URL shorteners."""
    shortener_urls = [
        "http://bit.ly/3xYz123",
        "https://tinyurl.com/sample",
        "http://t.co/abcXYZ"
    ]

    for u in shortener_urls:
        res = analyze_single_url(u)
        assert "URL_SHORTENER_DOMAIN" in res["indicators"]


def test_punycode_idn_domain_detection():
    """Test detection of Punycode / Internationalized Domain Names."""
    res = analyze_single_url("http://xn--p1ai.ru/free-offer")
    assert "PUNYCODE_IDN_DOMAIN" in res["indicators"]


def test_excessive_subdomains_and_high_risk_tld():
    """Test detection of excessive subdomains and high-risk TLDs."""
    res = analyze_single_url("http://a.b.c.d.example.top/claim")
    assert "EXCESSIVE_SUBDOMAINS" in res["indicators"]
    assert "SUSPICIOUS_HIGH_RISK_TLD" in res["indicators"]


def test_double_slash_and_executable_extensions():
    """Test detection of double slashes in path and executable file downloads."""
    res = analyze_single_url("http://example.com//downloads/setup.exe")
    assert "DOUBLE_SLASH_IN_PATH" in res["indicators"]
    assert "EXECUTABLE_FILE_EXTENSION" in res["indicators"]


def test_percent_encoded_control_characters():
    """Test detection of percent-encoded null bytes and control chars."""
    res = analyze_single_url("http://example.com/page.php%00.jpg")
    assert "PERCENT_ENCODED_CONTROL_CHARS" in res["indicators"]


def test_empty_and_malformed_inputs():
    """Test behavior on empty, None, and malformed URL inputs."""
    res_none = analyze_single_url(None)
    assert res_none["is_valid"] is False
    assert "MALFORMED_URL_SYNTAX" in res_none["indicators"]

    res_empty = analyze_single_url("")
    assert res_empty["is_valid"] is False

    summary_empty = analyze_email_urls([])
    assert summary_empty["analyzed_url_count"] == 0
    assert summary_empty["heuristic_risk_level"] == "LOW"


def test_extreme_length_url():
    """Test extreme length URL (10,000+ characters) without crashing."""
    long_url = "http://example.com/path?" + "a=" * 10000
    res = analyze_single_url(long_url)
    assert "EXTREME_URL_LENGTH" in res["indicators"]
    assert res["is_valid"] is True


def test_strict_network_isolation_assurance(monkeypatch):
    """
    STRICT SECURITY TEST: Verify that offline analysis makes ZERO network calls.
    Monkeypatch socket creation to immediately raise an error if any network operation occurs.
    """
    def forbidden_socket(*args, **kwargs):
        raise RuntimeError("CRITICAL SECURITY VIOLATION: Network socket access attempted during offline analysis!")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    monkeypatch.setattr(socket, "gethostbyname", forbidden_socket)

    test_urls = [
        "http://192.168.1.1/login.php",
        "http://bit.ly/3xYz123",
        "http://paypal.com@evil.com/",
        "https://www.google.com"
    ]

    # Should execute purely offline without invoking forbidden_socket
    summary = analyze_email_urls(test_urls)

    assert summary["analyzed_url_count"] == 4
    assert summary["flagged_url_count"] == 3
    assert summary["heuristic_risk_level"] == "HIGH"
    assert "IP_ADDRESS_HOST" in summary["observed_indicators"]
    assert "USERINFO_IN_AUTHORITY" in summary["observed_indicators"]
    assert "URL_SHORTENER_DOMAIN" in summary["observed_indicators"]
