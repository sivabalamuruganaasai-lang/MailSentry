"""
Offline URL and Email Security Analysis Module (Level 9).

Provides safe, offline static security analysis of URLs found within emails.
Detects heuristic structural risk indicators without performing ANY network requests,
DNS lookups, or external API calls.

STRICT SECURITY SAFEGUARDS:
- NO network requests (strictly offline string/syntax parsing).
- Treat all URLs and email content as untrusted input.
- Does NOT alter the frozen E11 ML model, features, or 0.4013 decision threshold.
- Clearly distinguishes heuristic risk indicators from confirmed malicious intent.
"""

import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

# Known URL shorteners & redirection services
KNOWN_SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly",
    "goo.gl", "bit.do", "mcaf.ee", "su.pr", "tiny.cc", "rebrand.ly", "cutt.ly",
    "shorturl.at", "v.gd", "qr.ae"
}

# High-risk TLDs historically associated with phishing/malware abuse telemetry
HIGH_RISK_TLDS = {
    "zip", "mov", "top", "xyz", "work", "click", "link", "gq", "cf", "ml", "tk",
    "country", "kim", "science", "party", "review"
}

# High-risk file extensions in URL paths
EXECUTABLE_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".vbs", ".ps1", ".cmd", ".pif", ".jar", ".apk",
    ".iso", ".img", ".dll", ".hta", ".js", ".wsf"
}

# Regex for IPv4 and bracketed IPv6 hostnames
IP_HOST_REGEX = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$|^\s*\[[0-9a-fA-F:]+\]\s*$')

# Indicator Catalog with Plain-Language Descriptions
INDICATOR_CATALOG: Dict[str, Dict[str, str]] = {
    "IP_ADDRESS_HOST": {
        "severity": "HIGH",
        "name": "Raw IP Address Host",
        "description": "URL uses a raw IPv4 or IPv6 address instead of a domain name, bypassing hostname resolution."
    },
    "USERINFO_IN_AUTHORITY": {
        "severity": "HIGH",
        "name": "User-Info @ Symbol in Authority",
        "description": "URL contains a user-info '@' symbol in authority component, which can deceive users into mistaking the prefix for the true destination domain."
    },
    "PERCENT_ENCODED_CONTROL_CHARS": {
        "severity": "HIGH",
        "name": "Encoded Null Byte or Control Character",
        "description": "URL contains percent-encoded null bytes (%00) or control characters commonly used in null-byte injection or filter evasion."
    },
    "EXECUTABLE_FILE_EXTENSION": {
        "severity": "HIGH",
        "name": "Executable File Download Path",
        "description": "URL path targets an executable binary, script installer, or disk image extension."
    },
    "MALFORMED_URL_SYNTAX": {
        "severity": "MEDIUM",
        "name": "Malformed URL Syntax",
        "description": "URL fails standard RFC 3986 parsing or contains invalid structural syntax."
    },
    "URL_SHORTENER_DOMAIN": {
        "severity": "MEDIUM",
        "name": "URL Shortener Service",
        "description": "URL uses a domain shortener service that obfuscates the ultimate target destination."
    },
    "NON_STANDARD_PORT": {
        "severity": "MEDIUM",
        "name": "Non-Standard Port Specification",
        "description": "URL specifies an explicit network port outside standard HTTP (80) and HTTPS (443)."
    },
    "PUNYCODE_IDN_DOMAIN": {
        "severity": "MEDIUM",
        "name": "Punycode / IDN Domain Encoding",
        "description": "Hostname uses Internationalized Domain Name (IDN) Punycode encoding ('xn--'), which may indicate homograph domain spoofing."
    },
    "EXCESSIVE_SUBDOMAINS": {
        "severity": "MEDIUM",
        "name": "Excessive Subdomain Depth",
        "description": "Hostname contains an unusually deep subdomain hierarchy (4 or more domain label levels)."
    },
    "DOUBLE_SLASH_IN_PATH": {
        "severity": "LOW",
        "name": "Double Slash in Path",
        "description": "URL path contains a double slash ('//'), often seen in open redirect schemes or path confusion."
    },
    "EXTREME_URL_LENGTH": {
        "severity": "LOW",
        "name": "Unusually Long URL / Hostname",
        "description": "URL length exceeds 2,000 characters or hostname exceeds 253 characters."
    },
    "SUSPICIOUS_HIGH_RISK_TLD": {
        "severity": "LOW",
        "name": "High-Abuse Top-Level Domain",
        "description": "Domain uses a top-level domain frequently associated with abuse in public security telemetry."
    }
}


def get_indicator_definitions() -> Dict[str, Dict[str, str]]:
    """Return dictionary of all defined security indicators and their plain-language descriptions."""
    return INDICATOR_CATALOG.copy()


def analyze_single_url(url: str) -> Dict[str, Any]:
    """
    Perform safe, offline static security analysis on a single URL.

    Args:
        url: Raw URL string to analyze.

    Returns:
        Dict containing URL string, validity status, detected indicators,
        severity breakdown, and parsed components.
    """
    url_str = str(url).strip() if url is not None else ""
    indicators: List[str] = []

    if not url_str:
        return {
            "url": url_str,
            "is_valid": False,
            "indicators": ["MALFORMED_URL_SYNTAX"],
            "flagged": True,
            "hostname": "",
            "scheme": ""
        }

    # 1. Check Extreme Length (>2000 chars)
    if len(url_str) > 2000:
        indicators.append("EXTREME_URL_LENGTH")

    # 2. Check Encoded Null Bytes / Control Chars (%00, %0a, %0d)
    lower_url = url_str.lower()
    if "%00" in lower_url or "%0a" in lower_url or "%0d" in lower_url:
        indicators.append("PERCENT_ENCODED_CONTROL_CHARS")

    # 3. Parse URL safely
    try:
        parsed = urlparse(url_str)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc
        path = parsed.path
    except Exception:
        indicators.append("MALFORMED_URL_SYNTAX")
        return {
            "url": url_str,
            "is_valid": False,
            "indicators": indicators,
            "flagged": True,
            "hostname": "",
            "scheme": ""
        }

    # If scheme missing, attempt fallback parsing
    if not netloc and not scheme and "/" in url_str:
        try:
            parsed_fallback = urlparse("http://" + url_str)
            netloc = parsed_fallback.netloc
            path = parsed_fallback.path
        except Exception:
            pass

    if not netloc:
        indicators.append("MALFORMED_URL_SYNTAX")

    # Extract Hostname and Port
    hostname = ""
    port_str = ""

    if netloc:
        # Check User-Info @ in Authority
        if "@" in netloc:
            indicators.append("USERINFO_IN_AUTHORITY")
            # Host is after '@'
            authority_host = netloc.split("@")[-1]
        else:
            authority_host = netloc

        # Separate Port if present
        if ":" in authority_host and not authority_host.startswith("["):
            parts = authority_host.split(":")
            hostname = parts[0].lower()
            port_str = parts[1] if len(parts) > 1 else ""
        elif authority_host.startswith("[") and "]" in authority_host:
            # Bracketed IPv6 address
            closing_bracket = authority_host.index("]")
            hostname = authority_host[:closing_bracket + 1].lower()
            if ":" in authority_host[closing_bracket + 1:]:
                port_str = authority_host[closing_bracket + 1:].replace(":", "")
        else:
            hostname = authority_host.lower()

    # 4. Check Non-Standard Port
    if port_str:
        if port_str.isdigit() and port_str not in ["80", "443"]:
            indicators.append("NON_STANDARD_PORT")

    # 5. Check IP Address Host
    if hostname and IP_HOST_REGEX.match(hostname):
        indicators.append("IP_ADDRESS_HOST")

    # 6. Check Punycode / IDN Domain
    if hostname and ("xn--" in hostname or any(ord(c) > 127 for c in hostname)):
        indicators.append("PUNYCODE_IDN_DOMAIN")

    # 7. Check URL Shortener Domain
    if hostname:
        for shortener in KNOWN_SHORTENER_DOMAINS:
            if hostname == shortener or hostname.endswith("." + shortener):
                indicators.append("URL_SHORTENER_DOMAIN")
                break

    # 8. Check Excessive Subdomains (4+ labels)
    if hostname and not IP_HOST_REGEX.match(hostname):
        labels = [l for l in hostname.split(".") if l]
        if len(labels) >= 4:
            indicators.append("EXCESSIVE_SUBDOMAINS")

        # 9. Check High-Abuse TLD
        if len(labels) >= 2:
            tld = labels[-1]
            if tld in HIGH_RISK_TLDS:
                indicators.append("SUSPICIOUS_HIGH_RISK_TLD")

    # 10. Check Double Slash in Path
    if path and "//" in path:
        indicators.append("DOUBLE_SLASH_IN_PATH")

    # 11. Check Executable File Extension
    if path:
        lower_path = path.lower()
        for ext in EXECUTABLE_EXTENSIONS:
            if lower_path.endswith(ext):
                indicators.append("EXECUTABLE_FILE_EXTENSION")
                break

    # Deduplicate indicators preserving order
    unique_indicators = list(dict.fromkeys(indicators))

    return {
        "url": url_str,
        "is_valid": "MALFORMED_URL_SYNTAX" not in unique_indicators,
        "indicators": unique_indicators,
        "flagged": len(unique_indicators) > 0,
        "hostname": hostname,
        "scheme": scheme
    }


def analyze_email_urls(urls: List[str]) -> Dict[str, Any]:
    """
    Analyze all URLs contained within an email and return an interpretable security summary.

    STRICT GUARANTEE: Performs ZERO network requests or external queries.

    Args:
        urls: List of URL strings extracted from an email.

    Returns:
        Dict containing URL counts, overall heuristic risk score/level, per-URL details,
        observed indicators, plain-language descriptions, and explicit safety disclaimers.
    """
    if not isinstance(urls, list) or not urls:
        return {
            "analyzed_url_count": 0,
            "flagged_url_count": 0,
            "heuristic_risk_level": "LOW",
            "observed_indicators": [],
            "indicator_descriptions": {},
            "url_details": [],
            "risk_distinction": (
                "Heuristic structural indicators quantify observed pattern risks and static "
                "syntactic anomalies. They do NOT constitute proof of malicious intent."
            ),
            "disclaimer": (
                "This offline static analysis assesses heuristic structural indicators only "
                "and cannot establish whether a URL or domain is currently active, safe, or malicious."
            )
        }

    url_details = [analyze_single_url(u) for u in urls]
    flagged_urls = [u for u in url_details if u["flagged"]]

    # Collect all unique observed indicators
    all_observed: List[str] = []
    for d in url_details:
        all_observed.extend(d["indicators"])
    unique_observed = list(dict.fromkeys(all_observed))

    # Build description dictionary for observed indicators
    descriptions = {
        ind: INDICATOR_CATALOG[ind]["description"]
        for ind in unique_observed
        if ind in INDICATOR_CATALOG
    }

    # Determine overall heuristic risk level
    severities = [
        INDICATOR_CATALOG[ind]["severity"]
        for ind in unique_observed
        if ind in INDICATOR_CATALOG
    ]

    if "HIGH" in severities or len(unique_observed) >= 3:
        risk_level = "HIGH"
    elif "MEDIUM" in severities or len(unique_observed) >= 1:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "analyzed_url_count": len(urls),
        "flagged_url_count": len(flagged_urls),
        "heuristic_risk_level": risk_level,
        "observed_indicators": unique_observed,
        "indicator_descriptions": descriptions,
        "url_details": url_details,
        "risk_distinction": (
            "Heuristic structural indicators quantify observed pattern risks and static "
            "syntactic anomalies. They do NOT constitute proof of malicious intent."
        ),
        "disclaimer": (
            "This offline static analysis assesses heuristic structural indicators only "
            "and cannot establish whether a URL or domain is currently active, safe, or malicious."
        )
    }
