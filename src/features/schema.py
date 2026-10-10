"""
Feature Schema Definition and Metadata Catalog.

Defines specifications for all features across Word TF-IDF, Character TF-IDF,
Structural, and URL feature families.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class FeatureSpec:
    name: str
    family: str
    data_type: str
    source: str
    calculation: str
    missing_value_behavior: str
    representation: str
    available_at_inference: bool

# Catalogue of non-text numerical features
NUMERICAL_FEATURE_SCHEMA: List[FeatureSpec] = [
    # Structural Features
    FeatureSpec("subject_length", "structural", "float64", "subject", "Character count of subject", "defaults to 0.0", "dense", True),
    FeatureSpec("body_length", "structural", "float64", "plain_body", "Character count of plain-text body", "defaults to 0.0", "dense", True),
    FeatureSpec("total_length", "structural", "float64", "subject + body", "Combined character count", "defaults to 0.0", "dense", True),
    FeatureSpec("word_count", "structural", "float64", "subject + body", "Total whitespace-separated tokens", "defaults to 0.0", "dense", True),
    FeatureSpec("sentence_count", "structural", "float64", "subject + body", "Punctuation-split sentence count", "defaults to 0.0", "dense", True),
    FeatureSpec("uppercase_ratio", "structural", "float64", "subject + body", "Upper-case char count / total length", "defaults to 0.0", "dense", True),
    FeatureSpec("digit_ratio", "structural", "float64", "subject + body", "Digit char count / total length", "defaults to 0.0", "dense", True),
    FeatureSpec("whitespace_ratio", "structural", "float64", "subject + body", "Whitespace char count / total length", "defaults to 0.0", "dense", True),
    FeatureSpec("punctuation_ratio", "structural", "float64", "subject + body", "Punctuation char count / total length", "defaults to 0.0", "dense", True),
    FeatureSpec("exclamation_count", "structural", "float64", "subject + body", "Count of '!' characters", "defaults to 0.0", "dense", True),
    FeatureSpec("question_count", "structural", "float64", "subject + body", "Count of '?' characters", "defaults to 0.0", "dense", True),
    FeatureSpec("repeated_char_ratio", "structural", "float64", "subject + body", "Count of 3+ repeated chars / total length", "defaults to 0.0", "dense", True),
    FeatureSpec("html_present", "structural", "float64", "html_body", "Binary flag 1.0 if HTML body present", "defaults to 0.0", "dense", True),
    FeatureSpec("html_tag_count", "structural", "float64", "html_body", "Regex count of '<...>' tags", "defaults to 0.0", "dense", True),
    FeatureSpec("url_count", "structural", "float64", "body text / hrefs", "Total URLs extracted", "defaults to 0.0", "dense", True),
    FeatureSpec("unique_url_count", "structural", "float64", "body text / hrefs", "Unique URLs extracted", "defaults to 0.0", "dense", True),
    FeatureSpec("email_address_count", "structural", "float64", "subject + body", "Regex count of email addresses", "defaults to 0.0", "dense", True),
    FeatureSpec("attachment_count", "structural", "float64", "MIME headers", "Count of MIME attachments", "defaults to 0.0", "dense", True),
    FeatureSpec("mime_part_count", "structural", "float64", "MIME structure", "1.0 if singlepart, 2.0+ if multipart", "defaults to 1.0", "dense", True),

    # URL Features
    FeatureSpec("http_count", "url", "float64", "urls", "Count of http:// URLs", "defaults to 0.0", "dense", True),
    FeatureSpec("https_count", "url", "float64", "urls", "Count of https:// URLs", "defaults to 0.0", "dense", True),
    FeatureSpec("avg_url_length", "url", "float64", "urls", "Mean character length of URLs", "defaults to 0.0", "dense", True),
    FeatureSpec("max_url_length", "url", "float64", "urls", "Maximum character length of URLs", "defaults to 0.0", "dense", True),
    FeatureSpec("avg_hostname_length", "url", "float64", "urls", "Mean hostname length", "defaults to 0.0", "dense", True),
    FeatureSpec("max_hostname_length", "url", "float64", "urls", "Maximum hostname length", "defaults to 0.0", "dense", True),
    FeatureSpec("avg_dot_count", "url", "float64", "urls", "Mean dot count in hostnames", "defaults to 0.0", "dense", True),
    FeatureSpec("max_dot_count", "url", "float64", "urls", "Maximum dot count in hostnames", "defaults to 0.0", "dense", True),
    FeatureSpec("subdomain_count", "url", "float64", "urls", "Count of URLs with >2 dots in host", "defaults to 0.0", "dense", True),
    FeatureSpec("ip_host_count", "url", "float64", "urls", "Count of URLs with IPv4/v6 host", "defaults to 0.0", "dense", True),
    FeatureSpec("percent_encoding_count", "url", "float64", "urls", "Count of URLs with '%'", "defaults to 0.0", "dense", True),
    FeatureSpec("suspicious_port_count", "url", "float64", "urls", "Count of non-80/443 ports", "defaults to 0.0", "dense", True),
    FeatureSpec("shortener_count", "url", "float64", "urls", "Count of known shortening domains", "defaults to 0.0", "dense", True),
    FeatureSpec("query_string_count", "url", "float64", "urls", "Count of URLs with query params", "defaults to 0.0", "dense", True),
    FeatureSpec("fragment_count", "url", "float64", "urls", "Count of URLs with anchors (#)", "defaults to 0.0", "dense", True),
]

TEXT_FEATURE_SCHEMA = [
    FeatureSpec("word_tfidf", "word_tfidf", "float64", "subject + body", "Word 1-2 gram TF-IDF weights", "defaults to 0.0", "sparse", True),
    FeatureSpec("char_tfidf", "char_tfidf", "float64", "subject + body", "Character 3-5 gram TF-IDF weights", "defaults to 0.0", "sparse", True),
]


def get_full_schema_df() -> List[Dict[str, Any]]:
    """Return schema specification as list of dictionaries."""
    all_specs = NUMERICAL_FEATURE_SCHEMA + TEXT_FEATURE_SCHEMA
    return [spec.__dict__ for spec in all_specs]
