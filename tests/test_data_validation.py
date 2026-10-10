"""
Unit tests for data validation, hashing, and deduplication logic.
"""

import pytest
from src.data.deduplicate import (
    compute_raw_sha256,
    compute_normalized_sha256,
    extract_char_shingles,
    jaccard_similarity
)
from src.data.inventory import FOLDER_LABEL_MAP


def test_label_mapping():
    assert FOLDER_LABEL_MAP["easy_ham"] == 0
    assert FOLDER_LABEL_MAP["easy_ham_2"] == 0
    assert FOLDER_LABEL_MAP["hard_ham"] == 0
    assert FOLDER_LABEL_MAP["spam"] == 1
    assert FOLDER_LABEL_MAP["spam_2"] == 1


def test_deterministic_raw_sha256():
    raw_b = b"Subject: Hello\n\nWorld"
    hash1 = compute_raw_sha256(raw_b)
    hash2 = compute_raw_sha256(raw_b)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_normalized_sha256_insensitivity_to_whitespace():
    hash1 = compute_normalized_sha256("Subject Test", "Hello   World\n\n", "")
    hash2 = compute_normalized_sha256("subject test", "hello world", "")
    assert hash1 == hash2


def test_jaccard_similarity():
    s1 = extract_char_shingles("The quick brown fox", k=4)
    s2 = extract_char_shingles("The quick brown fox!", k=4)
    s3 = extract_char_shingles("Completely different string", k=4)

    sim_high = jaccard_similarity(s1, s2)
    sim_low = jaccard_similarity(s1, s3)

    assert sim_high > 0.80
    assert sim_low < 0.20
