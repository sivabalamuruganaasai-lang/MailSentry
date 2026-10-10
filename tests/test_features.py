"""
Comprehensive Unit Tests for Feature Extractors and Feature Pipelines (Level 4).
"""

import tempfile
import joblib
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from src.features.word_tfidf import (
    WordTfidfExtractor,
    preprocess_text_light,
    preprocess_text_stopword_aware,
    preprocess_text_stemmed
)
from src.features.char_tfidf import CharTfidfExtractor, preprocess_text_char
from src.features.structural import StructuralFeatureExtractor, FEATURE_NAMES
from src.features.url_features import URLFeatureExtractor, URL_FEATURE_NAMES
from src.features.pipeline import CombinedFeaturePipeline


def test_deterministic_preprocessing_and_unicode_nfkd():
    # Full-width Unicode characters
    text = "Ｆｒｅｅ Ｍｏｎｅｙ Ｎｏｗ！！！"
    clean1 = preprocess_text_light(text)
    clean2 = preprocess_text_light(text)

    assert clean1 == clean2
    assert clean1 == "free money now!!!"


def test_preprocessing_modes():
    text = "The free offer is available now for you"
    light = preprocess_text_light(text)
    sw = preprocess_text_stopword_aware(text)
    stem = preprocess_text_stemmed(text)

    assert "the" in light
    assert "the" not in sw
    assert "free" in sw  # Preserved spam term!
    assert isinstance(stem, str)


def test_word_tfidf_train_only_fit_and_transform_consistency():
    train_texts = [
        "Free credit card offer now",
        "Important project update meeting tomorrow"
    ]
    test_texts = [
        "Free credit update"
    ]

    extractor = WordTfidfExtractor(mode="light", max_features=100)

    # Transform before fit must raise ValueError
    with pytest.raises(ValueError):
        extractor.transform(test_texts)

    # Fit on train
    extractor.fit(train_texts)
    train_mat = extractor.transform(train_texts)
    test_mat = extractor.transform(test_texts)

    assert issparse(train_mat)
    assert issparse(test_mat)
    assert train_mat.shape[0] == 2
    assert test_mat.shape[0] == 1
    # Test matrix must match train vocabulary size
    assert test_mat.shape[1] == train_mat.shape[1]

    # Deterministic transform consistency
    test_mat2 = extractor.transform(test_texts)
    np.testing.assert_array_equal(test_mat.toarray(), test_mat2.toarray())


def test_char_tfidf_obfuscation_handling():
    train_texts = ["free viagra offer", "fr33 v!agra offer"]
    extractor = CharTfidfExtractor(ngram_range=(3, 4), max_features=100)
    extractor.fit(train_texts)
    mat = extractor.transform(train_texts)

    assert mat.shape[0] == 2
    assert len(extractor.get_feature_names_out()) > 0


def test_structural_feature_extraction():
    record = {
        "subject": "URGENT!!",
        "plain_body": "Call 1800-555-0199 now for $1000 cash freeee!",
        "html_body": "<p>Click <a href='http://spam.org'>here</a></p>",
        "has_html": True,
        "urls": ["http://spam.org"],
        "url_count": 1,
        "attachment_count": 0,
        "is_multipart": False
    }

    extractor = StructuralFeatureExtractor(return_sparse=False)
    vec = extractor.transform([record])

    assert vec.shape == (1, len(FEATURE_NAMES))
    # Assert specific feature columns
    # subject_length is 1st column
    assert vec[0, 0] == len("URGENT!!")
    # html_present is 13th column
    assert vec[0, 12] == 1.0


def test_url_feature_extraction():
    record = {
        "urls": [
            "http://192.168.1.50:8080/login?user=admin#top",
            "https://bit.ly/3x8yZ"
        ]
    }

    extractor = URLFeatureExtractor(return_sparse=False)
    vec = extractor.transform([record])

    assert vec.shape == (1, len(URL_FEATURE_NAMES))
    # url_count is index 0
    assert vec[0, 0] == 2.0
    # ip_host_count is index 11
    assert vec[0, 11] == 1.0
    # suspicious_port_count is index 13
    assert vec[0, 13] == 1.0
    # shortener_count is index 14
    assert vec[0, 14] == 1.0


def test_empty_and_malformed_input_handling():
    empty_record = {}
    malformed_record = {"subject": None, "plain_body": None, "urls": "not a list"}

    s_extractor = StructuralFeatureExtractor()
    u_extractor = URLFeatureExtractor()

    s_vec = s_extractor.transform([empty_record, malformed_record])
    u_vec = u_extractor.transform([empty_record, malformed_record])

    assert s_vec.shape == (2, len(FEATURE_NAMES))
    assert u_vec.shape == (2, len(URL_FEATURE_NAMES))
    assert not np.isnan(s_vec).any()
    assert not np.isnan(u_vec).any()


def test_joblib_serialization_safety():
    pipeline = CombinedFeaturePipeline(feature_families=["word", "structural", "url"])
    records = [
        {"subject": "Hello", "plain_body": "World text http://test.com", "urls": ["http://test.com"]},
        {"subject": "Buy now", "plain_body": "Discount code 50%", "urls": []}
    ]

    pipeline.fit(records)
    mat_before = pipeline.transform(records)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        joblib.dump(pipeline, tmp.name)
        loaded_pipeline = joblib.load(tmp.name)

    mat_after = loaded_pipeline.transform(records)
    np.testing.assert_array_equal(mat_before.toarray(), mat_after.toarray())


def test_no_label_leakage():
    # Ensure extractor methods do not require or consume target labels
    records = [{"subject": "Test", "plain_body": "Body", "label": 1}]
    pipeline = CombinedFeaturePipeline()

    # Call fit/transform without passing y or expecting label in output
    pipeline.fit(records)
    mat = pipeline.transform(records)

    feature_names = pipeline.get_feature_names_out()
    assert "label" not in [f.lower() for f in feature_names]
    assert "x-spam" not in [f.lower() for f in feature_names]
