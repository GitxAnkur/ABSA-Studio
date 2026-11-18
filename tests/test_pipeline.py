"""End-to-end and unit tests for the ABSA project.

Run:
    pytest -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.aspect_lexicon import LexiconAspectDetector, tune_thresholds
from src.config import (
    ASPECT_DETECTOR_PATH,
    ASPECTS,
    LEXICON_PATH,
    LSTM_DIR,
    METRICS_PATH,
    SENTIMENT_LABELS,
    SENTIMENT_MODELS_PATH,
)
from src.data_loader import (
    build_sentence_frame,
    build_sentiment_frame,
    load_aspect_sentiment,
    load_sentences,
    parse_semeval_xml,
)
from src.evaluate import evaluate_aspect_detection, evaluate_sentiment
from src.features import build_vectorizer, ensure_writable
from src.plsa import PLSA
from src.predict import AspectSentimentEngine
from src.preprocessing import clean_text, normalize_text, tokenize
from src.tf_icf import tf_icf


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def test_semeval_files_parse():
    train = parse_semeval_xml(os.path.join(PROJECT_ROOT, "data/raw/semeval2014/Restaurants_Train_v2.xml"))
    test = parse_semeval_xml(os.path.join(PROJECT_ROOT, "data/raw/semeval2014/Restaurants_Test_Gold.xml"))
    assert len(train) == 3041
    assert len(test) == 800
    assert all(sentence["text"] for sentence in train[:50])
    categories = {cat["aspect"] for sentence in train for cat in sentence["categories"]}
    assert categories == set(ASPECTS)


def test_processed_frames_and_labels():
    train_sentences = load_sentences("train")
    train_sentiment = load_aspect_sentiment("train")
    assert {"text", "aspects", "num_aspects", *[f"aspect_{a}" for a in ASPECTS]}.issubset(
        train_sentences.columns
    )
    assert set(train_sentiment["aspect"]).issubset(set(ASPECTS))
    assert set(train_sentiment["label"]).issubset(set(SENTIMENT_LABELS))
    assert train_sentiment["text"].str.len().min() > 0
    assert train_sentiment["label_id"].between(0, 2).all()


def test_frame_builders():
    sentences = [
        {
            "sentence_id": "1",
            "text": "Great food and slow service.",
            "aspect_terms": [],
            "categories": [
                {"aspect": "food", "polarity": "positive"},
                {"aspect": "service", "polarity": "negative"},
            ],
        }
    ]
    sentence_frame = build_sentence_frame(sentences)
    sentiment_frame = build_sentiment_frame(sentences)
    assert sentence_frame.loc[0, "num_aspects"] == 2
    assert sentence_frame.loc[0, "aspect_food"] == 1
    assert sentence_frame.loc[0, "aspect_price"] == 0
    assert len(sentiment_frame) == 2


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_preprocessing_preserves_negation():
    tokens = tokenize("The food wasn't good, but the staff were not helpful!")
    assert "not" in tokens
    assert "good" in tokens
    assert "wasn't" not in tokens
    assert tokenize("great food") == ["great", "food"]


def test_preprocessing_removes_urls_and_html():
    cleaned = normalize_text("<b>Great</b> food http://example.com !!!")
    assert "http" not in cleaned
    assert "<b>" not in cleaned
    assert "exclamationmark" in cleaned
    assert "!" in tokenize("<b>Great</b> food http://example.com !!!")


def test_clean_text_is_string():
    value = clean_text("AMAZING food!")
    assert isinstance(value, str)
    assert value == "amazing food !"


# ---------------------------------------------------------------------------
# pLSA / TF-ICF
# ---------------------------------------------------------------------------
def test_plsa_training_and_transform():
    docs = [
        ["food", "tasty", "pizza", "staff", "friendly"],
        ["service", "slow", "staff", "rude"],
        ["price", "cheap", "value", "food"],
        ["ambience", "noisy", "music", "loud"],
    ] * 5
    model = PLSA(n_topics=2, max_iter=5, random_state=1, min_df=1, max_features=50).fit(docs)
    assert model.P_w_z_.shape[0] == 2
    assert np.allclose(model.P_w_z_.sum(axis=1), 1.0)
    assert np.allclose(model.P_z_d_.sum(axis=1), 1.0)
    transformed = model.transform(docs[:2])
    assert transformed.shape == (2, 2)
    assert np.allclose(transformed.sum(axis=1), 1.0)


def test_tf_icf_weights_are_cluster_specific():
    clusters = {
        0: [["food", "tasty", "pizza"], ["food", "great"]],
        1: [["staff", "rude", "service"], ["staff", "slow"]],
    }
    scores = tf_icf(clusters)
    assert scores[0]["food"] > 0
    assert scores[1]["staff"] > 0
    assert "food" not in scores[1]


def test_vectorizer_roundtrip():
    texts = ["great food", "slow service", "reasonable price", "noisy room"] * 3
    vectorizer = build_vectorizer(min_df=1, max_features=1000)
    features = ensure_writable(vectorizer.fit_transform(texts))
    assert features.shape[0] == len(texts)
    assert features.shape[1] > 0
    assert features.data.flags.writeable


# ---------------------------------------------------------------------------
# Lexicon baseline
# ---------------------------------------------------------------------------
def test_lexicon_detector_and_threshold_tuning():
    lexicon = {
        "aspects": {
            "food": {"seed_terms": ["food", "pizza"], "expanded_terms": [{"term": "pasta", "weight": 1.0}]},
            "service": {"seed_terms": ["staff", "waiter"], "expanded_terms": []},
            "price": {"seed_terms": ["price", "expensive"], "expanded_terms": []},
            "ambience": {"seed_terms": ["noisy", "music"], "expanded_terms": []},
            "miscellaneous": {"seed_terms": ["restaurant"], "expanded_terms": []},
        }
    }
    detector = LexiconAspectDetector(lexicon)
    scores = detector.score("The pizza was good but the staff were rude")
    assert scores["food"] > 0
    assert scores["service"] > 0
    probabilities = detector.probabilities("pizza and pasta")
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())

    texts = ["pizza", "staff", "price", "noisy", "restaurant"]
    labels = np.eye(len(ASPECTS), dtype=int)
    thresholds = tune_thresholds(detector, texts, labels)
    assert set(thresholds) == set(ASPECTS)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_sentiment_metrics():
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 1, 2, 1, 0]
    metrics = evaluate_sentiment(y_true, y_pred, ["food"] * 5)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert 0 <= metrics["macro_f1"] <= 1
    assert np.array(metrics["confusion_matrix"]).shape == (3, 3)
    assert "food" in metrics["per_aspect"]


def test_aspect_metrics():
    y_true = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 0]])
    probabilities = np.array([[0.9, 0.1, 0.1, 0.1, 0.8], [0.2, 0.7, 0.3, 0.2, 0.1]])
    thresholds = {aspect: 0.5 for aspect in ASPECTS}
    metrics = evaluate_aspect_detection(y_true, probabilities, thresholds)
    assert metrics["micro_f1"] == pytest.approx(1.0)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["per_aspect"]["food"]["support"] == 1


# ---------------------------------------------------------------------------
# Saved artefacts and inference
# ---------------------------------------------------------------------------
ARTEFACTS_READY = all(
    os.path.exists(path)
    for path in (ASPECT_DETECTOR_PATH, SENTIMENT_MODELS_PATH, LEXICON_PATH)
)


@pytest.fixture(scope="module")
def engine():
    if not ARTEFACTS_READY:
        pytest.skip("Model artefacts missing; run scripts/train_all.py first.")
    return AspectSentimentEngine()


def test_artefacts_exist():
    assert os.path.exists(ASPECT_DETECTOR_PATH)
    assert os.path.exists(SENTIMENT_MODELS_PATH)
    assert os.path.exists(LEXICON_PATH)
    assert os.path.isdir(LSTM_DIR)
    assert os.path.exists(METRICS_PATH)


def test_engine_prediction(engine):
    text = "The pasta was delicious and the staff were friendly, but the prices were too high."
    result = engine.predict(text, aspect_method="supervised", sentiment_model="tfidf_logreg")
    assert result["aspects"]
    assert set(result["detected_aspects"]).issubset(set(ASPECTS))
    for item in result["aspects"]:
        assert item["aspect"] in ASPECTS
        assert item["sentiment"] in SENTIMENT_LABELS
        assert 0.0 <= item["confidence"] <= 1.0
        assert abs(sum(item["probabilities"].values()) - 1.0) < 1e-5
    assert result["cleaned_text"].startswith("the pasta")


def test_engine_all_models(engine):
    text = "Great food but terrible service."
    for model in engine.available_sentiment_models:
        result = engine.predict(text, sentiment_model=model)
        assert result["aspects"]


def test_engine_lexicon_aspects(engine):
    result = engine.predict("the pizza was tasty but the room was noisy")
    assert result["aspect_method"] == "supervised"
    result = engine.predict(
        "the pizza was tasty but the room was noisy", aspect_method="lexicon"
    )
    assert result["aspect_method"] == "lexicon"
    assert result["aspects"]


def test_engine_rejects_empty_input(engine):
    with pytest.raises(ValueError):
        engine.predict("")
    with pytest.raises(ValueError):
        engine.predict("   ")


def test_metrics_file_is_valid_json():
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert "aspect_detection" in metrics
    assert "sentiment" in metrics
    assert metrics["test_set"]["sentences"] == 800


# ---------------------------------------------------------------------------
# Application smoke test
# ---------------------------------------------------------------------------
def test_streamlit_app_starts():
    from streamlit.testing.v1 import AppTest

    if not ARTEFACTS_READY:
        pytest.skip("Model artefacts missing.")
    app = AppTest.from_file(os.path.join(PROJECT_ROOT, "app.py"), default_timeout=90).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Analyse Review").run()
    assert not app.exception
