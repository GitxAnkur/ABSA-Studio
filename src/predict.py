"""Inference engine used by the Streamlit application and the CLI.

The engine combines three independent components:

* **Aspect detection** - supervised TF-IDF logistic regression (default) or the
  unsupervised pLSA + TF-ICF lexicon baseline.
* **Aspect-specific sentiment classification** - BiLSTM, TF-IDF logistic
  regression, or calibrated linear SVM.
* **Explanation** - per-aspect class probabilities and the cleaned text.

All predictions come from saved model artefacts; nothing is hard-coded.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd

from .aspect_lexicon import LexiconAspectDetector, load_lexicon
from .config import (
    ASPECT_DETECTOR_PATH,
    ASPECT_DISPLAY,
    ASPECT_THRESHOLD,
    ASPECTS,
    ID_TO_LABEL,
    LEXICON_PATH,
    LSTM_DIR,
    SENTIMENT_LABELS,
    SENTIMENT_MODELS_PATH,
)
from .features import ensure_writable
from .lstm_model import load_model, predict_proba
from .preprocessing import clean_text, is_valid_text, tokenize

SENTIMENT_MODEL_NAMES = {
    "bilstm": "BiLSTM (GloVe + BiLSTM)",
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "tfidf_svm": "TF-IDF + Linear SVM",
}


class AspectSentimentEngine:
    """End-to-end ABSA inference engine."""

    def __init__(
        self,
        aspect_detector_path: str = ASPECT_DETECTOR_PATH,
        sentiment_models_path: str = SENTIMENT_MODELS_PATH,
        lexicon_path: str = LEXICON_PATH,
        lstm_dir: str = LSTM_DIR,
    ) -> None:
        for path in (aspect_detector_path, sentiment_models_path, lexicon_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required model artefact not found: {path}. "
                    "Run: python -m scripts.train_all"
                )

        self.aspect_bundle = joblib.load(aspect_detector_path)
        self.sentiment_bundle = joblib.load(sentiment_models_path)
        svm_path = os.path.join(os.path.dirname(sentiment_models_path), "sentiment_svm.joblib")
        self.svm_bundle = joblib.load(svm_path) if os.path.exists(svm_path) else None
        self.lexicon = load_lexicon(lexicon_path)
        self.lexicon_detector = LexiconAspectDetector(
            self.lexicon,
            thresholds=self.lexicon.get("baseline_thresholds", {aspect: ASPECT_THRESHOLD for aspect in ASPECTS}),
        )

        self.lstm_models: Dict[str, tuple] = {}
        self.lstm_meta: Dict = {}
        if os.path.isdir(lstm_dir):
            for aspect in ASPECTS:
                path = os.path.join(lstm_dir, f"{aspect}.pt")
                if os.path.exists(path):
                    self.lstm_models[aspect] = load_model(path)
            meta_path = os.path.join(lstm_dir, "meta.json")
            if os.path.exists(meta_path):
                import json

                with open(meta_path, "r", encoding="utf-8") as handle:
                    self.lstm_meta = json.load(handle)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def available_sentiment_models(self) -> List[str]:
        models = []
        if self.lstm_models:
            models.append("bilstm")
        models.append("tfidf_logreg")
        if self.svm_bundle is not None:
            models.append("tfidf_svm")
        return models

    def model_info(self) -> Dict:
        return {
            "aspect_detector": self.aspect_bundle.get("metadata", {}),
            "sentiment_models": self.available_sentiment_models,
            "lstm": self.lstm_meta,
        }

    # ------------------------------------------------------------------
    # Aspect detection
    # ------------------------------------------------------------------
    def detect_aspects(
        self,
        text: str,
        method: str = "supervised",
        threshold: float | None = None,
    ) -> Dict[str, float]:
        """Return aspect-presence probabilities for one review."""
        if method == "lexicon":
            probabilities = self.lexicon_detector.probabilities(text)
        elif method == "supervised":
            vectorizer = self.aspect_bundle["vectorizer"]
            classifier = self.aspect_bundle["classifier"]
            features = ensure_writable(vectorizer.transform([text]))
            probabilities = classifier.predict_proba(features)[0]
            probabilities = {
                aspect: float(probabilities[idx]) for idx, aspect in enumerate(ASPECTS)
            }
        else:
            raise ValueError(f"Unknown aspect detection method: {method}")

        if threshold is not None:
            # Keep per-aspect tuned thresholds, but never below the user's floor.
            base = self.aspect_bundle.get("thresholds", {})
            probabilities = {
                aspect: float(score)
                for aspect, score in probabilities.items()
            }
            self._effective_thresholds = {
                aspect: max(float(base.get(aspect, ASPECT_THRESHOLD)), float(threshold))
                for aspect in ASPECTS
            }
        else:
            self._effective_thresholds = {
                aspect: float(self.aspect_bundle.get("thresholds", {}).get(aspect, ASPECT_THRESHOLD))
                for aspect in ASPECTS
            }
        return probabilities

    def _detected(
        self, probabilities: Dict[str, float]
    ) -> List[str]:
        return [
            aspect
            for aspect in ASPECTS
            if probabilities.get(aspect, 0.0) >= self._effective_thresholds.get(aspect, 0.5)
        ]

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------
    def predict_sentiment(
        self, text: str, aspect: str, model: str = "bilstm"
    ) -> Dict:
        """Return sentiment probabilities for one (text, aspect) pair."""
        if model not in self.available_sentiment_models:
            raise ValueError(
                f"Sentiment model '{model}' is not available. "
                f"Choose one of {self.available_sentiment_models}."
            )

        if model == "bilstm":
            lstm_model, vocab, _meta = self.lstm_models[aspect]
            probabilities = predict_proba(lstm_model, [text], vocab)[0]
            classes = list(range(len(SENTIMENT_LABELS)))
        elif model == "tfidf_logreg":
            model_object = self.sentiment_bundle["models"].get(aspect)
            if model_object is None:
                raise ValueError(f"No logistic-regression model for aspect '{aspect}'.")
            features = ensure_writable(self.sentiment_bundle["vectorizer"].transform([text]))
            probabilities = model_object.predict_proba(features)[0]
            classes = [int(c) for c in model_object.classes_]
        elif model == "tfidf_svm":
            model_object = self.svm_bundle["models"].get(aspect)
            if model_object is None:
                raise ValueError(f"No SVM model for aspect '{aspect}'.")
            features = ensure_writable(self.svm_bundle["vectorizer"].transform([text]))
            probabilities = model_object.predict_proba(features)[0]
            classes = [int(c) for c in model_object.classes_]
        else:  # pragma: no cover - guarded above
            raise ValueError(model)

        probability_map = {label: 0.0 for label in SENTIMENT_LABELS}
        for idx, class_id in enumerate(classes):
            if idx < len(probabilities):
                probability_map[ID_TO_LABEL[int(class_id)]] = float(probabilities[idx])
        total = sum(probability_map.values())
        if total > 0:
            probability_map = {label: value / total for label, value in probability_map.items()}
        best_label = max(probability_map, key=probability_map.get)
        return {
            "label": best_label,
            "confidence": float(probability_map[best_label]),
            "probabilities": probability_map,
        }

    # ------------------------------------------------------------------
    # End-to-end prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        text: str,
        aspect_method: str = "supervised",
        sentiment_model: str = "bilstm",
        threshold: float | None = None,
        top_k: int | None = None,
        include_all: bool = True,
    ) -> Dict:
        """Run the complete ABSA pipeline for one review.

        Parameters
        ----------
        text:
            Raw review text.
        aspect_method:
            ``"supervised"`` (TF-IDF logistic regression) or ``"lexicon"``
            (pLSA + TF-ICF weighted lexicon baseline).
        sentiment_model:
            ``"bilstm"``, ``"tfidf_logreg"`` or ``"tfidf_svm"``.
        threshold:
            Optional minimum aspect confidence (per-aspect tuned thresholds
            are still applied).
        top_k:
            If given, always analyse the ``k`` most probable aspects even when
            they fall below the threshold.
        include_all:
            Include probabilities for all aspects in the result.
        """
        if not is_valid_text(text, min_tokens=2):
            raise ValueError("Please enter a review with at least two words.")

        aspect_probabilities = self.detect_aspects(text, method=aspect_method, threshold=threshold)
        detected = self._detected(aspect_probabilities)

        ranked = sorted(aspect_probabilities.items(), key=lambda item: item[1], reverse=True)
        if top_k is not None and top_k > 0:
            selected = [aspect for aspect, _ in ranked[:top_k]]
        else:
            selected = detected

        # Fallback: never return an empty analysis.  Mark the result clearly.
        low_confidence = False
        if not selected:
            selected = [ranked[0][0]]
            low_confidence = True

        aspect_results = []
        for aspect in selected:
            sentiment = self.predict_sentiment(text, aspect, model=sentiment_model)
            aspect_results.append(
                {
                    "aspect": aspect,
                    "display": ASPECT_DISPLAY.get(aspect, aspect.title()),
                    "aspect_probability": float(aspect_probabilities.get(aspect, 0.0)),
                    "sentiment": sentiment["label"],
                    "confidence": sentiment["confidence"],
                    "probabilities": sentiment["probabilities"],
                }
            )

        overall = self._overall_sentiment(aspect_results, aspect_probabilities)
        return {
            "text": text,
            "cleaned_text": clean_text(text),
            "aspect_method": aspect_method,
            "sentiment_model": sentiment_model,
            "thresholds": dict(self._effective_thresholds),
            "aspect_probabilities": aspect_probabilities if include_all else {},
            "detected_aspects": detected,
            "aspects": aspect_results,
            "low_confidence": low_confidence,
            "overall": overall,
        }

    @staticmethod
    def _overall_sentiment(
        aspect_results: List[Dict], aspect_probabilities: Dict[str, float]
    ) -> Dict:
        """Weighted average of aspect-level probabilities (display only)."""
        if not aspect_results:
            return {"label": "neutral", "confidence": 0.0, "probabilities": {}}

        weights = np.array(
            [max(result["aspect_probability"], 1e-3) for result in aspect_results]
        )
        weights = weights / weights.sum()
        probabilities = {label: 0.0 for label in SENTIMENT_LABELS}
        for weight, result in zip(weights, aspect_results):
            for label, value in result["probabilities"].items():
                probabilities[label] += float(weight * value)
        label = max(probabilities, key=probabilities.get)
        return {
            "label": label,
            "confidence": float(probabilities[label]),
            "probabilities": probabilities,
        }

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def predict_batch(
        self,
        texts: Sequence[str],
        aspect_method: str = "supervised",
        sentiment_model: str = "bilstm",
    ) -> pd.DataFrame:
        """Predict a list of reviews and return a tidy DataFrame."""
        rows = []
        for text in texts:
            try:
                result = self.predict(
                    text,
                    aspect_method=aspect_method,
                    sentiment_model=sentiment_model,
                )
            except ValueError:
                continue
            for aspect_result in result["aspects"]:
                rows.append(
                    {
                        "review_text": text,
                        "aspect": aspect_result["aspect"],
                        "aspect_probability": round(aspect_result["aspect_probability"], 4),
                        "sentiment": aspect_result["sentiment"],
                        "confidence": round(aspect_result["confidence"], 4),
                    }
                )
        return pd.DataFrame(rows)


def summarise_prediction(result: Dict) -> str:
    """Human-readable summary used by the CLI demo."""
    if not result["aspects"]:
        return "No aspect detected."
    parts = [
        f"{item['display']}: {item['sentiment']} ({item['confidence']:.1%})"
        for item in result["aspects"]
    ]
    return "; ".join(parts)


def token_count(text: str) -> int:
    """Small helper for the UI."""
    return len(tokenize(text))
