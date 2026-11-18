"""Evaluate every trained model on the official SemEval-2014 test set.

Usage:
    python -m scripts.evaluate_all

Outputs
-------
outputs/reports/metrics.json
outputs/reports/test_aspect_predictions.csv
outputs/reports/test_sentiment_predictions.csv
outputs/reports/classification_report.txt
outputs/reports/error_analysis.csv
outputs/reports/correct_examples.csv
outputs/tables/model_comparison.csv
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aspect_lexicon import LexiconAspectDetector, load_lexicon
from src.config import (
    ASPECT_DETECTOR_PATH,
    ASPECTS,
    LEXICON_PATH,
    LSTM_DIR,
    METRICS_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SENTIMENT_LABELS,
    SENTIMENT_MODELS_PATH,
    TABLES_DIR,
)
from src.data_loader import load_aspect_sentiment, load_dataset_stats, load_sentences
from src.evaluate import (
    classification_report_dict,
    evaluate_aspect_detection,
    evaluate_sentiment,
    format_report,
)
from src.features import ensure_writable
from src.lstm_model import load_model, predict_proba
from src.train import majority_baseline

ASPECT_COLUMNS = [f"aspect_{aspect}" for aspect in ASPECTS]


def _aspect_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[ASPECT_COLUMNS].to_numpy(dtype=int)


# ---------------------------------------------------------------------------
# Aspect detection
# ---------------------------------------------------------------------------
def evaluate_aspect_models(test_sentences: pd.DataFrame) -> dict:
    results = {}
    y_true = _aspect_matrix(test_sentences)

    # 1. Supervised TF-IDF logistic regression.
    aspect_bundle = joblib.load(ASPECT_DETECTOR_PATH)
    features = ensure_writable(
        aspect_bundle["vectorizer"].transform(test_sentences["text"].tolist())
    )
    probabilities = aspect_bundle["classifier"].predict_proba(features)
    results["tfidf_logreg"] = evaluate_aspect_detection(
        y_true, probabilities, aspect_bundle["thresholds"]
    )

    # 2. Unsupervised pLSA + TF-ICF lexicon baseline.
    lexicon = load_lexicon(LEXICON_PATH)
    detector = LexiconAspectDetector(lexicon, lexicon.get("baseline_thresholds"))
    scores = np.array(
        [[detector.score(text)[aspect] for aspect in ASPECTS] for text in test_sentences["text"]]
    )
    lexicon_pred = np.array(
        [
            [int(scores[row, idx] >= detector.thresholds[aspect]) for idx, aspect in enumerate(ASPECTS)]
            for row in range(len(test_sentences))
        ]
    )
    results["lexicon_plsa_tficf"] = evaluate_aspect_detection(
        y_true, scores, detector.thresholds, y_pred=lexicon_pred
    )

    # Persist sentence-level predictions for inspection.
    prediction_frame = test_sentences[["sentence_id", "text", "aspects"]].copy()
    for idx, aspect in enumerate(ASPECTS):
        prediction_frame[f"prob_{aspect}"] = np.round(probabilities[:, idx], 6)
        prediction_frame[f"pred_{aspect}"] = (
            probabilities[:, idx] >= aspect_bundle["thresholds"][aspect]
        ).astype(int)
        prediction_frame[f"lexicon_pred_{aspect}"] = lexicon_pred[:, idx]
    prediction_frame.to_csv(
        os.path.join(REPORTS_DIR, "test_aspect_predictions.csv"), index=False
    )

    # Remove un-serialisable arrays from the saved metrics.
    for model in results.values():
        model.pop("y_true", None)
        model.pop("y_pred", None)
        model.pop("probabilities", None)
    return results


# ---------------------------------------------------------------------------
# Sentiment classification
# ---------------------------------------------------------------------------
def _predict_classical(bundle: dict, frame: pd.DataFrame) -> tuple:
    vectorizer = bundle["vectorizer"]
    features = ensure_writable(vectorizer.transform(frame["text"].tolist()))
    predictions = np.full(len(frame), -1, dtype=int)
    confidences = np.zeros(len(frame), dtype=float)
    probability_rows = [None] * len(frame)

    for aspect in ASPECTS:
        mask = (frame["aspect"] == aspect).to_numpy()
        if not mask.any() or aspect not in bundle["models"]:
            continue
        model = bundle["models"][aspect]
        subset_features = features[mask]
        subset_probs = model.predict_proba(subset_features)
        classes = [int(c) for c in model.classes_]
        indices = np.where(mask)[0]
        for row_idx, probs in zip(indices, subset_probs):
            probability_rows[row_idx] = {
                SENTIMENT_LABELS[class_id]: round(float(probs[pos]), 6)
                for pos, class_id in enumerate(classes)
            }
            best_pos = int(np.argmax(probs))
            predictions[row_idx] = classes[best_pos]
            confidences[row_idx] = float(probs[best_pos])
    return predictions, confidences, probability_rows


def _predict_lstm(frame: pd.DataFrame) -> tuple:
    predictions = np.full(len(frame), -1, dtype=int)
    confidences = np.zeros(len(frame), dtype=float)
    probability_rows = [None] * len(frame)

    for aspect in ASPECTS:
        model_path = os.path.join(LSTM_DIR, f"{aspect}.pt")
        if not os.path.exists(model_path):
            continue
        model, vocab, _meta = load_model(model_path)
        mask = (frame["aspect"] == aspect).to_numpy()
        if not mask.any():
            continue
        texts = frame.loc[mask, "text"].tolist()
        probs = predict_proba(model, texts, vocab)
        indices = np.where(mask)[0]
        for row_idx, row_probs in zip(indices, probs):
            probability_rows[row_idx] = {
                label: round(float(row_probs[idx]), 6)
                for idx, label in enumerate(SENTIMENT_LABELS)
            }
            predictions[row_idx] = int(np.argmax(row_probs))
            confidences[row_idx] = float(np.max(row_probs))
    return predictions, confidences, probability_rows


def evaluate_sentiment_models(
    test_sentiment: pd.DataFrame, train_sentiment: pd.DataFrame
) -> dict:
    results = {}
    y_true = test_sentiment["label_id"].to_numpy()
    aspects = test_sentiment["aspect"].tolist()

    # 1. Majority-class baseline.
    majority = majority_baseline(train_sentiment)
    majority_pred = np.array([majority[aspect] for aspect in aspects])
    results["majority"] = evaluate_sentiment(y_true, majority_pred, aspects)

    prediction_frame = test_sentiment[
        ["sentence_id", "text", "aspect", "label"]
    ].copy()

    # 2. TF-IDF + logistic regression.
    logreg_bundle = joblib.load(SENTIMENT_MODELS_PATH)
    pred, conf, probs = _predict_classical(logreg_bundle, test_sentiment)
    valid = pred >= 0
    results["tfidf_logreg"] = evaluate_sentiment(
        y_true[valid], pred[valid], [a for a, keep in zip(aspects, valid) if keep]
    )
    prediction_frame["pred_tfidf_logreg"] = [
        SENTIMENT_LABELS[p] if p >= 0 else None for p in pred
    ]
    prediction_frame["conf_tfidf_logreg"] = np.round(conf, 6)

    # 3. Calibrated TF-IDF linear SVM.
    svm_path = os.path.join(MODELS_DIR, "sentiment_svm.joblib")
    if os.path.exists(svm_path):
        svm_bundle = joblib.load(svm_path)
        pred_svm, conf_svm, _ = _predict_classical(svm_bundle, test_sentiment)
        valid_svm = pred_svm >= 0
        results["tfidf_svm"] = evaluate_sentiment(
            y_true[valid_svm],
            pred_svm[valid_svm],
            [a for a, keep in zip(aspects, valid_svm) if keep],
        )
        prediction_frame["pred_tfidf_svm"] = [
            SENTIMENT_LABELS[p] if p >= 0 else None for p in pred_svm
        ]
        prediction_frame["conf_tfidf_svm"] = np.round(conf_svm, 6)

    # 4. BiLSTM.
    if os.path.isdir(LSTM_DIR) and any(f.endswith(".pt") for f in os.listdir(LSTM_DIR)):
        pred_lstm, conf_lstm, probs_lstm = _predict_lstm(test_sentiment)
        valid_lstm = pred_lstm >= 0
        results["bilstm"] = evaluate_sentiment(
            y_true[valid_lstm],
            pred_lstm[valid_lstm],
            [a for a, keep in zip(aspects, valid_lstm) if keep],
        )
        prediction_frame["pred_bilstm"] = [
            SENTIMENT_LABELS[p] if p >= 0 else None for p in pred_lstm
        ]
        prediction_frame["conf_bilstm"] = np.round(conf_lstm, 6)
        if probs_lstm and probs_lstm[0] is not None:
            for label in SENTIMENT_LABELS:
                prediction_frame[f"prob_bilstm_{label}"] = [
                    row.get(label) if row else None for row in probs_lstm
                ]
    else:
        pred_lstm = np.full(len(test_sentiment), -1, dtype=int)
        conf_lstm = np.zeros(len(test_sentiment))
        probs_lstm = [None] * len(test_sentiment)

    prediction_frame["pred_majority"] = [
        SENTIMENT_LABELS[p] if p >= 0 else None for p in majority_pred
    ]
    prediction_frame.to_csv(
        os.path.join(REPORTS_DIR, "test_sentiment_predictions.csv"), index=False
    )

    # Error analysis for the strongest model (BiLSTM when available).
    analysis_pred = pred_lstm if "bilstm" in results else pred
    analysis_conf = conf_lstm if "bilstm" in results else conf
    analysis_name = "bilstm" if "bilstm" in results else "tfidf_logreg"
    errors = prediction_frame[
        (analysis_pred >= 0) & (analysis_pred != y_true)
    ].copy()
    errors["gold"] = [SENTIMENT_LABELS[i] for i in y_true[(analysis_pred >= 0) & (analysis_pred != y_true)]]
    errors["predicted"] = [SENTIMENT_LABELS[i] for i in analysis_pred[(analysis_pred >= 0) & (analysis_pred != y_true)]]
    errors["confidence"] = analysis_conf[(analysis_pred >= 0) & (analysis_pred != y_true)]
    errors["model"] = analysis_name
    errors.sort_values("confidence", ascending=False, inplace=True)
    errors.head(50).to_csv(
        os.path.join(REPORTS_DIR, "error_analysis.csv"), index=False
    )

    correct_mask = (analysis_pred >= 0) & (analysis_pred == y_true)
    correct = prediction_frame[correct_mask].copy()
    correct["gold"] = [SENTIMENT_LABELS[i] for i in y_true[correct_mask]]
    correct["predicted"] = [SENTIMENT_LABELS[i] for i in analysis_pred[correct_mask]]
    correct["confidence"] = analysis_conf[correct_mask]
    correct["model"] = analysis_name
    correct.sort_values("confidence", ascending=False, inplace=True)
    correct.head(40).to_csv(
        os.path.join(REPORTS_DIR, "correct_examples.csv"), index=False
    )

    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def build_comparison_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for name, value in metrics["sentiment"].items():
        rows.append(
            {
                "model": name,
                "accuracy": value["accuracy"],
                "macro_f1": value["macro_f1"],
                "weighted_f1": value["weighted_f1"],
                "n_samples": value["n_samples"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    os.makedirs(TABLES_DIR, exist_ok=True)
    frame.to_csv(os.path.join(TABLES_DIR, "model_comparison.csv"), index=False)

    aspect_rows = []
    for name, value in metrics["aspect_detection"].items():
        aspect_rows.append(
            {
                "model": name,
                "micro_f1": value["micro_f1"],
                "macro_f1": value["macro_f1"],
                "hamming_loss": value["hamming_loss"],
                "subset_accuracy": value["subset_accuracy"],
            }
        )
    pd.DataFrame(aspect_rows).sort_values("macro_f1", ascending=False).to_csv(
        os.path.join(TABLES_DIR, "aspect_detection_comparison.csv"), index=False
    )
    return frame


def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    test_sentences = load_sentences("test")
    test_sentiment = load_aspect_sentiment("test")
    train_sentiment = load_aspect_sentiment("train")

    print(f"[EVAL] Test sentences: {len(test_sentences)} | sentiment rows: {len(test_sentiment)}")

    aspect_metrics = evaluate_aspect_models(test_sentences)
    for name, value in aspect_metrics.items():
        print(
            f"[ASPECT] {name:<20} micro-F1={value['micro_f1']:.4f} "
            f"macro-F1={value['macro_f1']:.4f}"
        )

    sentiment_metrics = evaluate_sentiment_models(test_sentiment, train_sentiment)
    for name, value in sentiment_metrics.items():
        print(
            f"[SENTIMENT] {name:<16} accuracy={value['accuracy']:.4f} "
            f"macro-F1={value['macro_f1']:.4f}"
        )

    best_sentiment = max(
        (name for name in sentiment_metrics if name != "majority"),
        key=lambda name: sentiment_metrics[name]["macro_f1"],
    )
    best_predictions = pd.read_csv(
        os.path.join(REPORTS_DIR, "test_sentiment_predictions.csv")
    )
    best_predictions = best_predictions.dropna(subset=[f"pred_{best_sentiment}"])
    report = classification_report_dict(
        [SENTIMENT_LABELS.index(x) for x in best_predictions["label"]],
        [SENTIMENT_LABELS.index(x) for x in best_predictions[f"pred_{best_sentiment}"]],
    )
    with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as handle:
        handle.write(f"Best sentiment model: {best_sentiment}\n\n")
        handle.write(format_report(report))
        handle.write("\n")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_set": {
            "sentences": len(test_sentences),
            "aspect_sentiment_rows": len(test_sentiment),
        },
        "dataset": load_dataset_stats(),
        "aspect_detection": aspect_metrics,
        "sentiment": sentiment_metrics,
        "best_sentiment_model": best_sentiment,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    build_comparison_table(metrics)
    print(f"\n[EVAL] Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
