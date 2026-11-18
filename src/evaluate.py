"""Evaluation metrics used for both model selection and final reporting."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import ASPECTS, SENTIMENT_LABELS


def evaluate_aspect_detection(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Dict[str, float],
    y_pred: np.ndarray | None = None,
) -> Dict:
    """Multi-label aspect-detection metrics for one model.

    ``probabilities`` may be either calibrated probabilities or arbitrary
    (e.g. lexicon) scores; they are used for per-aspect ROC-AUC.  When
    ``y_pred`` is supplied it is used directly (required when the score scale
    makes a fixed threshold inappropriate).
    """
    if y_pred is None:
        y_pred = np.column_stack(
            [
                (probabilities[:, idx] >= thresholds.get(aspect, 0.5)).astype(int)
                for idx, aspect in enumerate(ASPECTS)
            ]
        )

    per_aspect = {}
    for idx, aspect in enumerate(ASPECTS):
        precision, recall, f1, _support = precision_recall_fscore_support(
            y_true[:, idx], y_pred[:, idx], average="binary", zero_division=0
        )
        entry = {
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "support": int(y_true[:, idx].sum()),
        }
        try:
            entry["roc_auc"] = round(
                float(roc_auc_score(y_true[:, idx], probabilities[:, idx])), 6
            )
        except ValueError:
            entry["roc_auc"] = None
        per_aspect[aspect] = entry

    return {
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 6),
        "subset_accuracy": round(float(np.mean(np.all(y_true == y_pred, axis=1))), 6),
        "per_aspect": per_aspect,
        "thresholds": {aspect: float(thresholds.get(aspect, 0.5)) for aspect in ASPECTS},
        "y_pred": y_pred,
        "y_true": y_true,
        "probabilities": probabilities,
    }


def evaluate_sentiment(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    aspects: Sequence[str] | None = None,
) -> Dict:
    """Three-class sentiment metrics, optionally broken down by aspect."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    aspects = list(aspects) if aspects is not None else [None] * len(y_true)

    labels = list(range(len(SENTIMENT_LABELS)))
    report = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "n_samples": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report["per_class"] = {
        label: {
            "precision": round(float(precision[idx]), 6),
            "recall": round(float(recall[idx]), 6),
            "f1": round(float(f1[idx]), 6),
            "support": int(support[idx]),
        }
        for idx, label in enumerate(SENTIMENT_LABELS)
    }

    per_aspect = {}
    for aspect in ASPECTS:
        mask = np.array([item == aspect for item in aspects])
        if not mask.any():
            continue
        aspect_truth = y_true[mask]
        aspect_pred = y_pred[mask]
        per_aspect[aspect] = {
            "n_samples": int(mask.sum()),
            "accuracy": round(float(accuracy_score(aspect_truth, aspect_pred)), 6),
            "macro_f1": round(
                float(f1_score(aspect_truth, aspect_pred, average="macro", zero_division=0)), 6
            ),
            "confusion_matrix": confusion_matrix(
                aspect_truth, aspect_pred, labels=labels
            ).tolist(),
        }
    report["per_aspect"] = per_aspect
    return report


def classification_report_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    """Detailed per-class report used in the textual report."""
    from sklearn.metrics import classification_report

    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(SENTIMENT_LABELS))),
        target_names=SENTIMENT_LABELS,
        output_dict=True,
        zero_division=0,
    )


def format_report(report: Dict) -> str:
    """Render the sklearn report dictionary as a readable table."""
    lines = [
        f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
        "-" * 52,
    ]
    for label in SENTIMENT_LABELS:
        row = report.get(label)
        if not row:
            continue
        lines.append(
            f"{label:<12}{row['precision']:>10.3f}{row['recall']:>10.3f}"
            f"{row['f1-score']:>10.3f}{int(row['support']):>10d}"
        )
    for key in ("macro avg", "weighted avg"):
        row = report.get(key)
        if row:
            lines.append(
                f"{key:<12}{row['precision']:>10.3f}{row['recall']:>10.3f}"
                f"{row['f1-score']:>10.3f}{int(row['support']):>10d}"
            )
    return "\n".join(lines)
