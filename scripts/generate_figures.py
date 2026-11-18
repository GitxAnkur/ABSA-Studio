"""Generate all report and application figures.

Usage:
    python -m scripts.generate_figures

Requires the training/evaluation artefacts produced by:
    python -m scripts.train_all
    python -m scripts.evaluate_all
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aspect_lexicon import load_lexicon
from src.config import (
    LEXICON_PATH,
    LSTM_META_PATH,
    METRICS_PATH,
    REPORTS_DIR,
)
from src.data_loader import load_dataset_stats
from src.lstm_model import load_metadata
from src.plots import (
    plot_architecture,
    plot_aspect_detection_comparison,
    plot_aspect_detection_per_aspect,
    plot_aspect_sentiment_heatmap,
    plot_confusion_matrices,
    plot_dataset_distribution,
    plot_error_confidence,
    plot_model_comparison,
    plot_per_aspect_f1,
    plot_plsa_topics,
    plot_training_curves,
    plot_workflow,
)


def main() -> None:
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"Metrics not found at {METRICS_PATH}. Run: python -m scripts.evaluate_all"
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    stats = load_dataset_stats()
    lexicon = load_lexicon(LEXICON_PATH)

    generated = [
        plot_architecture(),
        plot_workflow(),
        plot_dataset_distribution(stats),
        plot_aspect_sentiment_heatmap(stats),
        plot_aspect_detection_comparison(metrics),
        plot_aspect_detection_per_aspect(metrics),
        plot_model_comparison(metrics),
        plot_per_aspect_f1(metrics),
        plot_confusion_matrices(metrics),
        plot_plsa_topics(lexicon),
    ]

    if os.path.exists(LSTM_META_PATH):
        generated.append(plot_training_curves(load_metadata(LSTM_META_PATH)))

    errors_path = os.path.join(REPORTS_DIR, "error_analysis.csv")
    correct_path = os.path.join(REPORTS_DIR, "correct_examples.csv")
    if os.path.exists(errors_path) and os.path.exists(correct_path):
        errors = pd.read_csv(errors_path)
        correct = pd.read_csv(correct_path)
        if not errors.empty and not correct.empty:
            generated.append(plot_error_confidence(errors, correct))

    for path in generated:
        print(f"[FIGURE] {path}")


if __name__ == "__main__":
    main()
