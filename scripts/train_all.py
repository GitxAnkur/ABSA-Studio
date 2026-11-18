"""Train every ABSA component.

Usage:
    python -m scripts.train_all                 # full pipeline
    python -m scripts.train_all --skip-lstm     # fast classical run
    python -m scripts.train_all --skip-lexicon  # skip pLSA baseline

The script is idempotent: it overwrites previous artefacts under ``models/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from src.data_loader import load_aspect_sentiment, load_sentences
from src.train import (
    build_lexicon_artifacts,
    majority_baseline,
    split_sentences_dev,
    train_aspect_detector,
    train_lstm_models,
    train_sentiment_models,
)


def _ensure_data() -> None:
    required = [
        os.path.join(PROCESSED_DIR, "train_sentences.csv"),
        os.path.join(PROCESSED_DIR, "test_sentences.csv"),
        os.path.join(PROCESSED_DIR, "train_aspect_sentiment.csv"),
        os.path.join(PROCESSED_DIR, "test_aspect_sentiment.csv"),
    ]
    if not all(os.path.exists(path) for path in required):
        print("[SETUP] Processed data missing - running data preparation first.")
        from src.data_loader import prepare_datasets

        prepare_datasets(verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ABSA pipeline")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip BiLSTM training")
    parser.add_argument("--skip-lexicon", action="store_true", help="Skip pLSA baseline training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    started = time.time()
    _ensure_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_sentences = load_sentences("train")
    train_sentiment = load_aspect_sentiment("train")
    fit_sentences, dev_sentences = split_sentences_dev(train_sentences)

    print(
        f"\n[TRAIN] {len(train_sentences)} training sentences "
        f"({len(fit_sentences)} fit / {len(dev_sentences)} dev), "
        f"{len(train_sentiment)} aspect-sentiment rows\n"
    )

    summary = {
        "seed": args.seed,
        "train_sentences": len(train_sentences),
        "train_sentiment_rows": len(train_sentiment),
        "dev_sentences": len(dev_sentences),
        "components": {},
    }

    if not args.skip_lexicon:
        print("=" * 70)
        print("1/4  pLSA + TF-ICF lexicon baseline")
        print("=" * 70)
        _, _, lexicon_metrics = build_lexicon_artifacts(
            fit_sentences, dev_sentences, verbose=True
        )
        summary["components"]["lexicon_baseline"] = lexicon_metrics

    print("=" * 70)
    print("2/4  Supervised aspect detector (TF-IDF + logistic regression)")
    print("=" * 70)
    _, aspect_metrics = train_aspect_detector(
        fit_sentences, dev_sentences, verbose=True
    )
    summary["components"]["aspect_detector"] = {
        key: value for key, value in aspect_metrics.items()
    }

    print("=" * 70)
    print("3/4  Aspect-specific sentiment classifiers (TF-IDF + LR / SVM)")
    print("=" * 70)
    _, sentiment_metrics = train_sentiment_models(train_sentiment, verbose=True)
    summary["components"]["sentiment_classical"] = sentiment_metrics
    summary["components"]["majority_baseline"] = majority_baseline(train_sentiment)

    if not args.skip_lstm:
        print("=" * 70)
        print("4/4  Aspect-specific BiLSTM sentiment classifiers")
        print("=" * 70)
        lstm_metrics = train_lstm_models(
            fit_sentences, train_sentiment, verbose=True
        )
        summary["components"]["bilstm"] = lstm_metrics
    else:
        print("[SKIP] BiLSTM training skipped.")

    summary["training_seconds"] = round(time.time() - started, 2)
    summary_path = os.path.join(REPORTS_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 70)
    print(f"Training complete in {summary['training_seconds']:.1f}s")
    print(f"Summary: {summary_path}")
    print("Next: python -m scripts.evaluate_all")


if __name__ == "__main__":
    main()
