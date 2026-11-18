"""Training routines for the ABSA models.

The module provides three families of models:

1. A pLSA + TF-ICF weighted-lexicon baseline for aspect detection.
2. Supervised TF-IDF models (one-vs-rest logistic regression for aspect
   detection and per-aspect classifiers for sentiment).
3. Aspect-specific BiLSTM sentiment classifiers.

All tuning is performed on a development split carved out of the official
*training* set; the official test set is touched only during final evaluation.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from .aspect_lexicon import (
    LexiconAspectDetector,
    build_aspect_lexicon,
    save_lexicon,
    tune_thresholds,
)
from .config import (
    ASPECT_DETECTOR_PATH,
    ASPECTS,
    ASPECT_THRESHOLD,
    DEV_SIZE,
    GLOVE_PATH,
    LR_C_GRID,
    LSTM_DIR,
    LSTM_META_PATH,
    MAX_SEQ_LEN,
    PLSA_MAX_ITER,
    PLSA_MAX_VOCAB,
    PLSA_MIN_DF,
    PLSA_PATH,
    PLSA_TOPICS,
    RANDOM_STATE,
    SENTIMENT_LABELS,
    SENTIMENT_MODELS_PATH,
    LEXICON_PATH,
    TFIDF_MAX_FEATURES,
)
from .features import build_vectorizer, ensure_writable
from .lstm_model import (
    Vocabulary,
    load_glove_subset,
    save_metadata,
    save_model,
    set_seed,
    train_aspect_model,
)
from .plsa import PLSA
from .preprocessing import tokenize
ASPECT_COLUMNS = [f"aspect_{aspect}" for aspect in ASPECTS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _aspect_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[ASPECT_COLUMNS].to_numpy(dtype=int)


def tune_aspect_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    grid: Sequence[float] | None = None,
) -> Dict[str, float]:
    """Tune one probability threshold per aspect on dev data."""
    if grid is None:
        grid = np.arange(0.10, 0.91, 0.05)
    thresholds = {}
    for aspect_idx, aspect in enumerate(ASPECTS):
        truth = y_true[:, aspect_idx]
        if truth.sum() == 0:
            thresholds[aspect] = ASPECT_THRESHOLD
            continue
        best_threshold, best_f1 = ASPECT_THRESHOLD, -1.0
        for threshold in grid:
            predictions = (probabilities[:, aspect_idx] >= threshold).astype(int)
            score = f1_score(truth, predictions, zero_division=0)
            if score > best_f1:
                best_f1, best_threshold = float(score), float(threshold)
        thresholds[aspect] = round(best_threshold, 4)
    return thresholds


def sentiment_macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def split_sentences_dev(
    frame: pd.DataFrame, dev_size: float = DEV_SIZE, seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random sentence-level split for aspect-detector tuning."""
    train, dev = train_test_split(
        frame, test_size=dev_size, random_state=seed, shuffle=True
    )
    return train.reset_index(drop=True), dev.reset_index(drop=True)


def split_sentiment_dev(
    frame: pd.DataFrame, aspect: str, dev_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified dev split for one aspect's sentiment rows."""
    subset = frame[frame["aspect"] == aspect].reset_index(drop=True)
    counts = subset["label"].value_counts()
    stratify = subset["label"] if counts.min() >= 2 else None
    train, dev = train_test_split(
        subset,
        test_size=dev_size,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )
    return train.reset_index(drop=True), dev.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Unsupervised aspect lexicon + baseline detector
# ---------------------------------------------------------------------------
def build_lexicon_artifacts(
    train_sentences: pd.DataFrame,
    dev_sentences: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[Dict, LexiconAspectDetector, Dict]:
    """Fit pLSA, induce lexicons and tune baseline thresholds."""
    docs = [
        tokenize(text, remove_stopwords=True, lemmatize=True)
        for text in train_sentences["text"].tolist()
    ]
    plsa = PLSA(
        n_topics=PLSA_TOPICS,
        max_iter=PLSA_MAX_ITER,
        random_state=RANDOM_STATE,
        min_df=PLSA_MIN_DF,
        max_features=PLSA_MAX_VOCAB,
    ).fit(docs)
    joblib.dump(plsa, PLSA_PATH)

    lexicon = build_aspect_lexicon(
        train_sentences["text"].tolist(),
        n_topics=PLSA_TOPICS,
        verbose=verbose,
        plsa=plsa,
    )
    save_lexicon(lexicon, LEXICON_PATH)

    detector = LexiconAspectDetector(lexicon)
    thresholds = tune_thresholds(
        detector,
        dev_sentences["text"].tolist(),
        _aspect_matrix(dev_sentences),
    )
    detector.thresholds = thresholds

    # Persist thresholds alongside the lexicon for the app/baseline use.
    lexicon["baseline_thresholds"] = thresholds
    save_lexicon(lexicon, LEXICON_PATH)

    y_true = _aspect_matrix(dev_sentences)
    dev_scores = np.array(
        [[detector.score(text)[aspect] for aspect in ASPECTS] for text in dev_sentences["text"]]
    )
    dev_predictions = np.array(
        [
            [int(dev_scores[row, idx] >= thresholds[aspect]) for idx, aspect in enumerate(ASPECTS)]
            for row in range(len(dev_sentences))
        ]
    )
    metrics = {
        "macro_f1": round(f1_score(y_true, dev_predictions, average="macro", zero_division=0), 6),
        "micro_f1": round(f1_score(y_true, dev_predictions, average="micro", zero_division=0), 6),
        "thresholds": thresholds,
    }
    if verbose:
        print(f"[LEXICON] Dev macro-F1={metrics['macro_f1']:.4f}, micro-F1={metrics['micro_f1']:.4f}")
    return lexicon, detector, metrics


# ---------------------------------------------------------------------------
# 2. Supervised aspect detector
# ---------------------------------------------------------------------------
def train_aspect_detector(
    train_sentences: pd.DataFrame,
    dev_sentences: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """Train and tune a one-vs-rest TF-IDF logistic-regression detector."""
    y_train = _aspect_matrix(train_sentences)
    y_dev = _aspect_matrix(dev_sentences)

    best = None
    for C in LR_C_GRID:
        vectorizer = build_vectorizer()
        X_train = ensure_writable(vectorizer.fit_transform(train_sentences["text"].tolist()))
        X_dev = ensure_writable(vectorizer.transform(dev_sentences["text"].tolist()))

        classifier = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            n_jobs=None,
        )
        classifier.fit(X_train, y_train)
        probabilities = classifier.predict_proba(X_dev)

        thresholds = tune_aspect_thresholds(y_dev, probabilities)
        predictions = np.column_stack(
            [
                (probabilities[:, idx] >= thresholds[aspect]).astype(int)
                for idx, aspect in enumerate(ASPECTS)
            ]
        )
        macro_f1 = f1_score(y_dev, predictions, average="macro", zero_division=0)
        micro_f1 = f1_score(y_dev, predictions, average="micro", zero_division=0)
        if verbose:
            print(f"[ASPECT] C={C:<5} dev macro-F1={macro_f1:.4f} micro-F1={micro_f1:.4f}")

        if best is None or macro_f1 > best["dev_macro_f1"]:
            best = {
                "C": C,
                "dev_macro_f1": float(macro_f1),
                "dev_micro_f1": float(micro_f1),
                "thresholds": thresholds,
            }

    # Refit on the full training split with the selected hyper-parameter.
    vectorizer = build_vectorizer()
    X_train = ensure_writable(vectorizer.fit_transform(train_sentences["text"].tolist()))
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=best["C"],
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        n_jobs=None,
    )
    classifier.fit(X_train, y_train)

    bundle = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "thresholds": best["thresholds"],
        "aspects": ASPECTS,
        "metadata": {
            "model_name": "tfidf_logreg",
            "C": best["C"],
            "dev_macro_f1": best["dev_macro_f1"],
            "dev_micro_f1": best["dev_micro_f1"],
            "thresholds": best["thresholds"],
        },
    }
    os.makedirs(os.path.dirname(ASPECT_DETECTOR_PATH), exist_ok=True)
    joblib.dump(bundle, ASPECT_DETECTOR_PATH)
    if verbose:
        print(f"[ASPECT] Best C={best['C']} saved to {ASPECT_DETECTOR_PATH}")
    return bundle, best


# ---------------------------------------------------------------------------
# 3. Classical sentiment classifiers
# ---------------------------------------------------------------------------
def _fit_logreg(X_train, y_train, C: float):
    model = LogisticRegression(
        C=C,
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_sentiment_models(
    train_sentiment: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """Train TF-IDF + logistic regression and calibrated linear SVM models.

    Returns the saved bundle and the selected hyper-parameters / dev scores.
    """
    shared_vectorizer = build_vectorizer()
    shared_vectorizer.fit(train_sentiment["text"].tolist())

    logreg_models: Dict[str, object] = {}
    svm_models: Dict[str, object] = {}
    metadata: Dict = {"models": {}, "dev_size": DEV_SIZE}

    for aspect in ASPECTS:
        subset = train_sentiment[train_sentiment["aspect"] == aspect].reset_index(drop=True)
        if subset.empty:
            continue

        dev_train, dev = split_sentiment_dev(subset, aspect, DEV_SIZE, RANDOM_STATE)
        # Tune C on the dev split for logistic regression.
        dev_vectorizer = build_vectorizer()
        X_dev_train = ensure_writable(dev_vectorizer.fit_transform(dev_train["text"].tolist()))
        X_dev = ensure_writable(dev_vectorizer.transform(dev["text"].tolist()))

        best_c, best_f1 = LR_C_GRID[0], -1.0
        for C in LR_C_GRID:
            model = _fit_logreg(X_dev_train, dev_train["label_id"].to_numpy(), C)
            predictions = model.predict(X_dev)
            score = sentiment_macro_f1(dev["label_id"].to_numpy(), predictions)
            if score > best_f1:
                best_c, best_f1 = C, score

        # Final logistic regression on all rows for this aspect.
        X_full = ensure_writable(shared_vectorizer.transform(subset["text"].tolist()))
        final_logreg = _fit_logreg(X_full, subset["label_id"].to_numpy(), best_c)

        # Calibrated linear SVM (fixed C=1.0: it is a comparison model).
        svm = CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
            cv=3,
            method="sigmoid",
        )
        svm.fit(X_full, subset["label_id"].to_numpy())

        logreg_models[aspect] = final_logreg
        svm_models[aspect] = svm
        metadata["models"][aspect] = {
            "logreg_C": best_c,
            "dev_macro_f1": round(float(best_f1), 6),
            "train_size": int(len(subset)),
        }
        if verbose:
            print(
                f"[SENTIMENT] {aspect:<14} n={len(subset):<5} best C={best_c:<4} "
                f"dev macro-F1={best_f1:.4f}"
            )

    logreg_bundle = {
        "vectorizer": shared_vectorizer,
        "models": logreg_models,
        "classes": SENTIMENT_LABELS,
        "metadata": {"model_name": "tfidf_logreg", **metadata},
    }
    svm_bundle = {
        "vectorizer": shared_vectorizer,
        "models": svm_models,
        "classes": SENTIMENT_LABELS,
        "metadata": {"model_name": "tfidf_svm", **metadata},
    }

    os.makedirs(os.path.dirname(SENTIMENT_MODELS_PATH), exist_ok=True)
    joblib.dump(logreg_bundle, SENTIMENT_MODELS_PATH)
    joblib.dump(svm_bundle, os.path.join(os.path.dirname(SENTIMENT_MODELS_PATH), "sentiment_svm.joblib"))
    with open(os.path.join(os.path.dirname(SENTIMENT_MODELS_PATH), "sentiment_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    return logreg_bundle, metadata


# ---------------------------------------------------------------------------
# 4. LSTM training
# ---------------------------------------------------------------------------
def train_lstm_models(
    train_sentences: pd.DataFrame,
    train_sentiment: pd.DataFrame,
    verbose: bool = True,
) -> Dict:
    """Train one BiLSTM sentiment classifier per aspect."""
    set_seed()
    os.makedirs(LSTM_DIR, exist_ok=True)

    vocab = Vocabulary().build(train_sentences["text"].tolist())
    embedding_matrix, glove_coverage = load_glove_subset(GLOVE_PATH, vocab)
    if verbose:
        print(
            f"[LSTM] Vocabulary={len(vocab):,} | GloVe coverage={glove_coverage:.1%} "
            f"({os.path.basename(GLOVE_PATH)})"
        )

    metadata: Dict = {
        "model_name": "bilstm",
        "vocab_size": len(vocab),
        "embed_dim": int(embedding_matrix.shape[1]),
        "max_seq_len": MAX_SEQ_LEN,
        "glove_coverage": round(float(glove_coverage), 4),
        "glove_file": os.path.basename(GLOVE_PATH),
        "aspects": {},
    }

    for aspect in ASPECTS:
        subset = train_sentiment[train_sentiment["aspect"] == aspect].reset_index(drop=True)
        if subset.empty:
            continue
        fit_rows, dev_rows = split_sentiment_dev(subset, aspect, DEV_SIZE, RANDOM_STATE)
        if verbose:
            print(f"[LSTM] Training aspect '{aspect}' ({len(fit_rows)} train / {len(dev_rows)} dev)")

        model, model_meta = train_aspect_model(
            aspect=aspect,
            train_texts=fit_rows["text"].tolist(),
            train_labels=fit_rows["label_id"].tolist(),
            dev_texts=dev_rows["text"].tolist(),
            dev_labels=dev_rows["label_id"].tolist(),
            vocab=vocab,
            embedding_matrix=embedding_matrix,
            verbose=verbose,
            final_texts=subset["text"].tolist(),
            final_labels=subset["label_id"].tolist(),
        )

        path = os.path.join(LSTM_DIR, f"{aspect}.pt")
        save_model(model, vocab, path, meta=model_meta)
        metadata["aspects"][aspect] = model_meta

    save_metadata(metadata, LSTM_META_PATH)
    if verbose:
        print(f"[LSTM] Models saved under {LSTM_DIR}")
    return metadata


# ---------------------------------------------------------------------------
# 5. Majority baseline (for the comparison table)
# ---------------------------------------------------------------------------
def majority_baseline(train_sentiment: pd.DataFrame) -> Dict[str, int]:
    """Most frequent class per aspect in the training data."""
    majority = {}
    for aspect in ASPECTS:
        subset = train_sentiment[train_sentiment["aspect"] == aspect]
        if subset.empty:
            majority[aspect] = 2
            continue
        majority[aspect] = int(subset["label_id"].value_counts().idxmax())
    return majority
