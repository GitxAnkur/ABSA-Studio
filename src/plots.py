"""Figures and diagrams for the report and the Streamlit application."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import ASPECTS, ASPECT_DISPLAY, FIGURES_DIR, SENTIMENT_LABELS

PALETTE = {
    "blue": "#2563EB",
    "green": "#10B981",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "purple": "#8B5CF6",
    "slate": "#64748B",
    "light": "#E2E8F0",
    "dark": "#0F172A",
}
MODEL_COLORS = {
    "majority": PALETTE["slate"],
    "tfidf_logreg": PALETTE["blue"],
    "tfidf_svm": PALETTE["purple"],
    "bilstm": PALETTE["green"],
}
MODEL_LABELS = {
    "majority": "Majority class",
    "tfidf_logreg": "TF-IDF + LR",
    "tfidf_svm": "TF-IDF + SVM",
    "bilstm": "BiLSTM",
    "lexicon_plsa_tficf": "pLSA + TF-ICF lexicon",
}
LABEL_COLORS = {
    "negative": PALETTE["red"],
    "neutral": PALETTE["amber"],
    "positive": PALETTE["green"],
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "axes.edgecolor": "#CBD5E1",
            "figure.facecolor": "white",
        }
    )


def _save(fig, name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Dataset figures
# ---------------------------------------------------------------------------
def plot_dataset_distribution(stats: Dict) -> str:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    train_aspects = stats["splits"]["train"]["aspect_distribution"]
    test_aspects = stats["splits"]["test"]["aspect_distribution"]
    x = np.arange(len(ASPECTS))
    width = 0.38
    axes[0].bar(
        x - width / 2,
        [train_aspects.get(a, 0) for a in ASPECTS],
        width,
        label="Train",
        color=PALETTE["blue"],
    )
    axes[0].bar(
        x + width / 2,
        [test_aspects.get(a, 0) for a in ASPECTS],
        width,
        label="Test",
        color=PALETTE["amber"],
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([ASPECT_DISPLAY[a] for a in ASPECTS], rotation=15, ha="right")
    axes[0].set_ylabel("Annotations")
    axes[0].set_title("Aspect annotations per split")
    axes[0].legend(frameon=False)

    sentiments = SENTIMENT_LABELS + ["conflict"]
    train_pol = stats["splits"]["train"]["polarity_distribution"]
    test_pol = stats["splits"]["test"]["polarity_distribution"]
    x2 = np.arange(len(sentiments))
    axes[1].bar(
        x2 - width / 2,
        [train_pol.get(s, 0) for s in sentiments],
        width,
        label="Train",
        color=PALETTE["blue"],
    )
    axes[1].bar(
        x2 + width / 2,
        [test_pol.get(s, 0) for s in sentiments],
        width,
        label="Test",
        color=PALETTE["amber"],
    )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([s.title() for s in sentiments])
    axes[1].set_ylabel("Annotations")
    axes[1].set_title("Polarity distribution")
    axes[1].legend(frameon=False)

    fig.suptitle("SemEval-2014 Task 4 Restaurant dataset", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "dataset_distribution.png")


def plot_aspect_sentiment_heatmap(stats: Dict) -> str:
    _style()
    distribution = stats["splits"]["train"]["aspect_sentiment_distribution"]
    matrix = np.zeros((len(ASPECTS), 3), dtype=int)
    for row, aspect in enumerate(ASPECTS):
        for col, label in enumerate(SENTIMENT_LABELS):
            matrix[row, col] = distribution.get(f"{aspect}/{label}", 0)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels([s.title() for s in SENTIMENT_LABELS])
    ax.set_yticks(range(len(ASPECTS)))
    ax.set_yticklabels([ASPECT_DISPLAY[a] for a in ASPECTS])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else PALETTE["dark"],
                fontsize=11,
                fontweight="bold",
            )
    ax.set_title("Training labels by aspect and polarity")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Annotations")
    fig.tight_layout()
    return _save(fig, "aspect_sentiment_distribution.png")


# ---------------------------------------------------------------------------
# Aspect detection
# ---------------------------------------------------------------------------
def plot_aspect_detection_comparison(metrics: Dict) -> str:
    _style()
    detection = metrics["aspect_detection"]
    names = list(detection)
    x = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(
        x - width / 2,
        [detection[name]["micro_f1"] for name in names],
        width,
        label="Micro-F1",
        color=PALETTE["blue"],
    )
    ax.bar(
        x + width / 2,
        [detection[name]["macro_f1"] for name in names],
        width,
        label="Macro-F1",
        color=PALETTE["green"],
    )
    for idx, name in enumerate(names):
        ax.text(
            idx - width / 2,
            detection[name]["micro_f1"] + 0.01,
            f"{detection[name]['micro_f1']:.3f}",
            ha="center",
            fontsize=9,
        )
        ax.text(
            idx + width / 2,
            detection[name]["macro_f1"] + 0.01,
            f"{detection[name]['macro_f1']:.3f}",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(name, name) for name in names])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Aspect detection: supervised model vs unsupervised lexicon baseline")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return _save(fig, "aspect_detection_comparison.png")


def plot_aspect_detection_per_aspect(metrics: Dict) -> str:
    _style()
    detection = metrics["aspect_detection"]
    names = list(detection)
    x = np.arange(len(ASPECTS))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for idx, name in enumerate(names):
        values = [detection[name]["per_aspect"][a]["f1"] for a in ASPECTS]
        ax.bar(
            x - 0.4 + width / 2 + idx * width,
            values,
            width,
            label=MODEL_LABELS.get(name, name),
            color=[PALETTE["blue"], PALETTE["amber"], PALETTE["purple"]][idx % 3],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ASPECT_DISPLAY[a] for a in ASPECTS], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 score")
    ax.set_title("Aspect detection F1 by aspect category")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "aspect_detection_per_aspect.png")


# ---------------------------------------------------------------------------
# Sentiment models
# ---------------------------------------------------------------------------
def plot_model_comparison(metrics: Dict) -> str:
    _style()
    sentiment = metrics["sentiment"]
    names = list(sentiment)
    x = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(
        x - width / 2,
        [sentiment[name]["accuracy"] for name in names],
        width,
        label="Accuracy",
        color=PALETTE["blue"],
    )
    ax.bar(
        x + width / 2,
        [sentiment[name]["macro_f1"] for name in names],
        width,
        label="Macro-F1",
        color=PALETTE["green"],
    )
    for idx, name in enumerate(names):
        ax.text(
            idx - width / 2,
            sentiment[name]["accuracy"] + 0.01,
            f"{sentiment[name]['accuracy']:.3f}",
            ha="center",
            fontsize=8.5,
        )
        ax.text(
            idx + width / 2,
            sentiment[name]["macro_f1"] + 0.01,
            f"{sentiment[name]['macro_f1']:.3f}",
            ha="center",
            fontsize=8.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(name, name) for name in names], rotation=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Aspect-specific sentiment classification on the test set")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return _save(fig, "model_comparison.png")


def plot_per_aspect_f1(metrics: Dict) -> str:
    _style()
    sentiment = metrics["sentiment"]
    names = [name for name in sentiment if name != "majority"]
    x = np.arange(len(ASPECTS))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for idx, name in enumerate(names):
        values = [
            sentiment[name]["per_aspect"].get(a, {}).get("macro_f1", np.nan)
            for a in ASPECTS
        ]
        ax.bar(
            x - 0.4 + width / 2 + idx * width,
            values,
            width,
            label=MODEL_LABELS.get(name, name),
            color=MODEL_COLORS.get(name, PALETTE["slate"]),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ASPECT_DISPLAY[a] for a in ASPECTS], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Sentiment macro-F1 by aspect category")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    return _save(fig, "per_aspect_f1.png")


def plot_confusion_matrices(metrics: Dict) -> str:
    _style()
    sentiment = metrics["sentiment"]
    names = [name for name in ["majority", "tfidf_logreg", "tfidf_svm", "bilstm"] if name in sentiment]
    cols = 2
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 4.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        matrix = np.array(sentiment[name]["confusion_matrix"])
        normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(3):
            for col in range(3):
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]}\n({normalized[row, col]:.0%})",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if normalized[row, col] > 0.55 else PALETTE["dark"],
                )
        ax.set_xticks(range(3))
        ax.set_xticklabels([s.title() for s in SENTIMENT_LABELS])
        ax.set_yticks(range(3))
        ax.set_yticklabels([s.title() for s in SENTIMENT_LABELS])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(MODEL_LABELS.get(name, name))

    for ax in axes[len(names):]:
        ax.axis("off")

    fig.suptitle("Confusion matrices (test set, all aspects)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "confusion_matrices.png")


def plot_training_curves(meta: Dict) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["amber"], PALETTE["red"], PALETTE["purple"]]
    for idx, (aspect, info) in enumerate(meta.get("aspects", {}).items()):
        history = info.get("history", [])
        if not history:
            continue
        ax.plot(
            [item["epoch"] for item in history],
            [item["dev_macro_f1"] for item in history],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=colors[idx % len(colors)],
            label=ASPECT_DISPLAY.get(aspect, aspect),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Development macro-F1")
    ax.set_title("BiLSTM learning curves by aspect")
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "bilstm_training_curves.png")


def plot_error_confidence(errors: pd.DataFrame, correct: pd.DataFrame) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bins = np.linspace(0.3, 1.0, 25)
    ax.hist(correct["confidence"], bins=bins, alpha=0.65, label="Correct", color=PALETTE["green"])
    ax.hist(errors["confidence"], bins=bins, alpha=0.65, label="Incorrect", color=PALETTE["red"])
    ax.set_xlabel("Model confidence")
    ax.set_ylabel("Number of predictions")
    ax.set_title("Confidence of correct vs incorrect predictions")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, "error_confidence.png")


# ---------------------------------------------------------------------------
# PLSA topics
# ---------------------------------------------------------------------------
def plot_plsa_topics(lexicon: Dict, n_topics: int = 6, n_terms: int = 8) -> str:
    _style()
    topics = lexicon.get("topics", {})
    selected = list(topics.items())[:n_topics]
    if not selected:
        raise ValueError("No topics found in lexicon JSON.")

    cols = 2
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (topic_id, terms) in zip(axes, selected):
        chosen = terms[:n_terms][::-1]
        ax.barh(chosen, np.arange(1, len(chosen) + 1), color=PALETTE["blue"], alpha=0.85)
        ax.set_title(f"Topic {topic_id}", fontsize=11)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_xlabel("Rank")
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle("pLSA topic-word distributions (TF-ICF top terms)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "plsa_topics.png")


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------
def _box(ax, xy, width, height, text, color, text_color="white", fontsize=9.5):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight="bold",
        wrap=True,
    )


def _arrow(ax, start, end, color="#94A3B8"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.6, color=color
    )
    ax.add_patch(arrow)


def plot_architecture() -> str:
    _style()
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("System architecture", fontsize=16, fontweight="bold", pad=8)

    _box(ax, (0.02, 0.66), 0.18, 0.16, "SemEval-2014\nrestaurant XML", PALETTE["dark"], fontsize=10)
    _box(ax, (0.25, 0.66), 0.18, 0.16, "Data loader\n+ validation", PALETTE["slate"], fontsize=10)
    _box(ax, (0.48, 0.66), 0.20, 0.16, "Preprocessing\nnormalise · tokenise\nnegation · lemma", PALETTE["blue"], fontsize=9.5)
    _box(ax, (0.73, 0.66), 0.24, 0.16, "Two task views\naspect labels · sentiment labels", PALETTE["purple"], fontsize=9.5)

    _box(ax, (0.03, 0.38), 0.26, 0.16, "Aspect detection\nTF-IDF (word + char)\n→ one-vs-rest LR", PALETTE["blue"], fontsize=9.5)
    _box(ax, (0.35, 0.38), 0.28, 0.16, "Sentiment classification\nTF-IDF + LR / SVM\nBiLSTM (GloVe)", PALETTE["green"], fontsize=9.5)
    _box(ax, (0.69, 0.38), 0.28, 0.16, "Unsupervised baseline\npLSA topics → TF-ICF\n→ aspect lexicon", PALETTE["amber"], fontsize=9.5)

    _box(ax, (0.03, 0.10), 0.28, 0.15, "Evaluation\nmacro-F1 · reports\nconfusion matrices", PALETTE["red"], fontsize=9.5)
    _box(ax, (0.36, 0.10), 0.28, 0.15, "Saved artefacts\njoblib · PyTorch\nJSON lexicon", PALETTE["slate"], fontsize=9.5)
    _box(ax, (0.69, 0.10), 0.28, 0.15, "Streamlit application\noverview · analyse\nperformance · dataset", PALETTE["purple"], fontsize=9.5)

    _arrow(ax, (0.20, 0.74), (0.25, 0.74))
    _arrow(ax, (0.43, 0.74), (0.48, 0.74))
    _arrow(ax, (0.68, 0.74), (0.73, 0.74))
    _arrow(ax, (0.82, 0.66), (0.82, 0.54))
    _arrow(ax, (0.85, 0.66), (0.55, 0.54))
    _arrow(ax, (0.80, 0.66), (0.20, 0.54))
    _arrow(ax, (0.18, 0.38), (0.18, 0.25))
    _arrow(ax, (0.49, 0.38), (0.49, 0.25))
    _arrow(ax, (0.82, 0.38), (0.82, 0.25))
    _arrow(ax, (0.50, 0.25), (0.50, 0.18))
    _arrow(ax, (0.72, 0.175), (0.69, 0.175))
    _arrow(ax, (0.32, 0.175), (0.36, 0.175))

    fig.tight_layout()
    return _save(fig, "architecture.png")


def plot_workflow() -> str:
    _style()
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("NLP workflow", fontsize=16, fontweight="bold", pad=10)

    steps = [
        ("1. Data", "SemEval XML\n3,841 reviews"),
        ("2. Cleaning", "HTML/URL removal\ncontractions\nunicode NFKC"),
        ("3. Tokens", "regex tokeniser\nnegation kept\noptional lemma"),
        ("4. Features", "word 1-2 grams\nchar 3-5 grams\nTF-IDF"),
        ("5. Models", "aspect detector\nsentiment model\npLSA baseline"),
        ("6. Evaluation", "accuracy\nmacro-F1\nconfusion matrix"),
        ("7. Demo", "Streamlit app\nlive inference"),
    ]
    width = 0.118
    gap = 0.014
    x = 0.015
    for idx, (title, body) in enumerate(steps):
        color = [PALETTE["blue"], PALETTE["slate"], PALETTE["purple"], PALETTE["amber"], PALETTE["green"], PALETTE["red"], PALETTE["dark"]][idx]
        _box(ax, (x, 0.30), width, 0.42, "", color)
        ax.text(x + width / 2, 0.62, title, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
        ax.text(x + width / 2, 0.44, body, ha="center", va="center", color="white", fontsize=8.2)
        if idx < len(steps) - 1:
            _arrow(ax, (x + width + 0.002, 0.51), (x + width + gap - 0.002, 0.51), color=PALETTE["slate"])
        x += width + gap

    fig.tight_layout()
    return _save(fig, "workflow.png")
