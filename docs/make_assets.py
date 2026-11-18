"""Generate diagrams and evaluation charts for the ABSA Studio guide.

All figures are built from the implementation and from
``outputs/reports/metrics.json``; nothing is hard-coded beyond the values that
are also present in that report.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE, "assets")
ROOT = os.path.dirname(BASE)
os.makedirs(ASSETS, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette and typography
# ---------------------------------------------------------------------------
NAVY = "#0F172A"
INDIGO = "#4338CA"
PURPLE = "#7C3AED"
BLUE = "#2563EB"
GREEN = "#15803D"
AMBER = "#B45309"
RED = "#B91C1C"
MUTED = "#475569"
BORDER = "#CBD5E1"
SLATE = "#F8FAFC"

TONES = {
    "indigo": ("#EEF2FF", INDIGO),
    "blue": ("#EFF6FF", BLUE),
    "purple": ("#F5F3FF", PURPLE),
    "green": ("#F0FDF4", GREEN),
    "amber": ("#FFFBEB", AMBER),
    "red": ("#FEF2F2", RED),
    "slate": ("#F1F5F9", "#475569"),
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "text.color": NAVY,
        "svg.fonttype": "none",
    }
)


def canvas(width: float, height: float):
    fig = plt.figure(figsize=(width, height), dpi=260)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    sub="",
    tone="indigo",
    num=None,
    title_size=10.6,
    sub_size=7.8,
    num_color=None,
    sub_offset=0.11,
):
    fill, edge = TONES[tone]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.07",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=fill,
            zorder=3,
        )
    )
    if num is not None:
        cx, cy = x + 0.19, y + h - 0.19
        ax.add_patch(
            plt.Circle(
                (cx, cy),
                0.115,
                facecolor=num_color or edge,
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
            )
        )
        ax.text(
            cx,
            cy - 0.005,
            str(num),
            ha="center",
            va="center",
            color="white",
            fontsize=7.6,
            fontweight="bold",
            zorder=6,
        )
        text_y = y + h - 0.10
        ha = "left"
        tx = x + 0.37
    else:
        text_y = y + h - 0.13
        ha = "center"
        tx = x + w / 2

    ax.text(
        tx,
        text_y,
        title,
        ha=ha,
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=NAVY,
        linespacing=1.18,
        zorder=4,
    )
    if sub:
        ax.text(
            x + w / 2,
            y + sub_offset,
            sub,
            ha="center",
            va="bottom",
            fontsize=sub_size,
            color=MUTED,
            linespacing=1.25,
            zorder=4,
        )


def arrow(ax, start, end, color="#94A3B8", ls="-", rad=0.0, lw=1.15):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9.5,
            linewidth=lw,
            linestyle=ls,
            color=color,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
    )


# ---------------------------------------------------------------------------
# Figure 1: System architecture
# ---------------------------------------------------------------------------
def architecture():
    W, H = 10, 5.3
    fig, ax = canvas(W, H)

    ax.add_patch(
        Rectangle((0.15, 2.95), 9.7, 2.22, facecolor=SLATE, edgecolor="none", zorder=0)
    )
    ax.add_patch(
        Rectangle((0.15, 0.18), 9.7, 2.22, facecolor="#F5F7FF", edgecolor="none", zorder=0)
    )
    ax.text(0.3, 4.99, "TRAINING & EVALUATION  ·  OFFLINE", fontsize=7.6,
            color=MUTED, fontweight="bold")
    ax.text(0.3, 2.22, "INFERENCE  ·  ONLINE", fontsize=7.6,
            color=MUTED, fontweight="bold")

    top = [
        ("SemEval-2014 XML", "3,041 train · 800 test\nrestaurant sentences", "blue"),
        ("Data preparation", "parse · validate · dedupe\ndata_loader.py", "blue"),
        ("Training", "TF-IDF · pLSA + TF-ICF\nLR / SVM / BiLSTM", "purple"),
        ("Save artefacts", "models/*.joblib\nmodels/lstm/*.pt", "indigo"),
        ("Evaluation", "held-out test · metrics\nerrors · figures", "amber"),
    ]
    n, gap = len(top), 0.2
    bw = (9.4 - gap * (n - 1)) / n
    y, h = 3.62, 1.15
    centres = []
    for i, (t, s, tone) in enumerate(top):
        x = 0.3 + i * (bw + gap)
        box(ax, x, y, bw, h, t, s, tone, title_size=10.2)
        centres.append(x + bw / 2)
        if i:
            arrow(ax, (centres[i - 1] + 0.02, y + h / 2), (x - 0.03, y + h / 2))

    # artefacts -> runtime loader
    ax.add_patch(
        FancyBboxPatch(
            (3.35, 2.52),
            3.3,
            0.32,
            boxstyle="round,pad=0.01,rounding_size=0.15",
            linewidth=1.0,
            edgecolor=INDIGO,
            facecolor="white",
            linestyle=(0, (2.5, 1.6)),
            zorder=3,
        )
    )
    ax.text(5.0, 2.68, "artefacts loaded once at startup (joblib / torch)",
            ha="center", va="center", fontsize=7.3, color=INDIGO, zorder=4)
    arrow(ax, (centres[3], y - 0.01), (centres[3], 2.86), color=INDIGO,
          ls=(0, (2.5, 1.6)))

    bottom = [
        ("Raw review", "free text", "green"),
        ("Streamlit UI", "six pages · app.py", "blue"),
        ("Engine", "clean · tokenise\nsrc/predict.py", "indigo"),
        ("Aspect detection", "supervised or\npLSA + TF-ICF", "purple"),
        ("Sentiment", "per aspect\nLR · SVM · BiLSTM", "purple"),
        ("Structured result", "aspect · label\nconfidence", "green"),
    ]
    n, gap = len(bottom), 0.18
    bw2 = (9.4 - gap * (n - 1)) / n
    y2, h2 = 0.72, 1.05
    centres2 = []
    for i, (t, s, tone) in enumerate(bottom):
        x = 0.3 + i * (bw2 + gap)
        box(ax, x, y2, bw2, h2, t, s, tone, title_size=9.6, sub_size=7.2)
        centres2.append(x + bw2 / 2)
        if i:
            arrow(ax, (centres2[i - 1] + 0.02, y2 + h2 / 2), (x - 0.03, y2 + h2 / 2))

    # loader feeds detection + sentiment
    arrow(ax, (3.9, 2.51), (centres2[3], y2 + h2 + 0.02), color=INDIGO,
          ls=(0, (2.5, 1.6)))
    arrow(ax, (6.1, 2.51), (centres2[4], y2 + h2 + 0.02), color=INDIGO,
          ls=(0, (2.5, 1.6)))

    ax.text(9.55, 0.32, "The official test split is used once, for final evaluation.",
            ha="right", va="center", fontsize=7.2, color=MUTED, style="italic")

    fig.savefig(os.path.join(ASSETS, "fig_architecture.png"),
                facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: NLP pipeline (snake layout)
# ---------------------------------------------------------------------------
def pipeline():
    W, H = 10, 3.55
    fig, ax = canvas(W, H)

    stages = [
        ("Load dataset", "official XML files\n→ two CSV views", "blue"),
        ("Validate", "dedupe · check labels\nand aspect columns", "blue"),
        ("Preprocess", "NFKC · contractions\nURL / HTML removal", "indigo"),
        ("Tokenise", "lowercase tokens\nword 1–2 grams", "indigo"),
        ("Features", "word + char TF-IDF\nsublinear tf", "purple"),
        ("Aspect detection", "5 probabilities →\nper-aspect thresholds", "purple"),
        ("Sentiment", "3 classes for every\ndetected aspect", "green"),
        ("Format output", "cards · probabilities\n· cleaned text", "green"),
    ]

    bw, gap = 2.16, 0.34
    x0 = 0.2
    y_top, y_bot, h = 2.10, 0.38, 1.02
    for i in range(4):
        x = x0 + i * (bw + gap)
        t, s, tone = stages[i]
        box(ax, x, y_top, bw, h, t, s, tone, num=i + 1, title_size=10.2)
        if i:
            arrow(ax, (x - gap + 0.02, y_top + h / 2), (x - 0.03, y_top + h / 2))
    for j in range(4):
        idx = 7 - j  # rightmost box is stage 5
        x = x0 + j * (bw + gap)
        t, s, tone = stages[idx]
        box(ax, x, y_bot, bw, h, t, s, tone, num=idx + 1, title_size=10.2)
        if j:
            x_prev = x0 + (j - 1) * (bw + gap)
            arrow(ax, (x, y_bot + h / 2), (x_prev + bw + 0.02, y_bot + h / 2))
    # connector from stage 4 down into stage 5
    xr = x0 + 3 * (bw + gap) + bw / 2
    arrow(ax, (xr, y_top - 0.01), (xr, y_bot + h + 0.02), color="#94A3B8", lw=1.3)

    ax.text(0.2, 3.38, "Serving path — the same stages run for every review, with no fitting",
            fontsize=7.6, color=MUTED, fontweight="bold")

    fig.savefig(os.path.join(ASSETS, "fig_pipeline.png"),
                facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Training vs prediction
# ---------------------------------------------------------------------------
def train_predict():
    W, H = 10, 3.15
    fig, ax = canvas(W, H)

    ax.text(2.45, 2.92, "TRAINING  ·  OFFLINE", ha="center", fontsize=7.8,
            color=AMBER, fontweight="bold")
    ax.text(7.55, 2.92, "PREDICTION  ·  ONLINE", ha="center", fontsize=7.8,
            color=GREEN, fontweight="bold")

    left = [
        "Official training split (3,035 sentences)",
        "Carve out a 15% development split",
        "Tune C, thresholds and BiLSTM epochs",
        "Refit on the full training data",
        "Save joblib bundles and .pt weights",
    ]
    right = [
        "Raw review text from the user",
        "Clean and tokenise (no stop-word removal)",
        "Load the saved artefacts once",
        "Aspect detection, then per-aspect sentiment",
        "Return a structured result dictionary",
    ]
    y = 2.38
    for i in range(5):
        box(ax, 0.25, y, 4.25, 0.37, left[i], "", "amber", num=i + 1,
            title_size=8.6, sub_size=7)
        box(ax, 5.5, y, 4.25, 0.37, right[i], "", "green", num=i + 1,
            title_size=8.6, sub_size=7)
        y -= 0.47

    # Shared artefacts sit between the two lanes.
    ax.plot([5.0, 5.0], [0.2, 2.55], linestyle=(0, (3, 2)), color="#C7D2FE",
            linewidth=1.0, zorder=1)
    ax.text(5.0, 1.35, "models/ + outputs/", rotation=90, ha="center",
            va="center", fontsize=7.2, color=INDIGO, fontweight="bold")

    fig.savefig(os.path.join(ASSETS, "fig_train_predict.png"),
                facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Frontend to backend
# ---------------------------------------------------------------------------
def frontend_backend():
    W, H = 10, 2.95
    fig, ax = canvas(W, H)

    boxes = [
        ("Browser", "Streamlit pages\nradio · buttons · inputs", "blue"),
        ("app.py", "session state\ncached loaders", "indigo"),
        ("Inference engine", "AspectSentimentEngine\nsrc/predict.py", "purple"),
        ("Saved artefacts", "joblib bundles\nPyTorch weights", "amber"),
    ]
    bw, gap = 2.12, 0.34
    y, h = 0.95, 1.35
    centres = []
    for i, (t, s, tone) in enumerate(boxes):
        x = 0.28 + i * (bw + gap)
        box(ax, x, y, bw, h, t, s, tone, title_size=10.0)
        centres.append(x + bw / 2)

    for i in range(3):
        arrow(ax, (centres[i] + bw / 2 - 0.02, y + h / 2 + 0.22),
              (centres[i + 1] - bw / 2 + 0.02, y + h / 2 + 0.22))

    # Return path beneath the blocks: engine -> app.py -> browser.
    ax.plot(
        [centres[2], centres[0]],
        [0.62, 0.62],
        color="#15803D",
        linewidth=1.15,
        zorder=2,
    )
    arrow(ax, (centres[1], 0.62), (centres[0] + 0.02, 0.62), color=GREEN, lw=1.15)
    ax.plot([centres[2], centres[2]], [y - 0.01, 0.62], color=GREEN, lw=1.15, zorder=2)
    ax.plot([centres[1], centres[1]], [y - 0.01, 0.62], color=GREEN, lw=1.15, zorder=2)
    ax.text(5.0, 0.36, "result dict → app.py renders cards, probabilities and tables",
            ha="center", va="center", fontsize=7.2, color=GREEN)

    ax.text(5.0, 2.62,
            "Requests flow left to right; all calls are in-process Python and no model is edited.",
            ha="center", fontsize=7.6, color=MUTED, style="italic")

    fig.savefig(os.path.join(ASSETS, "fig_frontend_backend.png"),
                facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Worked example
# ---------------------------------------------------------------------------
def example_flow():
    W, H = 10, 2.0
    fig, ax = canvas(W, H)

    cards = [
        ("Raw review", "“The pizza was delicious and the\nservice was quick, but the restaurant\nwas noisy and the prices were too high.”", "slate"),
        ("Clean + tokens", "20 lowercase tokens;\npunctuation removed, negation kept\ne.g. the pizza was delicious …", "blue"),
        ("Aspect probabilities", "Food .871   Service .966\nPrice .785   Ambience .589\nMisc .000 · thresholds .30–.65", "purple"),
        ("Per-aspect sentiment", "Food → positive (.875)\nService → positive (.764)\nPrice → negative (.537)", "green"),
        ("Displayed result", "3 aspects analysed\noverall summary positive (.690)\nambience below its .65 threshold", "green"),
    ]
    bw, gap = 1.84, 0.12
    y, h = 0.15, 1.58
    for i, card in enumerate(cards):
        t, s, tone = card[0], card[1], card[2]
        x = 0.25 + i * (bw + gap)
        box(ax, x, y, bw, h, t, s, tone, title_size=9.6, sub_size=7.4,
            sub_offset=0.34)
        if i:
            arrow(ax, (x - gap + 0.01, y + h / 2), (x - 0.02, y + h / 2))

    fig.savefig(os.path.join(ASSETS, "fig_example_flow.png"),
                facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation charts (values read from metrics.json)
# ---------------------------------------------------------------------------
def _clean_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color="#E2E8F0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def charts():
    with open(os.path.join(ROOT, "outputs", "reports", "metrics.json")) as handle:
        metrics = json.load(handle)

    names = ["majority", "tfidf_logreg", "tfidf_svm", "bilstm"]
    labels = ["Majority", "TF-IDF\n+ LR", "TF-IDF\n+ SVM", "BiLSTM"]
    colors = ["#94A3B8", INDIGO, PURPLE, "#0D9488"]

    # 1. Aggregate sentiment comparison
    fig, ax = plt.subplots(figsize=(3.5, 2.25), dpi=260)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.84, bottom=0.19)
    width = 0.26
    xs = range(len(names))
    for offset, (key, colour, label) in enumerate(
        [("accuracy", "#94A3B8", "Accuracy"),
         ("macro_f1", INDIGO, "Macro-F1"),
         ("weighted_f1", PURPLE, "Weighted-F1")]
    ):
        values = [metrics["sentiment"][n][key] for n in names]
        positions = [x + (offset - 1) * width for x in xs]
        bars = ax.bar(positions, values, width=width, color=colour, label=label,
                      zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=6.0, color="#334155")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylim(0, 0.95)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("score", fontsize=7.4)
    ax.legend(fontsize=6.6, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), handlelength=1.0, columnspacing=0.9)
    _clean_axes(ax)
    fig.savefig(os.path.join(ASSETS, "chart_model_scores.png"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # 2. Confusion matrix for the best model
    cm = metrics["sentiment"]["tfidf_logreg"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(3.05, 2.25), dpi=260)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.86, bottom=0.20)
    im = ax.imshow(cm, cmap="Purples", vmin=0, vmax=320)
    labels_cm = ["Neg", "Neu", "Pos"]
    ax.set_xticks(range(3), labels_cm, fontsize=7.4)
    ax.set_yticks(range(3), labels_cm, fontsize=7.4)
    ax.set_xlabel("Predicted", fontsize=7.6, labelpad=2)
    ax.set_ylabel("Gold", fontsize=7.6, labelpad=2)
    for i in range(3):
        for j in range(3):
            value = cm[i][j]
            colour = "white" if value > 380 else NAVY
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8.4,
                    color=colour, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("TF-IDF + LR · 973 test rows", fontsize=7.8, color=MUTED,
                 pad=5, loc="left")
    fig.savefig(os.path.join(ASSETS, "chart_confusion.png"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # 3. Per-aspect sentiment macro-F1
    aspects = ["food", "service", "price", "ambience", "miscellaneous"]
    aspect_labels = ["Food", "Service", "Price", "Ambience", "Overall /\nmisc."]
    fig, ax = plt.subplots(figsize=(3.5, 2.27), dpi=260)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.22)
    width = 0.26
    for offset, (model, colour, label) in enumerate(
        [("tfidf_logreg", INDIGO, "LR"), ("tfidf_svm", PURPLE, "SVM"),
         ("bilstm", "#0D9488", "BiLSTM")]
    ):
        values = [metrics["sentiment"][model]["per_aspect"][a]["macro_f1"] for a in aspects]
        positions = [x + (offset - 1) * width for x in range(len(aspects))]
        ax.bar(positions, values, width=width, color=colour, label=label, zorder=3)
    ax.set_xticks(range(len(aspects)))
    ax.set_xticklabels(aspect_labels, fontsize=6.8)
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("macro-F1", fontsize=7.4)
    ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.17), handlelength=1.0, columnspacing=1.0)
    _clean_axes(ax)
    fig.savefig(os.path.join(ASSETS, "chart_sentiment_aspect.png"),
                facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # 4. Per-aspect aspect-detection F1
    fig, ax = plt.subplots(figsize=(3.5, 2.27), dpi=260)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.22)
    width = 0.36
    for offset, (model, colour, label) in enumerate(
        [("tfidf_logreg", INDIGO, "Supervised LR"),
         ("lexicon_plsa_tficf", PURPLE, "pLSA + TF-ICF")]
    ):
        values = [metrics["aspect_detection"][model]["per_aspect"][a]["f1"] for a in aspects]
        positions = [x + (offset - 0.5) * width for x in range(len(aspects))]
        bars = ax.bar(positions, values, width=width, color=colour, label=label, zorder=3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom",
                    fontsize=5.7, color="#475569")
    ax.set_xticks(range(len(aspects)))
    ax.set_xticklabels(aspect_labels, fontsize=6.8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("F1", fontsize=7.4)
    ax.legend(fontsize=6.7, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.17), handlelength=1.0, columnspacing=1.0)
    _clean_axes(ax)
    fig.savefig(os.path.join(ASSETS, "chart_detection.png"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    pipeline()
    train_predict()
    frontend_backend()
    example_flow()
    charts()
    print("Assets written to", ASSETS)
