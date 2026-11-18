"""ABSA Studio — a polished Streamlit frontend for the ABSA NLP project.

The application is a presentation layer only: every prediction, metric and
dataset figure is produced by the existing backend under ``src/`` and the
saved artefacts under ``models/``.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import os
import sys
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from frontend.components import (
    badge,
    empty_state,
    hero,
    highlight_card,
    info_card,
    kpi_card,
    page_header,
    pipeline_steps,
    probability_bar,
    result_card,
    section_header,
    sidebar_brand,
    sidebar_footer,
    sidebar_project,
    sidebar_status,
    table_html,
    timeline,
    verdict_card,
)
from frontend.styles import GLOBAL_CSS
from src.config import (
    ASPECT_DESCRIPTIONS,
    ASPECT_DISPLAY,
    ASPECTS,
    FIGURES_DIR,
    LSTM_META_PATH,
    METRICS_PATH,
    REPORTS_DIR,
    SENTIMENT_LABELS,
)
from src.data_loader import load_aspect_sentiment, load_dataset_stats, load_sentences
from src.predict import AspectSentimentEngine

# ---------------------------------------------------------------------------
# Page configuration and design system
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ABSA Studio | Restaurant Review Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

PAGES = ["Overview", "Analyse Review", "Model Performance", "Dataset", "Methodology", "About"]
NAV_ICONS = {
    "Overview": "◈",
    "Analyse Review": "◎",
    "Model Performance": "◫",
    "Dataset": "▤",
    "Methodology": "◇",
    "About": "ⓘ",
}
SENTIMENT_COLORS = {
    "positive": "#16A34A",
    "negative": "#DC2626",
    "neutral": "#D97706",
}
MODEL_DISPLAY = {
    "majority": "Majority class",
    "tfidf_logreg": "TF-IDF + LR",
    "tfidf_svm": "TF-IDF + SVM",
    "bilstm": "BiLSTM",
    "lexicon_plsa_tficf": "pLSA + TF-ICF lexicon",
}


# ---------------------------------------------------------------------------
# Cached data / model access
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_engine() -> AspectSentimentEngine:
    return AspectSentimentEngine()


@st.cache_data(show_spinner=False)
def get_metrics() -> Dict:
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def get_stats() -> Dict:
    try:
        return load_dataset_stats()
    except FileNotFoundError:
        return {}


@st.cache_data(show_spinner=False)
def get_test_sentences() -> pd.DataFrame:
    try:
        return load_sentences("test")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_test_sentiment() -> pd.DataFrame:
    try:
        return load_aspect_sentiment("test")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_error_analysis() -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, "error_analysis.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_classification_report() -> str:
    path = os.path.join(REPORTS_DIR, "classification_report.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return ""


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
def figure_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, name)


def show_figure(name: str, caption: str | None = None) -> None:
    path = figure_path(name)
    if not os.path.exists(path):
        st.info(f"Figure not found: {name}. Run `python -m scripts.generate_figures`.")
        return
    with st.container(border=True):
        st.image(path, width="stretch")
        if caption:
            st.caption(caption)


def render_metric_cards(items: List[tuple]) -> None:
    cols = st.columns(len(items), gap="medium")
    for col, (icon_name, label, value, subtitle, tone) in zip(cols, items):
        with col:
            st.markdown(
                kpi_card(icon_name, label, value, subtitle, tone),
                unsafe_allow_html=True,
            )


def render_table(frame: pd.DataFrame) -> None:
    st.markdown(table_html(frame), unsafe_allow_html=True)


def aspect_card_html(item: Dict) -> str:
    sentiment = item["sentiment"]
    color = SENTIMENT_COLORS[sentiment]
    return (
        f'<div class="aspect-card {sentiment}">'
        f'<div class="aspect-head"><div>'
        f'<div class="aspect-name">{html.escape(item["display"])}</div>'
        f'<div class="aspect-sub">Aspect presence · {item["aspect_probability"]:.1%}</div>'
        f"</div>{badge(sentiment)}</div>"
        f'{probability_bar(item["aspect_probability"], "#4338CA")}'
        f'<div class="aspect-sub" style="margin-top:11px;">Sentiment confidence · '
        f'{item["confidence"]:.1%}</div>'
        f'{probability_bar(item["confidence"], color, 11)}'
        "</div>"
    )


def sentiment_chart(probabilities: Dict[str, float]) -> alt.Chart:
    frame = pd.DataFrame(
        {
            "Sentiment": [label.title() for label in SENTIMENT_LABELS],
            "Probability": [probabilities[label] for label in SENTIMENT_LABELS],
        }
    )
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Probability",
                axis=alt.Axis(format=".0%", grid=True),
            ),
            y=alt.Y(
                "Sentiment:N",
                sort=["Negative", "Neutral", "Positive"],
                title=None,
            ),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["Negative", "Neutral", "Positive"],
                    range=["#DC2626", "#D97706", "#16A34A"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Sentiment:N", title="Sentiment"),
                alt.Tooltip("Probability:Q", format=".3f", title="Probability"),
            ],
        )
        .properties(height=118)
        .configure_view(stroke=None)
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
stats = get_stats()
train_sentences = stats.get("splits", {}).get("train", {}).get("sentences") if stats else None
test_sentences = stats.get("splits", {}).get("test", {}).get("sentences") if stats else None

try:
    _engine = get_engine()
    engine_ok = True
    engine_status = f"Models loaded · {len(_engine.available_sentiment_models)} sentiment models"
except Exception as exc:  # pragma: no cover - only when artefacts are missing
    engine_ok = False
    engine_status = "Models unavailable"

with st.sidebar:
    st.markdown(sidebar_brand(), unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Workspace</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
        format_func=lambda name: f"{NAV_ICONS.get(name, '•')}  {name}",
    )
    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Live system</div>', unsafe_allow_html=True)
    st.markdown(sidebar_status(engine_ok, engine_status), unsafe_allow_html=True)
    st.markdown(sidebar_project(train_sentences, test_sentences), unsafe_allow_html=True)
    st.markdown(sidebar_footer(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    metrics = get_metrics()
    aspect_micro = (
        metrics.get("aspect_detection", {}).get("tfidf_logreg", {}).get("micro_f1")
        if metrics
        else None
    )
    sentiment_macro = (
        metrics.get("sentiment", {}).get("tfidf_logreg", {}).get("macro_f1")
        if metrics
        else None
    )
    total_reviews = (
        train_sentences + test_sentences
        if train_sentences is not None and test_sentences is not None
        else None
    )

    st.markdown(
        hero(
            "Natural Language Processing · Coursework Project",
            "Aspect-Based Sentiment Analysis of Restaurant Reviews",
            "An NLP-powered analytics platform that identifies restaurant review aspects "
            "and understands customer sentiment at a granular level — moving beyond a "
            "single star rating to reveal what diners actually praise or criticise.",
            [
                "Topic Modelling · pLSA",
                "TF-IDF features",
                "TF-ICF lexicon expansion",
                "Aspect detection",
                "Sentiment classification",
                "GloVe BiLSTM",
            ],
        ),
        unsafe_allow_html=True,
    )

    render_metric_cards(
        [
            (
                "message",
                "Reviews analysed",
                f"{total_reviews:,}" if total_reviews is not None else "—",
                "Official train + test splits",
                "indigo",
            ),
            (
                "layers",
                "Aspect categories",
                str(len(ASPECTS)),
                "Food · service · price · ambience · misc.",
                "purple",
            ),
            (
                "target",
                "Aspect detection",
                f"{aspect_micro:.3f}" if aspect_micro is not None else "—",
                "Micro-F1 · held-out test set",
                "blue",
            ),
            (
                "activity",
                "Sentiment macro-F1",
                f"{sentiment_macro:.3f}" if sentiment_macro is not None else "—",
                "Best classical model · TF-IDF + LR",
                "green",
            ),
        ]
    )

    st.write("")
    st.markdown(
        section_header(
            "Why it matters",
            "A review is not one opinion — it is a collection of opinions",
            "Aspect-level analysis turns free text into actionable feedback.",
        ),
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            info_card(
                "alert",
                "The problem",
                "Star ratings collapse an entire restaurant visit into one number. A restaurant "
                "can serve excellent food while the service is rude and the prices are unfair. "
                "Owners and analysts need to know <b>which part</b> of the experience is driving "
                "the rating, yet manually reading thousands of reviews is not scalable.",
                "problem",
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            info_card(
                "lightbulb",
                "The solution",
                "This system reads a free-text review, detects the aspects it discusses — "
                "<b>food, service, price, ambience and overall experience</b> — and predicts the "
                "polarity of each aspect independently. It combines a transparent topic model "
                "(pLSA) and TF-ICF lexicon induction with supervised TF-IDF classifiers and a "
                "GloVe-initialised BiLSTM.",
                "solution",
            ),
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        section_header(
            "System architecture",
            "A transparent, end-to-end NLP pipeline",
            "Every stage below is implemented in the repository and evaluated on real data.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        pipeline_steps(
            [
                ("database", "SemEval dataset", "3,041 training + 800 test restaurant reviews.", "indigo"),
                ("file", "Data loading", "XML annotations parsed into two validated task views.", "blue"),
                ("cpu", "Preprocessing", "Normalise, expand contractions, preserve negation, tokenise.", "purple"),
                ("layers", "Topic modelling", "20 latent pLSA topics discovered with EM.", "amber"),
                ("grid", "Lexicon expansion", "TF-ICF transfers cluster-specific terms to aspects.", "purple"),
                ("target", "Aspect detection", "TF-IDF word + character features with one-vs-rest LR.", "blue"),
                ("activity", "Sentiment models", "Per-aspect LR, calibrated SVM and GloVe BiLSTM.", "green"),
                ("chart", "Evaluation & app", "Test metrics, error analysis and live Streamlit inference.", "red"),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        section_header(
            "Capabilities",
            "What the prototype demonstrates",
            "A complete NLP workflow, not only a sentiment classifier.",
        ),
        unsafe_allow_html=True,
    )
    highlights = [
        ("target", "Aspect-level detection", "Finds the five SemEval aspect categories, including mixed-opinion reviews.", "indigo"),
        ("activity", "Fine-grained sentiment", "Predicts negative, neutral or positive for each aspect independently.", "green"),
        ("layers", "Topic modelling", "pLSA reveals latent themes; TF-ICF turns them into an interpretable aspect lexicon.", "purple"),
        ("chart", "Rigorous evaluation", "Macro-F1, Hamming loss, subset accuracy, confusion matrices and error analysis.", "blue"),
        ("message", "Interactive analysis", "Analyse a review live, compare models and inspect confidence scores.", "amber"),
        ("cpu", "End-to-end engineering", "Reproducible training, saved artefacts, automated tests and a polished UI.", "red"),
    ]
    for start in range(0, len(highlights), 3):
        cols = st.columns(3, gap="medium")
        for col, (icon_name, title, body, tone) in zip(cols, highlights[start : start + 3]):
            with col:
                st.markdown(
                    highlight_card(icon_name, title, body, tone),
                    unsafe_allow_html=True,
                )

    st.write("")
    st.markdown(
        section_header(
            "Live demo",
            "Try the trained model in seconds",
            "This quick demo uses the same saved artefacts as the full review workspace.",
        ),
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">Quick review analysis</div>'
            '<p class="panel-sub">Enter a review or keep the multi-aspect example below.</p>',
            unsafe_allow_html=True,
        )
        quick_default = (
            "The pasta was absolutely delicious and the staff were friendly, "
            "but the music was too loud and the prices are a bit high."
        )
        quick_text = st.text_area(
            "Review text",
            value=quick_default,
            height=105,
            key="overview_input",
        )
        run_quick = st.button("Analyse review", type="primary", key="overview_run")
        if run_quick:
            try:
                result = get_engine().predict(
                    quick_text,
                    aspect_method="supervised",
                    sentiment_model="tfidf_logreg",
                )
                detected = ", ".join(
                    ASPECT_DISPLAY[item] for item in result["detected_aspects"]
                ) or "none above threshold"
                st.markdown(
                    f"**Detected aspects:** {html.escape(detected)}"
                )
                cols = st.columns(len(result["aspects"]), gap="small")
                for col, item in zip(cols, result["aspects"]):
                    with col:
                        st.markdown(result_card(item), unsafe_allow_html=True)
                st.caption(
                    "Open **Analyse Review** for model selection, thresholds, top-k analysis "
                    "and full class probabilities."
                )
            except Exception as exc:
                st.error(str(exc))


# ---------------------------------------------------------------------------
# Page: Analyse Review
# ---------------------------------------------------------------------------
elif page == "Analyse Review":
    st.markdown(
        page_header(
            "Review workspace",
            "Analyse a review",
            "Enter a restaurant review. The system detects the aspects it discusses and "
            "predicts the sentiment for each aspect independently.",
        ),
        unsafe_allow_html=True,
    )

    test_frame = get_test_sentences()
    examples: List[str] = []
    if not test_frame.empty:
        multi = test_frame[test_frame["num_aspects"] >= 2]
        pool = multi if len(multi) >= 6 else test_frame
        examples = pool.sample(n=min(6, len(pool)), random_state=7)["text"].tolist()

    if "review_input" not in st.session_state:
        st.session_state["review_input"] = (
            "The pizza was delicious and the service was quick, but the restaurant was noisy "
            "and the prices were too high."
        )
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None

    try:
        engine_for_list = get_engine()
        available_models = engine_for_list.available_sentiment_models
    except Exception as exc:
        st.error(f"Models are not available: {exc}")
        st.stop()

    left, right = st.columns([1.02, 0.98], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Review input</div>'
                '<p class="panel-sub">Use a real held-out example or write your own review.</p>',
                unsafe_allow_html=True,
            )
            if examples:
                st.caption("Load an example from the held-out test set")
                for start in range(0, len(examples), 3):
                    example_cols = st.columns(3, gap="small")
                    for offset, example in enumerate(examples[start : start + 3]):
                        idx = start + offset
                        with example_cols[offset]:
                            if st.button(
                                f"Example {idx + 1}",
                                key=f"example_{idx}",
                                width="stretch",
                            ):
                                st.session_state["review_input"] = example
                                st.session_state["analysis_result"] = None
                                st.rerun()

            text = st.text_area(
                "Review text",
                height=190,
                key="review_input",
                placeholder="e.g. Great food and friendly staff, but the room was too loud...",
            )

            with st.expander("Model settings", expanded=True):
                aspect_method = st.selectbox(
                    "Aspect detection method",
                    ["supervised", "lexicon"],
                    format_func=lambda value: {
                        "supervised": "Supervised · TF-IDF + LR",
                        "lexicon": "Unsupervised · pLSA + TF-ICF",
                    }[value],
                    help="The unsupervised baseline is weaker but shows what pLSA + TF-ICF learned without labels.",
                )
                sentiment_model = st.selectbox(
                    "Sentiment classifier",
                    available_models,
                    index=(
                        available_models.index("tfidf_logreg")
                        if "tfidf_logreg" in available_models
                        else 0
                    ),
                    format_func=lambda value: {
                        "bilstm": "BiLSTM · GloVe-initialised",
                        "tfidf_logreg": "TF-IDF + LR · best macro-F1",
                        "tfidf_svm": "TF-IDF + SVM · calibrated",
                    }.get(value, value),
                )
                threshold = st.slider(
                    "Minimum aspect confidence",
                    min_value=0.0,
                    max_value=0.9,
                    value=0.0,
                    step=0.05,
                    help="Per-aspect tuned thresholds are always applied; this slider can only make them stricter.",
                )
                top_k = st.slider(
                    "Always analyse the top-k aspects",
                    min_value=0,
                    max_value=5,
                    value=0,
                    step=1,
                    help="Set to 0 to analyse only aspects that pass their tuned threshold.",
                )

            run = st.button("Analyse review", type="primary", width="stretch")
            st.caption(
                "Input validation rejects empty reviews and reviews with fewer than two words."
            )

            if run:
                try:
                    with st.spinner("Running aspect detection and sentiment classification..."):
                        result = get_engine().predict(
                            text,
                            aspect_method=aspect_method,
                            sentiment_model=sentiment_model,
                            threshold=threshold if threshold > 0 else None,
                            top_k=top_k if top_k > 0 else None,
                        )
                    st.session_state["analysis_result"] = result
                except ValueError as exc:
                    st.warning(str(exc))
                    st.session_state["analysis_result"] = None
                except Exception as exc:  # pragma: no cover - defensive
                    st.error(f"Prediction failed: {exc}")
                    st.session_state["analysis_result"] = None

    with right:
        st.markdown('<div class="section-kicker">Result</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model output</div>', unsafe_allow_html=True)
        result = st.session_state.get("analysis_result")

        if result is None:
            st.markdown(
                empty_state(
                    "search",
                    "No analysis yet",
                    "Enter a review and press Analyse review. Each detected aspect, its "
                    "sentiment and the model's confidence will appear here.",
                ),
                unsafe_allow_html=True,
            )
        else:
            if result.get("low_confidence"):
                st.warning(
                    "No aspect passed the detection threshold. The most likely aspect is "
                    "shown below, but treat the result as low confidence."
                )

            overall = result["overall"]
            st.markdown(
                verdict_card(
                    overall["label"],
                    overall["confidence"],
                    "Weighted by aspect-presence probability. This is a display summary, "
                    "not a separately trained classifier.",
                ),
                unsafe_allow_html=True,
            )

            for item in result["aspects"]:
                st.markdown(aspect_card_html(item), unsafe_allow_html=True)
                with st.expander(f"Class probabilities · {item['display']}"):
                    st.altair_chart(
                        sentiment_chart(item["probabilities"]),
                        width="stretch",
                    )

            with st.expander("Aspect-presence probabilities · all five aspects"):
                aspect_frame = pd.DataFrame(
                    {
                        "Aspect": [ASPECT_DISPLAY[aspect] for aspect in ASPECTS],
                        "Probability": [
                            round(result["aspect_probabilities"].get(aspect, 0.0), 4)
                            for aspect in ASPECTS
                        ],
                        "Detected": [
                            "Yes" if aspect in result["detected_aspects"] else "No"
                            for aspect in ASPECTS
                        ],
                    }
                ).sort_values("Probability", ascending=False)
                render_table(aspect_frame)
                st.bar_chart(
                    aspect_frame.set_index("Aspect")["Probability"],
                    height=240,
                    color="#4338CA",
                )
                st.caption(
                    "Tuned thresholds · "
                    + " · ".join(
                        f"{ASPECT_DISPLAY[aspect]} {result['thresholds'][aspect]:.2f}"
                        for aspect in ASPECTS
                    )
                )

            with st.expander("Preprocessing and model configuration"):
                st.markdown("**Cleaned text**")
                st.code(result["cleaned_text"], language=None)
                st.markdown("**Configuration used for this prediction**")
                st.json(
                    {
                        "aspect_method": result["aspect_method"],
                        "sentiment_model": result["sentiment_model"],
                        "tuning_thresholds": result["thresholds"],
                    }
                )


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    metrics = get_metrics()
    if not metrics:
        st.error(
            "Metrics not found. Run `python -m scripts.evaluate_all` and "
            "`python -m scripts.generate_figures`."
        )
        st.stop()

    st.markdown(
        page_header(
            "Evaluation dashboard",
            "Model performance",
            "Actual results on the official SemEval-2014 test split. No test data was used "
            "for training or model selection.",
        ),
        unsafe_allow_html=True,
    )

    test_info = metrics.get("test_set", {})
    st.caption(
        f"{test_info.get('sentences', '—')} test sentences · "
        f"{test_info.get('aspect_sentiment_rows', '—')} aspect-sentiment annotations"
    )

    sentiment = metrics["sentiment"]
    best_name = max(
        (name for name in sentiment if name != "majority"),
        key=lambda name: sentiment[name]["macro_f1"],
    )
    best = sentiment[best_name]
    render_metric_cards(
        [
            ("zap", "Best model", MODEL_DISPLAY.get(best_name, best_name), "Selected by macro-F1", "indigo"),
            ("check", "Accuracy", f"{best['accuracy']:.3f}", "Three-class accuracy", "blue"),
            ("target", "Macro-F1", f"{best['macro_f1']:.3f}", "Equal weight per class", "green"),
            ("trending", "Weighted-F1", f"{best['weighted_f1']:.3f}", "Support-weighted", "purple"),
        ]
    )

    st.write("")
    tab_sentiment, tab_aspect, tab_errors = st.tabs(
        ["Sentiment classification", "Aspect detection", "Error analysis"]
    )

    with tab_sentiment:
        st.markdown(
            section_header(
                "Model comparison",
                "Sentiment classification results",
                "Macro-F1 is emphasised because the neutral class is rare.",
            ),
            unsafe_allow_html=True,
        )
        comparison = pd.DataFrame(
            [
                {
                    "Model": MODEL_DISPLAY.get(name, name),
                    "Accuracy": round(value["accuracy"], 4),
                    "Macro-F1": round(value["macro_f1"], 4),
                    "Weighted-F1": round(value["weighted_f1"], 4),
                    "N": value["n_samples"],
                }
                for name, value in sentiment.items()
            ]
        ).sort_values("Macro-F1", ascending=False)
        render_table(comparison)

        show_figure(
            "model_comparison.png",
            "Accuracy, macro-F1 and weighted-F1 across sentiment models.",
        )
        show_figure(
            "confusion_matrices.png",
            "Confusion matrices for the majority baseline, TF-IDF models and BiLSTM.",
        )
        show_figure(
            "per_aspect_f1.png",
            "Per-aspect sentiment macro-F1 for the strongest models.",
        )
        show_figure(
            "bilstm_training_curves.png",
            "Development macro-F1 by epoch for the five aspect-specific BiLSTMs.",
        )

        report = get_classification_report()
        if report:
            with st.expander("Classification report · best model"):
                st.code(report, language=None)

    with tab_aspect:
        st.markdown(
            section_header(
                "Aspect detection",
                "Supervised detection vs. the unsupervised baseline",
                "The supervised model learns pragmatic cues that a lexicon cannot encode.",
            ),
            unsafe_allow_html=True,
        )
        detection = metrics["aspect_detection"]
        detection_rows = []
        for name, value in detection.items():
            detection_rows.append(
                {
                    "Model": MODEL_DISPLAY.get(name, name),
                    "Micro-F1": round(value["micro_f1"], 4),
                    "Macro-F1": round(value["macro_f1"], 4),
                    "Hamming loss": round(value["hamming_loss"], 4),
                    "Subset accuracy": round(value["subset_accuracy"], 4),
                }
            )
        render_table(pd.DataFrame(detection_rows))

        show_figure(
            "aspect_detection_comparison.png",
            "Aggregate aspect-detection quality for both approaches.",
        )
        show_figure(
            "aspect_detection_per_aspect.png",
            "Per-aspect detection quality across the five categories.",
        )

        st.markdown(
            section_header(
                "Per-aspect detail",
                "Where detection succeeds and where it struggles",
            ),
            unsafe_allow_html=True,
        )
        detail_rows = []
        for aspect in ASPECTS:
            detail_rows.append(
                {
                    "Aspect": ASPECT_DISPLAY[aspect],
                    "Precision": round(
                        detection["tfidf_logreg"]["per_aspect"][aspect]["precision"], 4
                    ),
                    "Recall": round(
                        detection["tfidf_logreg"]["per_aspect"][aspect]["recall"], 4
                    ),
                    "F1": round(
                        detection["tfidf_logreg"]["per_aspect"][aspect]["f1"], 4
                    ),
                    "Lexicon F1": round(
                        detection["lexicon_plsa_tficf"]["per_aspect"][aspect]["f1"], 4
                    ),
                }
            )
        render_table(pd.DataFrame(detail_rows))

        with st.expander("Why the supervised detector wins"):
            st.write(
                "The lexicon baseline is fully unsupervised: pLSA discovers 20 latent topics, "
                "TF-ICF selects cluster-specific terms, and topics are aligned to aspect seeds "
                "by cosine similarity. It is interpretable and needs no labels, but it cannot "
                "learn that phrases such as 'we will be back' signal a general judgement without "
                "lexical overlap. The supervised one-vs-rest classifier learns these cues from "
                "3,035 labelled sentences, which explains the large gap in macro-F1."
            )

    with tab_errors:
        st.markdown(
            section_header(
                "Qualitative analysis",
                "The most confident mistakes",
                "High-confidence errors reveal annotation ambiguity and aspect-specific context.",
            ),
            unsafe_allow_html=True,
        )
        show_figure(
            "error_confidence.png",
            "Confidence distribution for correct predictions and errors.",
        )
        errors = get_error_analysis()
        if not errors.empty:
            view = errors.head(15)[
                ["text", "aspect", "gold", "predicted", "confidence"]
            ].copy()
            view["aspect"] = view["aspect"].map(
                lambda value: ASPECT_DISPLAY.get(value, value)
            )
            view["gold"] = view["gold"].str.title()
            view["predicted"] = view["predicted"].str.title()
            view["confidence"] = view["confidence"].map(lambda value: f"{value:.3f}")
            view = view.rename(
                columns={
                    "text": "Review",
                    "aspect": "Aspect",
                    "gold": "Gold",
                    "predicted": "Predicted",
                    "confidence": "Confidence",
                }
            )
            render_table(view)
        else:
            st.info("No error analysis file found. Run `python -m scripts.evaluate_all`.")


# ---------------------------------------------------------------------------
# Page: Dataset
# ---------------------------------------------------------------------------
elif page == "Dataset":
    stats = get_stats()
    if not stats:
        st.error("Dataset statistics not found. Run `python -m scripts.prepare_data`.")
        st.stop()

    st.markdown(
        page_header(
            "Data foundation",
            "Dataset",
            "The official SemEval-2014 Task 4 restaurant benchmark, parsed into "
            "sentence-level and aspect–sentiment views.",
        ),
        unsafe_allow_html=True,
    )

    train = stats["splits"]["train"]
    test = stats["splits"]["test"]
    render_metric_cards(
        [
            ("file", "Training sentences", f"{train['sentences']:,}", "Official training split", "indigo"),
            ("check", "Test sentences", f"{test['sentences']:,}", "Official gold test split", "blue"),
            ("layers", "Training annotations", f"{train['aspect_annotations']:,}", "Aspect-category labels", "purple"),
            ("activity", "Sentiment rows", f"{test['aspect_annotations']:,}", "Test aspect annotations", "green"),
        ]
    )

    st.write("")
    st.markdown(
        info_card(
            "database",
            "Source and provenance",
            "<b>SemEval-2014 Task 4: Aspect Based Sentiment Analysis</b> — restaurant domain "
            "(Pontiki et al., 2014). Each sentence is annotated with aspect categories "
            "(<i>food, service, price, ambience, anecdotes/miscellaneous</i>) and a polarity "
            "for each category. The official training and gold test splits are used unchanged; "
            "six exact duplicate training texts were removed from the training split only. "
            "The <code>conflict</code> polarity is excluded from the three-class sentiment task "
            "because it is not a coherent polarity class.",
            "solution",
        ),
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        section_header(
            "Structure",
            "How the benchmark is organised",
            "The loader exposes two task views built from the same source annotations.",
        ),
        unsafe_allow_html=True,
    )
    structure = pd.DataFrame(
        [
            (
                "Sentence view",
                "text · aspect labels · num_aspects",
                "One row per review sentence, used for multi-label aspect detection.",
            ),
            (
                "Aspect–sentiment view",
                "text · aspect · label · label_id",
                "One row per annotated aspect, used for per-aspect sentiment classification.",
            ),
        ],
        columns=["Task view", "Available fields", "Purpose"],
    )
    render_table(structure)

    multi_train = sum(
        count
        for key, count in train["aspects_per_sentence"].items()
        if key != "1"
    )
    multi_test = sum(
        count
        for key, count in test["aspects_per_sentence"].items()
        if key != "1"
    )
    profile = pd.DataFrame(
        [
            (
                "Mean tokens per sentence",
                f"{train['mean_tokens']:.2f}",
                f"{test['mean_tokens']:.2f}",
            ),
            (
                "Longest sentence (tokens)",
                f"{train['max_tokens']}",
                f"{test['max_tokens']}",
            ),
            (
                "Sentences with 2+ aspects",
                f"{multi_train:,}",
                f"{multi_test:,}",
            ),
            (
                "Duplicate texts removed",
                f"{train['duplicate_texts']}",
                f"{test['duplicate_texts']}",
            ),
        ],
        columns=["Corpus statistic", "Train", "Test"],
    )
    render_table(profile)

    st.write("")
    show_figure(
        "dataset_distribution.png",
        "Aspect and polarity distributions across the official training and test splits.",
    )
    show_figure(
        "aspect_sentiment_distribution.png",
        "Aspect-by-sentiment heatmap for the training split.",
    )

    st.markdown(
        section_header(
            "Class distribution",
            "Sentiment labels by aspect",
            "Training annotations after excluding conflict polarity.",
        ),
        unsafe_allow_html=True,
    )
    dist_rows = []
    for aspect in ASPECTS:
        row = {"Aspect": ASPECT_DISPLAY[aspect]}
        for label in SENTIMENT_LABELS:
            row[label.title()] = train["aspect_sentiment_distribution"].get(
                f"{aspect}/{label}", 0
            )
        dist_rows.append(row)
    render_table(pd.DataFrame(dist_rows))

    st.markdown(
        section_header(
            "Sample records",
            "What the model sees",
            "Random examples from the held-out test set.",
        ),
        unsafe_allow_html=True,
    )
    test_sentiment = get_test_sentiment()
    if not test_sentiment.empty:
        sample = test_sentiment.sample(n=min(10, len(test_sentiment)), random_state=11)
        view = sample[["text", "aspect", "label"]].copy()
        view["aspect"] = view["aspect"].map(
            lambda value: ASPECT_DISPLAY.get(value, value)
        )
        view["label"] = view["label"].str.title()
        render_table(
            view.rename(
                columns={"text": "Review", "aspect": "Aspect", "label": "Polarity"}
            )
        )

    with st.expander("Dataset statistics · JSON"):
        st.json(stats)

    with st.expander("Preprocessing examples"):
        sentence_frame = get_test_sentences()
        if not sentence_frame.empty:
            examples = sentence_frame.sample(
                n=min(4, len(sentence_frame)), random_state=3
            )["text"].tolist()
            from src.preprocessing import clean_text

            for example in examples:
                with st.container(border=True):
                    st.markdown(f"**Raw:** {example}")
                    st.markdown(f"**Cleaned:** `{clean_text(example)}`")


# ---------------------------------------------------------------------------
# Page: Methodology
# ---------------------------------------------------------------------------
elif page == "Methodology":
    st.markdown(
        page_header(
            "Technical approach",
            "NLP methodology",
            "Every stage below is implemented in the repository; results and figures are "
            "generated by the actual pipeline.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        section_header(
            "Workflow",
            "From raw XML to live predictions",
            "The same data and feature pipeline feeds both the unsupervised baseline and the supervised models.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        timeline(
            [
                ("Data collection", "The official SemEval-2014 Task 4 restaurant XML files are parsed and validated into two task views: sentence-level multi-label aspects and aspect–sentiment pairs.", "database"),
                ("Text preprocessing", "Unicode NFKC normalisation, HTML/URL removal, contraction expansion that preserves negation, lowercasing, emphasis handling and regex tokenisation.", "cpu"),
                ("Feature extraction", "Word 1–2 gram TF-IDF captures local sentiment and negation; character 3–5 gram TF-IDF absorbs morphology and spelling variation.", "grid"),
                ("Topic modelling", "pLSA discovers 20 latent topics with EM over a sparse document–term matrix. Topics provide an interpretable view of the corpus.", "layers"),
                ("Lexicon expansion", "TF-ICF scores cluster-specific terms, aligns topics to aspect seeds by cosine similarity and expands each aspect lexicon.", "zap"),
                ("Aspect detection", "A one-vs-rest TF-IDF logistic regression predicts five aspect categories; per-aspect thresholds are tuned on a development split.", "target"),
                ("Sentiment classification", "Per-aspect TF-IDF LR, calibrated linear SVM and GloVe-initialised BiLSTM classifiers predict negative, neutral or positive.", "activity"),
                ("Evaluation", "Accuracy, macro-F1, weighted-F1, per-aspect F1, Hamming loss, subset accuracy, confusion matrices and error analysis on the official test split.", "chart"),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        section_header(
            "Mathematical foundations",
            "Topic discovery and cluster-specific weighting",
        ),
        unsafe_allow_html=True,
    )
    eq_left, eq_right = st.columns(2, gap="large")
    with eq_left:
        with st.container(border=True):
            st.markdown("**pLSA · topic responsibility**")
            st.latex(
                r"P(z \mid d, w) = \frac{P(w \mid z)\,P(z \mid d)}{\sum_{z'} P(w \mid z')\,P(z' \mid d)}"
            )
            st.caption(
                "EM alternates between topic responsibilities and re-estimation of "
                "P(w | z) and P(z | d)."
            )
    with eq_right:
        with st.container(border=True):
            st.markdown("**TF-ICF · cluster specificity**")
            st.latex(
                r"\mathrm{tf\text{-}icf}(t, c) = \mathrm{tf}(t, c)\cdot \log\frac{N}{\mathrm{cf}(t)}"
            )
            st.caption(
                "Terms frequent in one cluster but rare elsewhere receive high weights "
                "and are transferred to the aligned aspect lexicon."
            )

    st.write("")
    st.markdown(
        section_header(
            "Models",
            "Model families and configurations",
        ),
        unsafe_allow_html=True,
    )
    model_rows = pd.DataFrame(
        [
            ("Majority class", "—", "Lower-bound sentiment baseline: most frequent polarity per aspect."),
            ("Aspect detector", "TF-IDF + one-vs-rest LR", "C tuned on 15% dev; per-aspect F1 thresholds."),
            ("Sentiment baseline", "TF-IDF + LR", "Balanced class weights; per-aspect C tuned on dev."),
            ("Comparison model", "TF-IDF + calibrated SVM", "Sigmoid calibration, three folds, balanced weights."),
            ("Neural model", "GloVe + BiLSTM", "50-d GloVe, 96 units per direction, mean + max pooling, dropout 0.35."),
            ("Unsupervised baseline", "pLSA + TF-ICF", "20 topics, lexicon induction, thresholds tuned on dev."),
        ],
        columns=["Component", "Model", "Configuration"],
    )
    render_table(model_rows)

    st.markdown(
        section_header(
            "Evaluation protocol",
            "Honest reporting on a held-out benchmark",
        ),
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            """
- The official training split is used for fitting.
- A 15% development split carved out of training is used for C selection, aspect thresholds and LSTM early stopping.
- The official gold test split is used once for final reporting; no test data is used for fitting or tuning.
- Metrics: accuracy, macro-F1, weighted-F1, per-aspect F1, micro/macro-F1, Hamming loss, subset accuracy and confusion matrices.
- Macro-F1 is emphasised because the neutral class is rare.
            """
        )

    show_figure(
        "architecture.png",
        "Implemented system architecture. Each block corresponds to code in the repository.",
    )
    show_figure(
        "plsa_topics.png",
        "Top TF-ICF terms for a selection of the discovered pLSA topics.",
    )


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.markdown(
        page_header(
            "Project information",
            "About the project",
            "A reproducible NLP coursework project combining topic modelling, classical "
            "text classification and neural sequence modelling.",
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            info_card(
                "book",
                "Project overview",
                "This project implements aspect-based sentiment analysis for restaurant reviews "
                "on the official SemEval-2014 Task 4 benchmark. It detects five aspect categories "
                "and predicts negative, neutral or positive sentiment for each one. The system "
                "compares an unsupervised pLSA + TF-ICF pipeline with supervised TF-IDF models "
                "and a GloVe-initialised BiLSTM.",
                "problem",
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            info_card(
                "shield",
                "Academic purpose",
                "The project demonstrates the full NLP lifecycle: data preparation, "
                "preprocessing, feature engineering, statistical topic modelling, supervised "
                "and neural classification, rigorous evaluation, error analysis and interactive "
                "deployment. Results are reported honestly, including cases where the neural "
                "model does not beat a strong classical baseline.",
                "solution",
            ),
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        section_header(
            "Technology",
            "Stack and objectives",
        ),
        unsafe_allow_html=True,
    )
    tech_left, tech_right = st.columns(2, gap="large")
    with tech_left:
        with st.container(border=True):
            st.markdown("**Technology stack**")
            st.markdown(
                """
- **Python 3.11**, NumPy and pandas
- **scikit-learn** — TF-IDF, logistic regression, linear SVM, metrics
- **PyTorch** — aspect-specific BiLSTM classifiers
- **NLTK** — stop-words and WordNet lemmatisation for pLSA
- **Matplotlib** — evaluation and architecture figures
- **Streamlit + Altair** — interactive application
- **pytest** — automated test suite
                """
            )
    with tech_right:
        with st.container(border=True):
            st.markdown("**Project objectives**")
            st.markdown(
                """
- Build a complete, reproducible ABSA pipeline.
- Compare interpretable topic/lexicon methods with supervised and neural models.
- Evaluate both aspect detection and aspect-specific sentiment.
- Package the trained system in a polished, usable application.
- Document limitations and failure modes honestly.
                """
            )

    st.write("")
    st.markdown(
        section_header(
            "Limitations and future scope",
            "Where the prototype can improve",
        ),
        unsafe_allow_html=True,
    )
    limit_left, limit_right = st.columns(2, gap="large")
    with limit_left:
        with st.container(border=True):
            st.markdown("**Limitations**")
            st.markdown(
                """
- **Category-level, not span-level:** the system does not extract the exact opinion target.
- **Domain and language:** trained on English restaurant reviews only.
- **Neutral class:** rare and semantically fuzzy, so macro-F1 is much lower than accuracy.
- **Aspect leakage:** whole-sentence classifiers can confuse polarities across aspects.
- **Pragmatics:** sarcasm, conditionals and long-range negation remain difficult.
- **BiLSTM data efficiency:** ~3.5k aspect–sentiment rows limit the neural model.
                """
            )
    with limit_right:
        with st.container(border=True):
            st.markdown("**Future scope**")
            st.markdown(
                """
- Aspect-term extraction with BIO tagging or span-based models.
- Transformer fine-tuning (e.g. RoBERTa) with GPU resources.
- Aspect-aware attention or masking to reduce leakage.
- Multi-domain and multilingual evaluation.
- Better confidence calibration and ordinal polarity modelling.
- Active learning to reduce annotation cost.
                """
            )

    st.write("")
    st.markdown(
        section_header(
            "References",
            "Selected sources",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
1. Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis.* SemEval 2014, pp. 27–35. <https://aclanthology.org/S14-2004>
2. Hofmann, T. (1999). *Probabilistic Latent Semantic Analysis.* UAI 1999.
3. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP 2014.
4. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
5. Joachims, T. (1998). *Text Categorization with Support Vector Machines.* ECML 1998.
6. Original repository: `wiqilee/absa-plsa-tficf-lstm` (MIT licence).
        """
    )
    st.caption(
        "Built for an NLP coursework submission. Figures and metrics are generated by the "
        "scripts under `scripts/` and stored under `outputs/`."
    )
