"""Streamlit application for aspect-based sentiment analysis.

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

from src.config import (
    ASPECTS,
    ASPECT_DESCRIPTIONS,
    ASPECT_DISPLAY,
    FIGURES_DIR,
    LSTM_META_PATH,
    METRICS_PATH,
    REPORTS_DIR,
    SENTIMENT_LABELS,
)
from src.data_loader import load_aspect_sentiment, load_dataset_stats, load_sentences
from src.predict import AspectSentimentEngine

# ---------------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ABSA Studio | Restaurant Reviews",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.stApp { background: #F8FAFC; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }

h1, h2, h3 { color: #0F172A; letter-spacing: -0.02em; }
p, li, span, div { color: #334155; }

.hero {
    background: linear-gradient(120deg, #0F172A 0%, #1E3A8A 55%, #2563EB 100%);
    border-radius: 22px; padding: 38px 42px; color: white; margin-bottom: 26px;
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.22);
}
.hero h1 { color: #FFFFFF; font-size: 2.35rem; line-height: 1.15; margin: 0 0 10px 0; }
.hero p { color: #DBEAFE; font-size: 1.03rem; max-width: 880px; margin: 0; }
.hero .kicker { display:inline-block; background: rgba(255,255,255,0.14); border:1px solid rgba(255,255,255,0.22);
    padding: 5px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 16px; color:#E0E7FF; }

.card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:22px 24px;
    box-shadow: 0 6px 18px rgba(15,23,42,0.05); height:100%; }
.card h3 { font-size:1.02rem; margin:0 0 8px 0; }
.card p { font-size:0.92rem; margin:0; color:#475569; line-height:1.55; }

.kpi { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:16px 18px;
    box-shadow:0 4px 12px rgba(15,23,42,0.04); }
.kpi .label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748B; font-weight:700; }
.kpi .value { font-size:1.7rem; font-weight:800; color:#0F172A; margin-top:2px; }
.kpi .hint { font-size:0.78rem; color:#94A3B8; }

.badge { display:inline-block; padding:5px 13px; border-radius:999px; font-size:0.78rem;
    font-weight:700; text-transform:capitalize; letter-spacing:0.01em; }
.badge-positive { background:#DCFCE7; color:#15803D; border:1px solid #BBF7D0; }
.badge-negative { background:#FEE2E2; color:#B91C1C; border:1px solid #FECACA; }
.badge-neutral  { background:#FEF3C7; color:#B45309; border:1px solid #FDE68A; }

.aspect-title { font-size:1.02rem; font-weight:700; color:#0F172A; }
.aspect-sub { font-size:0.8rem; color:#64748B; }
.prob-track { background:#EEF2F7; border-radius:999px; height:9px; overflow:hidden; margin:7px 0 3px 0; }
.prob-fill { height:100%; border-radius:999px; }
.small-note { font-size:0.78rem; color:#64748B; }
.section-label { font-size:0.72rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase; color:#64748B; margin-bottom:6px; }

div[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E2E8F0; }
div[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.stButton>button { border-radius:10px; font-weight:600; border:1px solid #CBD5E1; }
.stButton>button[kind="primary"] { background:#2563EB; border-color:#2563EB; }
.stTextArea textarea { border-radius:12px; border:1px solid #CBD5E1; font-size:0.95rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px 10px 0 0; padding:8px 16px; }
.stTabs [aria-selected="true"] { background:#EFF6FF; border-color:#BFDBFE; color:#1D4ED8; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
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


def figure_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, name)


def show_figure(name: str, caption: str | None = None) -> None:
    path = figure_path(name)
    if os.path.exists(path):
        st.image(path, width="stretch")
        if caption:
            st.caption(caption)
    else:
        st.info(f"Figure not found: {name}. Run `python -m scripts.generate_figures`.")


# ---------------------------------------------------------------------------
# Reusable HTML components
# ---------------------------------------------------------------------------
def badge(label: str) -> str:
    return f'<span class="badge badge-{label}">{label.title()}</span>'


def probability_bar(value: float, color: str = "#2563EB", height: int = 9) -> str:
    return (
        f'<div class="prob-track" style="height:{height}px">'
        f'<div class="prob-fill" style="width:{max(0, min(1, value)) * 100:.1f}%;background:{color}"></div>'
        f"</div>"
    )


SENTIMENT_COLORS = {"positive": "#10B981", "negative": "#EF4444", "neutral": "#F59E0B"}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.25rem;font-weight:800;color:#0F172A;letter-spacing:-0.02em;">🍽️ ABSA Studio</div>'
        '<div class="small-note" style="margin-bottom:18px;">Aspect-Based Sentiment Analysis<br>'
        "SemEval-2014 · Restaurant domain</div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Analyse Review",
            "Model Performance",
            "Dataset",
            "Methodology",
            "About",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<div class="section-label">System status</div>', unsafe_allow_html=True)
    try:
        _engine = get_engine()
        st.success(f"Models loaded · {len(_engine.available_sentiment_models)} sentiment models")
    except Exception as exc:  # pragma: no cover - only on missing artefacts
        st.error(f"Models not loaded:\n\n{exc}")
    st.markdown(
        '<div class="small-note" style="margin-top:14px;">Trained on 3,035 reviews · evaluated on 800 held-out reviews.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    metrics = get_metrics()
    stats = get_stats()

    best_aspect_micro = (
        metrics.get("aspect_detection", {}).get("tfidf_logreg", {}).get("micro_f1")
    )
    best_sentiment_macro = metrics.get("sentiment", {}).get("tfidf_logreg", {}).get("macro_f1")

    st.markdown(
        """
        <div class="hero">
          <div class="kicker">Natural Language Processing · Coursework Project</div>
          <h1>Aspect-Based Sentiment Analysis<br>of Restaurant Reviews</h1>
          <p>Discover <b>what</b> diners are talking about and <b>how</b> they feel about each aspect —
          food, service, price, ambience and overall experience — using topic modelling,
          TF-ICF lexicon expansion, TF-IDF classifiers and a BiLSTM neural network.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="kpi"><div class="label">Reviews</div><div class="value">3,841</div>'
            '<div class="hint">SemEval-2014 restaurant</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="kpi"><div class="label">Aspect categories</div><div class="value">5</div>'
            '<div class="hint">food · service · price · ambience · misc.</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi"><div class="label">Aspect detection</div><div class="value">{best_aspect_micro:.3f}</div>'
            '<div class="hint">micro-F1 · held-out test set</div></div>'
            if best_aspect_micro is not None
            else '<div class="kpi"><div class="label">Aspect detection</div><div class="value">—</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi"><div class="label">Sentiment macro-F1</div><div class="value">{best_sentiment_macro:.3f}</div>'
            '<div class="hint">TF-IDF + logistic regression</div></div>'
            if best_sentiment_macro is not None
            else '<div class="kpi"><div class="label">Sentiment macro-F1</div><div class="value">—</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(
            """
            <div class="card">
              <h3>The problem</h3>
              <p>Star ratings collapse an entire visit into a single number. A restaurant can serve
              excellent food while the service is rude and the prices are unfair. Businesses need
              <b>aspect-level</b> feedback to know what to fix, yet manually reading thousands of
              reviews is not scalable.</p>
              <br>
              <h3>The solution</h3>
              <p>This system reads a free-text review, detects which aspects are discussed, and
              predicts the polarity of each aspect independently. It combines a transparent
              probabilistic topic model (pLSA) and TF-ICF lexicon induction with supervised
              text classifiers, and wraps them in an interactive demonstration.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
            <div class="card">
              <h3>What you can do here</h3>
              <p>• Analyse your own review and inspect per-aspect confidence scores.<br>
              • Compare the BiLSTM against TF-IDF baselines on the same input.<br>
              • Explore the pLSA topics and the induced aspect lexicon.<br>
              • Review actual test-set metrics, confusion matrices and error analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-label">Pipeline</div>', unsafe_allow_html=True)
        show_figure("architecture.png")

    st.divider()
    st.subheader("Try a review")
    st.caption("This quick demo uses the saved models. Open **Analyse Review** for full controls.")
    quick_default = (
        "The pasta was absolutely delicious and the staff were friendly, "
        "but the music was too loud and the prices are a bit high."
    )
    quick_text = st.text_area("Review text", value=quick_default, height=100, key="overview_input")
    qcol1, qcol2 = st.columns([0.2, 0.8])
    with qcol1:
        run_quick = st.button("Analyse", type="primary", key="overview_run")
    if run_quick:
        try:
            engine = get_engine()
            result = engine.predict(
                quick_text,
                aspect_method="supervised",
                sentiment_model="tfidf_logreg",
            )
            st.markdown(
                f"**Aspects detected:** {', '.join(ASPECT_DISPLAY[a] for a in result['detected_aspects']) or 'none above threshold'}"
            )
            cols = st.columns(len(result["aspects"]))
            for col, item in zip(cols, result["aspects"]):
                with col:
                    st.markdown(
                        f'<div class="card"><div class="aspect-title">{item["display"]}</div>'
                        f'<div style="margin:8px 0;">{badge(item["sentiment"])}</div>'
                        f'<div class="aspect-sub">confidence {item["confidence"]:.1%}</div>'
                        f'{probability_bar(item["confidence"], SENTIMENT_COLORS[item["sentiment"]])}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        except Exception as exc:
            st.error(str(exc))

    st.write("")
    with st.expander("How the model works in one paragraph"):
        st.write(
            "Each review is normalised (lowercase, HTML/URL removal, contraction expansion and "
            "tokenisation that preserves negation). A one-vs-rest TF-IDF logistic regression "
            "detects which of the five aspect categories are present. For every detected aspect, "
            "an aspect-specific classifier predicts negative, neutral or positive sentiment from "
            "word and character n-grams (or a GloVe-initialised BiLSTM). The pLSA + TF-ICF component "
            "independently discovers latent topics and induces an aspect lexicon, which serves as an "
            "unsupervised baseline and as an interpretable view of the corpus."
        )


# ---------------------------------------------------------------------------
# Page: Analyse Review
# ---------------------------------------------------------------------------
elif page == "Analyse Review":
    st.title("Analyse a review")
    st.caption(
        "Enter a restaurant review. The system detects the aspects it discusses and predicts "
        "the sentiment for each one."
    )

    test_sentences = get_test_sentences()
    examples: List[str] = []
    if not test_sentences.empty:
        multi = test_sentences[test_sentences["num_aspects"] >= 2]
        pool = multi if len(multi) >= 6 else test_sentences
        examples = pool.sample(n=min(6, len(pool)), random_state=7)["text"].tolist()

    if "review_input" not in st.session_state:
        st.session_state["review_input"] = (
            "The pizza was delicious and the service was quick, but the restaurant was noisy "
            "and the prices were too high."
        )
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        if examples:
            st.caption("Load an example from the held-out test set:")
            example_cols = st.columns(3)
            for idx, example in enumerate(examples):
                with example_cols[idx % 3]:
                    if st.button(f"Example {idx + 1}", key=f"example_{idx}", width="stretch"):
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
                    "supervised": "Supervised · TF-IDF + one-vs-rest LR",
                    "lexicon": "Unsupervised baseline · pLSA + TF-ICF lexicon",
                }[value],
                help="The unsupervised baseline is weaker but demonstrates what pLSA + TF-ICF learned without labels.",
            )
            engine_for_list = get_engine()
            sentiment_model = st.selectbox(
                "Sentiment classifier",
                engine_for_list.available_sentiment_models,
                index=(
                    engine_for_list.available_sentiment_models.index("tfidf_logreg")
                    if "tfidf_logreg" in engine_for_list.available_sentiment_models
                    else 0
                ),
                format_func=lambda value: {
                    "bilstm": "BiLSTM (GloVe-initialised)",
                    "tfidf_logreg": "TF-IDF + Logistic Regression",
                    "tfidf_svm": "TF-IDF + Linear SVM (calibrated)",
                }.get(value, value),
            )
            threshold = st.slider(
                "Minimum aspect confidence",
                min_value=0.0,
                max_value=0.9,
                value=0.0,
                step=0.05,
                help="Per-aspect tuned thresholds are applied automatically; this slider can only make them stricter.",
            )
            top_k = st.slider(
                "Always analyse the top-k aspects (0 = only detected aspects)",
                min_value=0,
                max_value=5,
                value=0,
                step=1,
            )

        run = st.button("Analyse review", type="primary", width="stretch")
        st.caption(
            "Input is validated: empty reviews and reviews with fewer than two words are rejected."
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
        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
        result = st.session_state.get("analysis_result")
        if result is None:
            st.markdown(
                '<div class="card"><div class="aspect-title">No analysis yet</div>'
                '<div class="aspect-sub" style="margin-top:6px;">Enter a review and press '
                "<b>Analyse review</b>. The system will show each aspect, its sentiment and the "
                "model's confidence.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            if result.get("low_confidence"):
                st.warning(
                    "No aspect passed the detection threshold. The most likely aspect is shown "
                    "below, but treat the result as low confidence."
                )
            overall = result["overall"]
            st.markdown(
                f'<div class="card"><div class="section-label">Overall verdict (weighted)</div>'
                f'<div style="display:flex;align-items:center;gap:14px;margin-top:6px;">'
                f'<div style="font-size:1.15rem;font-weight:700;color:#0F172A;">{overall["label"].title()}</div>'
                f"{badge(overall['label'])}"
                f'<div class="aspect-sub">{overall["confidence"]:.1%} weighted confidence</div></div>'
                f'<div class="small-note" style="margin-top:8px;">Weighted by aspect-presence probability; '
                f"it is a summary, not a separately trained classifier.</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")

            for item in result["aspects"]:
                probs = item["probabilities"]
                st.markdown(
                    f'<div class="card" style="margin-bottom:12px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;">'
                    f'<div><div class="aspect-title">{item["display"]}</div>'
                    f'<div class="aspect-sub">aspect presence {item["aspect_probability"]:.1%}</div></div>'
                    f"<div>{badge(item['sentiment'])}</div></div>"
                    f'{probability_bar(item["aspect_probability"], "#2563EB")}'
                    f'<div style="margin-top:12px;"><div class="aspect-sub" style="margin-bottom:5px;">'
                    f'Sentiment confidence · {item["confidence"]:.1%}</div>'
                    f'{probability_bar(item["confidence"], SENTIMENT_COLORS[item["sentiment"]], 11)}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander(f"{item['display']} · class probabilities"):
                    prob_frame = pd.DataFrame(
                        {
                            "sentiment": [label.title() for label in SENTIMENT_LABELS],
                            "probability": [probs[label] for label in SENTIMENT_LABELS],
                        }
                    )
                    chart = (
                        alt.Chart(prob_frame)
                        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                        .encode(
                            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                            y=alt.Y("sentiment:N", sort=None, title=None),
                            color=alt.Color(
                                "sentiment:N",
                                scale=alt.Scale(
                                    domain=["Negative", "Neutral", "Positive"],
                                    range=["#EF4444", "#F59E0B", "#10B981"],
                                ),
                                legend=None,
                            ),
                            tooltip=["sentiment", alt.Tooltip("probability:Q", format=".3f")],
                        )
                        .properties(height=110)
                    )
                    st.altair_chart(chart, width="stretch")

            with st.expander("Aspect-presence probabilities (all five aspects)"):
                aspect_frame = pd.DataFrame(
                    {
                        "aspect": [ASPECT_DISPLAY[a] for a in ASPECTS],
                        "probability": [result["aspect_probabilities"].get(a, 0.0) for a in ASPECTS],
                        "detected": [
                            "yes" if a in result["detected_aspects"] else "no" for a in ASPECTS
                        ],
                    }
                ).sort_values("probability", ascending=False)
                st.dataframe(
                    aspect_frame.style.format({"probability": "{:.3f}"}),
                    width="stretch",
                    hide_index=True,
                )
                st.bar_chart(
                    aspect_frame.set_index("aspect")["probability"], height=240, color="#2563EB"
                )
                st.caption(
                    f"Tuned thresholds: "
                    + ", ".join(f"{ASPECT_DISPLAY[a]} {result['thresholds'][a]:.2f}" for a in ASPECTS)
                )

            with st.expander("Preprocessing details"):
                st.write("**Cleaned text**")
                st.code(result["cleaned_text"], language=None)
                st.write("**Model configuration**")
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
    st.title("Model performance")
    metrics = get_metrics()
    if not metrics:
        st.error(
            "Metrics not found. Run `python -m scripts.evaluate_all` and "
            "`python -m scripts.generate_figures`."
        )
        st.stop()

    test_info = metrics.get("test_set", {})
    st.caption(
        f"All numbers below are computed on the official SemEval-2014 test split "
        f"({test_info.get('sentences', '—')} sentences, {test_info.get('aspect_sentiment_rows', '—')} "
        f"aspect-sentiment annotations). No test data was used for training or tuning."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Sentiment classification", "Aspect detection", "Error analysis"]
    )

    with tab1:
        sentiment = metrics["sentiment"]
        best_name = max(
            (n for n in sentiment if n != "majority"),
            key=lambda n: sentiment[n]["macro_f1"],
        )
        best = sentiment[best_name]
        display_names = {
            "majority": "Majority class",
            "tfidf_logreg": "TF-IDF + Logistic Regression",
            "tfidf_svm": "TF-IDF + Linear SVM",
            "bilstm": "BiLSTM",
        }
        cols = st.columns(4)
        with cols[0]:
            st.metric("Best model", display_names.get(best_name, best_name))
        with cols[1]:
            st.metric("Accuracy", f"{best['accuracy']:.3f}")
        with cols[2]:
            st.metric("Macro-F1", f"{best['macro_f1']:.3f}")
        with cols[3]:
            st.metric("Weighted-F1", f"{best['weighted_f1']:.3f}")

        st.write("")
        st.subheader("Model comparison")
        table = pd.DataFrame(
            [
                {
                    "Model": display_names.get(name, name),
                    "Accuracy": round(value["accuracy"], 4),
                    "Macro-F1": round(value["macro_f1"], 4),
                    "Weighted-F1": round(value["weighted_f1"], 4),
                    "N": value["n_samples"],
                }
                for name, value in sentiment.items()
            ]
        ).sort_values("Macro-F1", ascending=False)
        st.dataframe(table, width="stretch", hide_index=True)

        show_figure("model_comparison.png")
        show_figure("confusion_matrices.png")
        show_figure("per_aspect_f1.png")
        show_figure("bilstm_training_curves.png")

        report = get_classification_report()
        if report:
            with st.expander("Classification report (best model)"):
                st.code(report, language=None)

    with tab2:
        detection = metrics["aspect_detection"]
        display_names = {
            "tfidf_logreg": "Supervised TF-IDF + LR",
            "lexicon_plsa_tficf": "pLSA + TF-ICF lexicon (unsupervised)",
        }
        rows = []
        for name, value in detection.items():
            rows.append(
                {
                    "Model": display_names.get(name, name),
                    "Micro-F1": round(value["micro_f1"], 4),
                    "Macro-F1": round(value["macro_f1"], 4),
                    "Hamming loss": round(value["hamming_loss"], 4),
                    "Subset accuracy": round(value["subset_accuracy"], 4),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        show_figure("aspect_detection_comparison.png")
        show_figure("aspect_detection_per_aspect.png")

        st.subheader("Per-aspect detection quality")
        detail_rows = []
        for aspect in ASPECTS:
            detail_rows.append(
                {
                    "Aspect": ASPECT_DISPLAY[aspect],
                    "Supervised precision": round(
                        detection["tfidf_logreg"]["per_aspect"][aspect]["precision"], 4
                    ),
                    "Supervised recall": round(
                        detection["tfidf_logreg"]["per_aspect"][aspect]["recall"], 4
                    ),
                    "Supervised F1": round(
                        detection["tfidf_logreg"]["per_aspect"][aspect]["f1"], 4
                    ),
                    "Lexicon F1": round(
                        detection["lexicon_plsa_tficf"]["per_aspect"][aspect]["f1"], 4
                    ),
                }
            )
        st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)

        with st.expander("Why the supervised detector wins"):
            st.write(
                "The lexicon baseline is fully unsupervised: pLSA discovers 20 latent topics, "
                "TF-ICF selects cluster-specific terms, and topics are aligned to aspect seeds by "
                "cosine similarity. It is interpretable and needs no labels, but it cannot learn "
                "that phrases such as 'we will be back' signal a general (miscellaneous) aspect "
                "without lexical overlap. The supervised one-vs-rest classifier learns these cues "
                "from 3,035 labelled sentences, which explains the large gap in macro-F1."
            )

    with tab3:
        show_figure("error_confidence.png")
        errors = get_error_analysis()
        if not errors.empty:
            st.subheader("Most confident mistakes")
            st.caption(
                "High-confidence errors reveal annotation ambiguity and aspect-specific context "
                "that the model misses."
            )
            view = errors.head(15)[
                ["text", "aspect", "gold", "predicted", "confidence"]
            ].rename(
                columns={
                    "text": "Review",
                    "aspect": "Aspect",
                    "gold": "Gold",
                    "predicted": "Predicted",
                    "confidence": "Confidence",
                }
            )
            st.dataframe(
                view.style.format({"Confidence": "{:.3f}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No error analysis file found.")


# ---------------------------------------------------------------------------
# Page: Dataset
# ---------------------------------------------------------------------------
elif page == "Dataset":
    st.title("Dataset")
    stats = get_stats()
    if not stats:
        st.error("Dataset statistics not found. Run `python -m scripts.prepare_data`.")
        st.stop()

    train = stats["splits"]["train"]
    test = stats["splits"]["test"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training sentences", f"{train['sentences']:,}")
    c2.metric("Test sentences", f"{test['sentences']:,}")
    c3.metric("Training aspect annotations", f"{train['aspect_annotations']:,}")
    c4.metric("Test aspect annotations", f"{test['aspect_annotations']:,}")

    st.write("")
    st.markdown(
        """
        <div class="card">
          <h3>Source</h3>
          <p><b>SemEval-2014 Task 4: Aspect Based Sentiment Analysis</b> — restaurant domain
          (Pontiki et al., 2014). Each sentence is annotated with aspect categories
          (<i>food, service, price, ambience, anecdotes/miscellaneous</i>) and a polarity for each
          category. The official training and gold test splits are used unchanged; six exact
          duplicate texts were removed from the training split only.</p>
          <p style="margin-top:8px;">Data obtained from the public ACL Anthology / SemEval release
          (mirrored on Hugging Face: <code>alexcadillon/SemEval2014Task4</code>).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    show_figure("dataset_distribution.png")
    show_figure("aspect_sentiment_distribution.png")

    st.subheader("Class distribution")
    dist_rows = []
    for aspect in ASPECTS:
        row = {"Aspect": ASPECT_DISPLAY[aspect]}
        for label in SENTIMENT_LABELS:
            row[label.title()] = train["aspect_sentiment_distribution"].get(
                f"{aspect}/{label}", 0
            )
        dist_rows.append(row)
    st.dataframe(pd.DataFrame(dist_rows), width="stretch", hide_index=True)
    st.caption(
        "`conflict` annotations (5.2% of training categories) are excluded from the three-class "
        "sentiment task because they do not represent a coherent polarity."
    )

    st.subheader("Sample records")
    test_sentiment = get_test_sentiment()
    if not test_sentiment.empty:
        sample = test_sentiment.sample(n=min(10, len(test_sentiment)), random_state=11)
        st.dataframe(
            sample[["text", "aspect", "label"]].rename(
                columns={"text": "Review", "aspect": "Aspect", "label": "Polarity"}
            ),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Dataset statistics (JSON)"):
        st.json(stats)

    with st.expander("Preprocessing examples"):
        from src.preprocessing import clean_text

        sentence_frame = get_test_sentences()
        examples = sentence_frame.sample(n=min(4, len(sentence_frame)), random_state=3)["text"].tolist()
        for example in examples:
            st.markdown(f"**Raw:** {example}")
            st.markdown(f"**Cleaned:** `{clean_text(example)}`")
            st.divider()


# ---------------------------------------------------------------------------
# Page: Methodology
# ---------------------------------------------------------------------------
elif page == "Methodology":
    st.title("NLP methodology")
    st.caption(
        "Every stage below is implemented in the repository; the figures and numbers are "
        "generated from the actual pipeline."
    )

    show_figure("workflow.png")

    st.subheader("1 · Task formulation")
    st.write(
        "Given a review sentence, the system solves two sub-tasks: (i) **aspect category "
        "detection**, a five-label multi-label classification problem, and (ii) **aspect "
        "sentiment classification**, a three-class problem (negative / neutral / positive) for "
        "every detected aspect. This decomposition follows the SemEval-2014 ABSA formulation and "
        "mirrors real review behaviour, where one sentence can mention several aspects with "
        "different polarities."
    )

    st.subheader("2 · Preprocessing")
    preprocessing_rows = [
        ("Unicode normalisation (NFKC)", "Yes", "Canonicalises ligatures and full-width characters."),
        ("HTML / URL removal", "Yes", "Review text scraped from the web contains markup."),
        ("Contraction expansion", "Yes", "Preserves negation (`wasn't` → `was not`)."),
        ("Lowercasing", "Yes", "Reduces vocabulary sparsity."),
        ("Punctuation handling", "Partial", "Dropped, but `!` becomes an emphasis token."),
        ("Tokenisation", "Yes", "Regex tokeniser; keeps apostrophes and negators."),
        ("Stop-word removal", "Only for pLSA", "Removes topical noise; supervised models keep negators."),
        ("Lemmatisation", "Only for pLSA", "Groups inflections for topic discovery; avoids harming GloVe."),
        ("Duplicate removal", "Training only", "Six exact duplicate texts removed; test kept intact."),
    ]
    st.dataframe(
        pd.DataFrame(preprocessing_rows, columns=["Technique", "Applied", "Rationale"]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("3 · Feature extraction")
    st.write(
        "The supervised models use a union of **word 1–2 gram TF-IDF** and **character 3–5 gram "
        "TF-IDF** features. Word n-grams capture negation and local sentiment (`not good`, "
        "`great food`); character n-grams absorb spelling variation and morphology. Sublinear "
        "term-frequency scaling prevents very frequent words from dominating. Vectorisers are "
        "fitted on training text only."
    )

    st.subheader("4 · pLSA + TF-ICF")
    st.write(
        "pLSA models each document as a mixture of latent topics and each topic as a distribution "
        "over words. EM estimates the parameters:"
    )
    st.latex(r"P(z \mid d, w) = \frac{P(w \mid z)\,P(z \mid d)}{\sum_{z'} P(w \mid z')\,P(z' \mid d)}")
    st.write(
        "Documents are clustered by their dominant topic. TF-ICF then scores terms that are "
        "frequent inside one cluster but rare elsewhere:"
    )
    st.latex(r"\mathrm{tf\text{-}icf}(t, c) = \mathrm{tf}(t, c)\cdot \log\frac{N}{\mathrm{cf}(t)}")
    st.write(
        "Aspect seeds are aligned to topics by cosine similarity, and the top TF-ICF terms from "
        "aligned topics are transferred into the aspect lexicon. This produces an unsupervised "
        "aspect detector and a human-readable summary of what the corpus talks about."
    )
    show_figure("plsa_topics.png")

    st.subheader("5 · Models")
    model_rows = [
        ("Majority class", "—", "Lower-bound baseline: predicts the most frequent polarity per aspect."),
        ("Aspect detector", "TF-IDF + one-vs-rest logistic regression", "C tuned on a 15% dev split; per-aspect thresholds tuned for F1."),
        ("Baseline sentiment", "TF-IDF + logistic regression", "C tuned per aspect on the dev split; balanced class weights."),
        ("Comparison sentiment", "TF-IDF + calibrated linear SVM", "Sigmoid calibration on three folds; balanced class weights."),
        ("Neural sentiment", "GloVe-initialised BiLSTM", "50-d GloVe, 96 units per direction, masked mean + max pooling, dropout 0.35, early stopping on dev macro-F1."),
        ("Unsupervised aspect baseline", "pLSA (20 topics) + TF-ICF lexicon", "Thresholds tuned on dev; no labels used for lexicon induction."),
    ]
    st.dataframe(
        pd.DataFrame(model_rows, columns=["Component", "Model", "Configuration"]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("6 · Evaluation protocol")
    st.write(
        "The official training split is used for fitting. A 15% development split carved out of "
        "training is used for C selection, aspect thresholds and LSTM early stopping. The official "
        "gold test split is used once for final reporting. Metrics: accuracy, macro-F1, weighted-F1, "
        "per-aspect F1, micro/macro-F1 for aspect detection, Hamming loss, subset accuracy and "
        "confusion matrices. Macro-F1 is emphasised because the neutral class is rare."
    )
    show_figure("architecture.png")


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.title("About the project")
    st.markdown(
        """
        <div class="card">
          <h3>Objectives</h3>
          <p>1. Build a complete, reproducible ABSA pipeline for restaurant reviews.<br>
          2. Compare an interpretable topic-model/lexicon approach with supervised classifiers
          and a neural model.<br>
          3. Evaluate aspect detection and aspect-specific sentiment with standard metrics.<br>
          4. Deliver an interactive demonstration that uses the trained models — no hard-coded
          predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Technology stack")
        st.markdown(
            """
            - **Python 3.11**, NumPy, pandas
            - **scikit-learn** — TF-IDF, logistic regression, linear SVM, metrics
            - **PyTorch** — BiLSTM classifier
            - **NLTK** — stop-words and WordNet lemmatisation (pLSA only)
            - **Matplotlib** — figures
            - **Streamlit + Altair** — application
            - **pytest** — automated tests
            """
        )
    with c2:
        st.subheader("Repository layout")
        st.code(
            """app.py                  Streamlit application
src/
  data_loader.py        SemEval XML → CSV views
  preprocessing.py      normalisation and tokenisation
  features.py           TF-IDF feature union
  plsa.py               EM-based pLSA
  tf_icf.py             TF-ICF weighting
  aspect_lexicon.py     lexicon induction + baseline
  lstm_model.py         BiLSTM training/inference
  train.py              training routines
  evaluate.py           metrics
  predict.py            inference engine
scripts/
  prepare_data.py       build processed datasets
  train_all.py          train every component
  evaluate_all.py       test-set evaluation
  generate_figures.py   report figures
tests/                  pytest suite""",
            language=None,
        )

    st.subheader("Limitations")
    st.markdown(
        """
        - **Category-level supervision.** The model predicts aspect categories and polarity; it
          does not extract the exact opinion target span (e.g. “the *truffle pasta* was divine”).
        - **Domain and language.** Trained on English restaurant reviews from SemEval-2014;
          performance on hotels, electronics or other languages is not established.
        - **Neutral class.** Neutral is rare and semantically fuzzy, so macro-F1 is noticeably
          lower than accuracy. The majority baseline cannot predict neutral at all.
        - **BiLSTM data efficiency.** With ~3.5k annotated aspect-sentiment rows, TF-IDF
          baselines remain competitive; the BiLSTM is hurt by limited data and 50-d embeddings.
        - **Annotator ambiguity.** SemEval’s `conflict` label and subjective sentences create
          irreducible errors, visible in the error analysis.
        - **Lexicon baseline.** pLSA + TF-ICF is unsupervised and interpretable but cannot learn
          pragmatic cues; it is included for comparison, not as the production detector.
        """
    )

    st.subheader("Future work")
    st.markdown(
        """
        - Aspect-term extraction with BIO tagging or span-based models.
        - Fine-tune a transformer (e.g. RoBERTa) if GPU resources are available.
        - Multi-domain and multilingual evaluation.
        - Ordinal modelling of polarity and confidence calibration.
        - Active learning to reduce annotation cost.
        """
    )

    st.subheader("References")
    st.markdown(
        """
        1. Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., &
           Manandhar, S. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis.*
           Proceedings of SemEval 2014, pp. 27–35. https://aclanthology.org/S14-2004
        2. Hofmann, T. (1999). *Probabilistic Latent Semantic Analysis.* Proceedings of UAI 1999.
        3. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word
           Representation.* Proceedings of EMNLP 2014.
        4. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
        5. Joachims, T. (1998). *Text Categorization with Support Vector Machines.* ECML 1998.
        6. This project extends the open-source repository
           `wiqilee/absa-plsa-tficf-lstm` (MIT licence).
        """
    )
    st.caption(
        "Built for an NLP coursework submission. Results are generated by the scripts in "
        "`scripts/` and stored under `outputs/`."
    )
