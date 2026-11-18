"""Reusable HTML components for the ABSA Studio interface.

These helpers only render presentation markup.  Predictions, metrics and
dataset values are always supplied by the existing backend in ``app.py``.
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence

import pandas as pd

_ICONS = {
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>',
    "chart": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.7-4 3-9 3s-9-1.3-9-3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "trending": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "sparkles": '<path d="m12 3-1.9 5.7A2 2 0 0 1 8.7 10L3 12l5.7 1.9a2 2 0 0 1 1.4 1.4L12 21l1.9-5.7a2 2 0 0 1 1.4-1.4L21 12l-5.7-1.9a2 2 0 0 1-1.4-1.4z"/>',
    "alert": '<path d="M10.3 3.2 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.2a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "lightbulb": '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7V16h8v-1.3A7 7 0 0 0 12 2z"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "arrow": '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
}


def icon(name: str, size: int = 20) -> str:
    """Return an inline stroke icon that inherits the current text colour."""
    path = _ICONS.get(name, _ICONS["info"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{path}</svg>'
    )


def badge(label: str) -> str:
    safe = html.escape(str(label).lower())
    return f'<span class="badge badge-{safe}">{safe.title()}</span>'


def probability_bar(value: float, color: str = "#4338CA", height: int = 9) -> str:
    width = max(0.0, min(1.0, float(value))) * 100
    return (
        f'<div class="prob-track" style="height:{height}px">'
        f'<div class="prob-fill" style="width:{width:.1f}%;background:{color}"></div>'
        "</div>"
    )


def kpi_card(
    icon_name: str,
    label: str,
    value: str,
    subtitle: str,
    tone: str = "indigo",
) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-top"><span class="icon-box icon-{html.escape(tone)}">'
        f"{icon(icon_name)}</span>"
        f'<span class="kpi-label">{html.escape(label)}</span></div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{subtitle}</div>'
        "</div>"
    )


def section_header(kicker: str, title: str, subtitle: str | None = None) -> str:
    sub = f'<p class="section-sub">{html.escape(subtitle)}</p>' if subtitle else ""
    return (
        f'<div class="section-head"><div class="section-kicker">{html.escape(kicker)}</div>'
        f'<div class="section-title">{html.escape(title)}</div>{sub}</div>'
    )


def page_header(kicker: str, title: str, subtitle: str) -> str:
    return (
        f'<div class="page-head"><div class="page-kicker">{icon("sparkles", 13)} '
        f"{html.escape(kicker)}</div>"
        f'<div class="page-title">{html.escape(title)}</div>'
        f'<p class="page-sub">{html.escape(subtitle)}</p></div>'
    )


def hero(
    kicker: str,
    title: str,
    subtitle: str,
    chips: Sequence[str],
) -> str:
    chip_html = "".join(
        f'<span class="tech-chip">{html.escape(chip)}</span>' for chip in chips
    )
    return (
        '<div class="hero">'
        f'<div class="hero-kicker">{icon("sparkles", 13)} {html.escape(kicker)}</div>'
        f'<div class="hero-title">{html.escape(title)}</div>'
        f'<p class="hero-sub">{html.escape(subtitle)}</p>'
        f'<div class="hero-chips">{chip_html}</div>'
        "</div>"
    )


def info_card(
    icon_name: str,
    title: str,
    body: str,
    variant: str = "problem",
) -> str:
    return (
        f'<div class="info-card {variant}">'
        f'<div class="info-head"><span class="icon-box icon-indigo">{icon(icon_name)}</span>'
        f'<h3 class="info-title">{html.escape(title)}</h3></div>'
        f'<p class="info-body">{body}</p>'
        "</div>"
    )


def pipeline_steps(steps: Iterable[tuple[str, str, str, str]]) -> str:
    cards = []
    for index, (icon_name, title, description, tone) in enumerate(steps, start=1):
        cards.append(
            f'<div class="pipeline-step">'
            f'<div class="pipeline-num">Step {index:02d}</div>'
            f'<div class="pipeline-icon icon-{html.escape(tone)}">{icon(icon_name, 18)}</div>'
            f'<div class="pipeline-title">{html.escape(title)}</div>'
            f'<p class="pipeline-desc">{html.escape(description)}</p>'
            "</div>"
        )
    return f'<div class="pipeline-grid">{"".join(cards)}</div>'


def highlight_card(icon_name: str, title: str, body: str, tone: str = "indigo") -> str:
    return (
        f'<div class="highlight-card">'
        f'<span class="icon-box icon-{html.escape(tone)}">{icon(icon_name)}</span>'
        f'<div class="highlight-title">{html.escape(title)}</div>'
        f'<p class="highlight-body">{html.escape(body)}</p>'
        "</div>"
    )


def sidebar_brand() -> str:
    return (
        '<div class="sidebar-brand">'
        f'<div class="sidebar-mark">{icon("zap", 22)}</div>'
        '<div><div class="sidebar-name">ABSA Studio</div>'
        '<div class="sidebar-sub">Aspect-Based Sentiment Analysis<br>'
        "SemEval-2014 · Restaurant domain</div></div></div>"
    )


def sidebar_status(ok: bool, text: str) -> str:
    dot = "status-dot" if ok else "status-dot error"
    return (
        '<div class="side-card"><div class="side-card-title">System status</div>'
        f'<div class="status-row"><span class="{dot}"></span>'
        f'<span class="status-text">{html.escape(text)}</span></div>'
        '<div class="side-meta"><span>Inference</span><b>CPU · local</b></div>'
        '<div class="side-meta"><span>Artefacts</span><b>Loaded from disk</b></div>'
        "</div>"
    )


def sidebar_project(train_sentences: int | None, test_sentences: int | None) -> str:
    train_value = f"{train_sentences:,}" if train_sentences is not None else "—"
    test_value = f"{test_sentences:,}" if test_sentences is not None else "—"
    return (
        '<div class="side-card"><div class="side-card-title">Project snapshot</div>'
        f'<div class="side-meta"><span>Training</span><b>{train_value} reviews</b></div>'
        f'<div class="side-meta"><span>Evaluation</span><b>{test_value} reviews</b></div>'
        '<div class="side-meta"><span>Aspects</span><b>5 categories</b></div>'
        '<div class="side-meta"><span>Models</span><b>LR · SVM · BiLSTM</b></div>'
        "</div>"
    )


def sidebar_footer() -> str:
    return (
        '<div class="side-foot"><b>NLP coursework project</b><br>'
        "Project ID: ABSA-S14-REST<br>"
        "Built with Streamlit · scikit-learn · PyTorch</div>"
    )


def empty_state(icon_name: str, title: str, body: str) -> str:
    return (
        '<div class="empty-state">'
        f'<div class="empty-icon">{icon(icon_name, 25)}</div>'
        f'<div class="empty-title">{html.escape(title)}</div>'
        f'<p class="empty-text">{html.escape(body)}</p>'
        "</div>"
    )


def verdict_card(label: str, confidence: float, note: str) -> str:
    return (
        '<div class="verdict-card"><div class="verdict-label">Overall verdict · weighted</div>'
        f'<div class="verdict-row"><span class="verdict-value">{html.escape(label.title())}</span>'
        f"{badge(label)}"
        f'<span class="result-meta">{confidence:.1%} weighted confidence</span></div>'
        f'<p class="verdict-note">{html.escape(note)}</p></div>'
    )


def result_card(item: dict) -> str:
    sentiment = item["sentiment"]
    color = {"positive": "#16A34A", "negative": "#DC2626", "neutral": "#D97706"}[sentiment]
    return (
        f'<div class="result-card {sentiment}">'
        f'<div class="result-head"><span class="result-aspect">{html.escape(item["display"])}</span>'
        f"{badge(sentiment)}</div>"
        f'<div class="result-meta">aspect presence {item["aspect_probability"]:.1%}</div>'
        f'{probability_bar(item["aspect_probability"], "#4338CA")}'
        f'<div class="result-conf">Sentiment confidence · {item["confidence"]:.1%}</div>'
        f'{probability_bar(item["confidence"], color, 10)}'
        "</div>"
    )


def table_html(frame: pd.DataFrame) -> str:
    """Render a compact presentation table with readable headers and zebra rows."""
    return (
        '<div class="table-wrap">'
        + frame.to_html(index=False, classes="data-table", border=0, escape=True)
        + "</div>"
    )


def timeline(items: Iterable[tuple[str, str, str]]) -> str:
    rows = []
    for index, (title, description, icon_name) in enumerate(items, start=1):
        rows.append(
            '<div class="timeline-item">'
            f'<div class="timeline-dot">{index:02d}</div>'
            f'<div class="timeline-title">{icon(icon_name, 15)} {html.escape(title)}</div>'
            f'<p class="timeline-desc">{html.escape(description)}</p>'
            "</div>"
        )
    return f'<div class="timeline">{"".join(rows)}</div>'
