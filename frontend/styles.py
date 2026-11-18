"""Design-system CSS for the ABSA Studio Streamlit frontend.

The stylesheet is intentionally scoped to the application shell and to
explicit component classes.  It avoids broad ``div``/``span`` colour rules so
that dynamically rendered HTML keeps its intended contrast.
"""

from __future__ import annotations

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #4338CA;
    --primary-dark: #312E81;
    --secondary: #7C3AED;
    --accent: #2563EB;
    --bg: #F8FAFC;
    --surface: #FFFFFF;
    --surface-soft: #F1F5F9;
    --ink: #111827;
    --body: #475569;
    --muted: #64748B;
    --border: #E2E8F0;
    --border-strong: #CBD5E1;
    --success: #16A34A;
    --warning: #D97706;
    --danger: #DC2626;
    --radius: 18px;
    --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
    --shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 6% -4%, rgba(199, 210, 254, 0.55) 0, rgba(248, 250, 252, 0) 27rem),
        radial-gradient(circle at 96% 2%, rgba(221, 214, 254, 0.45) 0, rgba(248, 250, 252, 0) 24rem),
        var(--bg);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    background: transparent;
}

[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
[data-testid="stMainMenu"] {
    display: none !important;
}

[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
    box-shadow: var(--shadow-sm);
}

[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"]:hover {
    border-color: #A5B4FC !important;
    color: var(--primary) !important;
}

.block-container {
    max-width: 1340px;
    padding: 1.35rem 2.35rem 4.2rem;
}

p, li {
    color: var(--body);
}

h1, h2, h3, h4, h5 {
    color: var(--ink);
    letter-spacing: -0.02em;
}

a {
    color: var(--primary);
}

/* ------------------------------------------------------------------ */
/* Sidebar                                                             */
/* ------------------------------------------------------------------ */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
    box-shadow: 10px 0 34px rgba(15, 23, 42, 0.035);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1rem 0.85rem 1.4rem;
}

[data-testid="stSidebar"] .block-container {
    padding: 0;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 4px 20px;
}

.sidebar-mark {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    color: #FFFFFF !important;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    box-shadow: 0 10px 20px rgba(67, 56, 202, 0.22);
    flex: 0 0 auto;
}

.sidebar-mark svg {
    width: 22px;
    height: 22px;
}

.sidebar-name {
    font-size: 1.12rem;
    line-height: 1.1;
    font-weight: 800;
    color: var(--ink) !important;
    letter-spacing: -0.03em;
}

.sidebar-sub {
    margin-top: 4px;
    font-size: 0.72rem;
    line-height: 1.35;
    color: var(--muted) !important;
}

.sidebar-section {
    margin: 8px 4px 8px;
    font-size: 0.66rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    color: #94A3B8 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.28rem;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 10px 12px;
    border: 1px solid transparent;
    border-radius: 11px;
    cursor: pointer;
    transition: background 0.16s ease, border-color 0.16s ease, transform 0.16s ease;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
    background: #F8FAFC;
    border-color: var(--border);
    transform: translateX(2px);
}

[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
    background: linear-gradient(90deg, #EEF2FF 0%, #F5F3FF 100%);
    border-color: #C7D2FE;
    box-shadow: inset 3px 0 0 var(--primary);
}

[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {
    display: none;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"] p {
    margin: 0;
    font-size: 0.9rem;
    font-weight: 600;
    color: #334155 !important;
    white-space: nowrap;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {
    color: var(--primary) !important;
    font-weight: 700;
}

.side-divider {
    height: 1px;
    margin: 16px 4px;
    background: var(--border);
}

.side-card {
    margin: 0 2px 10px;
    padding: 14px 14px 12px;
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
}

.side-card-title {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted) !important;
    margin-bottom: 10px;
}

.status-row {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    margin-bottom: 8px;
}

.status-dot {
    margin-top: 4px;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 0 4px rgba(22, 163, 74, 0.12);
    flex: 0 0 auto;
}

.status-dot.error {
    background: var(--danger);
    box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.12);
}

.status-text {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--ink) !important;
}

.side-meta {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 5px 0;
    font-size: 0.77rem;
    border-bottom: 1px dashed #EDF1F7;
}

.side-meta:last-child {
    border-bottom: none;
}

.side-meta span {
    color: var(--muted) !important;
}

.side-meta b {
    color: #334155 !important;
    font-weight: 700;
    text-align: right;
}

.side-foot {
    margin: 16px 6px 0;
    font-size: 0.7rem;
    line-height: 1.55;
    color: #94A3B8 !important;
}

.side-foot b {
    color: #64748B !important;
}

/* ------------------------------------------------------------------ */
/* Hero and page headings                                              */
/* ------------------------------------------------------------------ */
.hero {
    position: relative;
    overflow: hidden;
    padding: 46px 48px 42px;
    margin-bottom: 24px;
    border-radius: 26px;
    background: linear-gradient(120deg, #312E81 0%, #4338CA 47%, #7C3AED 100%);
    box-shadow: 0 26px 60px rgba(67, 56, 202, 0.24);
}

.hero::before {
    content: "";
    position: absolute;
    top: -150px;
    right: -80px;
    width: 390px;
    height: 390px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
}

.hero::after {
    content: "";
    position: absolute;
    right: 90px;
    bottom: -170px;
    width: 310px;
    height: 310px;
    border-radius: 50%;
    border: 1px solid rgba(255, 255, 255, 0.16);
}

.hero-kicker {
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border: 1px solid rgba(255, 255, 255, 0.26);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.13);
    color: #E0E7FF !important;
    font-size: 0.69rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero-title {
    position: relative;
    z-index: 1;
    max-width: 980px;
    margin: 18px 0 14px;
    color: #FFFFFF !important;
    font-size: clamp(2rem, 3.3vw, 3rem);
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -0.04em;
}

.hero-sub {
    position: relative;
    z-index: 1;
    max-width: 900px;
    margin: 0;
    color: #E0E7FF !important;
    font-size: 1.04rem;
    line-height: 1.68;
}

.hero-chips {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 22px;
}

.tech-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 11px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.11);
    color: #FFFFFF !important;
    font-size: 0.76rem;
    font-weight: 600;
}

.page-head {
    margin: 2px 0 20px;
}

.page-kicker {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 5px;
    color: var(--primary) !important;
    font-size: 0.69rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.page-title {
    margin: 0 0 8px;
    color: var(--ink) !important;
    font-size: clamp(1.85rem, 2.6vw, 2.35rem);
    line-height: 1.14;
    font-weight: 800;
    letter-spacing: -0.035em;
}

.page-sub {
    max-width: 960px;
    margin: 0;
    color: var(--body) !important;
    font-size: 0.98rem;
    line-height: 1.62;
}

.section-head {
    margin: 8px 0 14px;
}

.section-kicker {
    color: var(--secondary) !important;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.section-title {
    margin: 3px 0 0;
    color: var(--ink) !important;
    font-size: 1.42rem;
    line-height: 1.25;
    font-weight: 800;
    letter-spacing: -0.025em;
}

.section-sub {
    margin: 5px 0 0;
    color: var(--muted) !important;
    font-size: 0.88rem;
    line-height: 1.55;
}

/* ------------------------------------------------------------------ */
/* Cards                                                               */
/* ------------------------------------------------------------------ */
.kpi-card,
.info-card,
.highlight-card,
.surface-card,
.pipeline-step,
.result-card,
.aspect-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
}

.kpi-card {
    height: 100%;
    min-height: 168px;
    padding: 19px 20px 17px;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow);
    border-color: #C7D2FE;
}

.kpi-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 14px;
}

.icon-box {
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    flex: 0 0 auto;
}

.icon-box svg {
    width: 20px;
    height: 20px;
}

.icon-indigo { background: #EEF2FF; color: var(--primary); }
.icon-purple { background: #F5F3FF; color: var(--secondary); }
.icon-blue { background: #EFF6FF; color: var(--accent); }
.icon-green { background: #F0FDF4; color: var(--success); }
.icon-amber { background: #FFFBEB; color: var(--warning); }
.icon-red { background: #FEF2F2; color: var(--danger); }

.kpi-label {
    color: var(--muted) !important;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.kpi-value {
    margin: 5px 0 4px;
    color: var(--ink) !important;
    font-size: 2.1rem;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -0.045em;
}

.kpi-sub {
    color: var(--muted) !important;
    font-size: 0.81rem;
    line-height: 1.42;
}

.info-card {
    height: 100%;
    padding: 24px 25px 22px;
    border-top: 4px solid var(--primary);
}

.info-card.solution {
    border-top-color: var(--secondary);
}

.info-head {
    display: flex;
    align-items: center;
    gap: 11px;
    margin-bottom: 12px;
}

.info-title {
    margin: 0;
    color: var(--ink) !important;
    font-size: 1.05rem;
    font-weight: 800;
}

.info-body {
    margin: 0;
    color: var(--body) !important;
    font-size: 0.92rem;
    line-height: 1.68;
}

.surface-card {
    padding: 22px 24px;
}

.highlight-card {
    height: 100%;
    min-height: 196px;
    padding: 19px 20px;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.highlight-card:hover {
    transform: translateY(-2px);
    border-color: #C7D2FE;
    box-shadow: var(--shadow);
}

.highlight-title {
    margin: 12px 0 5px;
    color: var(--ink) !important;
    font-size: 0.96rem;
    font-weight: 800;
}

.highlight-body {
    margin: 0;
    color: var(--muted) !important;
    font-size: 0.82rem;
    line-height: 1.55;
}

/* ------------------------------------------------------------------ */
/* Pipeline                                                            */
/* ------------------------------------------------------------------ */
.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    margin: 8px 0 10px;
}

.pipeline-step {
    position: relative;
    min-height: 150px;
    padding: 16px 16px 14px;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.pipeline-step:hover {
    transform: translateY(-2px);
    border-color: #C7D2FE;
    box-shadow: var(--shadow);
}

.pipeline-step:not(:nth-child(4n)):not(:last-child)::after {
    content: "→";
    position: absolute;
    top: 50%;
    right: -17px;
    transform: translateY(-50%);
    color: #A5B4FC;
    font-size: 1.05rem;
    font-weight: 800;
    z-index: 2;
}

.pipeline-num {
    color: var(--primary) !important;
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.pipeline-icon {
    margin: 11px 0 9px;
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border-radius: 10px;
}

.pipeline-icon svg {
    width: 18px;
    height: 18px;
}

.pipeline-title {
    color: var(--ink) !important;
    font-size: 0.92rem;
    font-weight: 800;
    line-height: 1.25;
}

.pipeline-desc {
    margin: 4px 0 0;
    color: var(--muted) !important;
    font-size: 0.76rem;
    line-height: 1.46;
}

/* ------------------------------------------------------------------ */
/* Results                                                             */
/* ------------------------------------------------------------------ */
.result-card {
    height: 100%;
    padding: 15px 16px 14px;
    border-left: 4px solid var(--primary);
}

.result-card.positive { border-left-color: var(--success); }
.result-card.negative { border-left-color: var(--danger); }
.result-card.neutral { border-left-color: var(--warning); }

.result-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 5px;
}

.result-aspect {
    color: var(--ink) !important;
    font-size: 0.92rem;
    font-weight: 800;
}

.result-meta {
    color: var(--muted) !important;
    font-size: 0.74rem;
    line-height: 1.4;
}

.result-conf {
    margin-top: 8px;
    color: #334155 !important;
    font-size: 0.76rem;
    font-weight: 700;
}

.aspect-card {
    padding: 17px 18px 15px;
    margin-bottom: 12px;
    border-left: 4px solid var(--primary);
}

.aspect-card.positive { border-left-color: var(--success); }
.aspect-card.negative { border-left-color: var(--danger); }
.aspect-card.neutral { border-left-color: var(--warning); }

.aspect-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
}

.aspect-name {
    color: var(--ink) !important;
    font-size: 1rem;
    font-weight: 800;
}

.aspect-sub {
    margin-top: 2px;
    color: var(--muted) !important;
    font-size: 0.78rem;
}

.verdict-card {
    padding: 20px 22px;
    margin-bottom: 14px;
    border: 1px solid #C7D2FE;
    border-radius: 16px;
    background: linear-gradient(120deg, #EEF2FF 0%, #F5F3FF 100%);
}

.verdict-label {
    color: var(--primary) !important;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.verdict-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 7px 0 5px;
    flex-wrap: wrap;
}

.verdict-value {
    color: var(--ink) !important;
    font-size: 1.35rem;
    font-weight: 800;
}

.verdict-note {
    margin: 0;
    color: var(--muted) !important;
    font-size: 0.78rem;
    line-height: 1.45;
}

/* ------------------------------------------------------------------ */
/* Badges and bars                                                     */
/* ------------------------------------------------------------------ */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 800;
    line-height: 1;
    text-transform: capitalize;
    white-space: nowrap;
}

.badge-positive {
    background: #DCFCE7;
    color: #15803D !important;
    border: 1px solid #BBF7D0;
}

.badge-negative {
    background: #FEE2E2;
    color: #B91C1C !important;
    border: 1px solid #FECACA;
}

.badge-neutral {
    background: #FEF3C7;
    color: #B45309 !important;
    border: 1px solid #FDE68A;
}

.prob-track {
    width: 100%;
    margin: 6px 0 4px;
    overflow: hidden;
    border-radius: 999px;
    background: #EEF2F7;
}

.prob-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease;
}

/* ------------------------------------------------------------------ */
/* Panels, empty states, timeline                                      */
/* ------------------------------------------------------------------ */
.panel-title {
    margin: 0 0 2px;
    color: var(--ink) !important;
    font-size: 1.02rem;
    font-weight: 800;
}

.panel-sub {
    margin: 0 0 13px;
    color: var(--muted) !important;
    font-size: 0.82rem;
    line-height: 1.48;
}

.empty-state {
    padding: 44px 30px;
    text-align: center;
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px dashed var(--border-strong);
    border-radius: 18px;
}

.empty-icon {
    width: 52px;
    height: 52px;
    margin: 0 auto 14px;
    display: grid;
    place-items: center;
    border-radius: 16px;
    background: #EEF2FF;
    color: var(--primary);
}

.empty-icon svg {
    width: 25px;
    height: 25px;
}

.empty-title {
    margin: 0 0 6px;
    color: var(--ink) !important;
    font-size: 1.08rem;
    font-weight: 800;
}

.empty-text {
    max-width: 430px;
    margin: 0 auto;
    color: var(--muted) !important;
    font-size: 0.86rem;
    line-height: 1.6;
}

.timeline {
    position: relative;
    padding-left: 6px;
    margin: 6px 0 12px;
}

.timeline::before {
    content: "";
    position: absolute;
    left: 21px;
    top: 8px;
    bottom: 8px;
    width: 2px;
    background: linear-gradient(180deg, #C7D2FE 0%, #DDD6FE 100%);
}

.timeline-item {
    position: relative;
    padding: 0 0 18px 56px;
}

.timeline-dot {
    position: absolute;
    left: 7px;
    top: 0;
    width: 30px;
    height: 30px;
    display: grid;
    place-items: center;
    border: 2px solid var(--primary);
    border-radius: 50%;
    background: #FFFFFF;
    color: var(--primary) !important;
    font-size: 0.7rem;
    font-weight: 800;
    box-shadow: 0 0 0 5px #EEF2FF;
}

.timeline-title {
    margin: 3px 0 4px;
    color: var(--ink) !important;
    font-size: 0.98rem;
    font-weight: 800;
}

.timeline-desc {
    margin: 0;
    color: var(--body) !important;
    font-size: 0.87rem;
    line-height: 1.58;
}

/* ------------------------------------------------------------------ */
/* Tables                                                              */
/* ------------------------------------------------------------------ */
.table-wrap {
    width: 100%;
    margin: 4px 0 16px;
    overflow-x: auto;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow-sm);
}

table.data-table {
    width: 100%;
    border-collapse: collapse;
    color: var(--ink);
    font-size: 0.84rem;
}

table.data-table thead th {
    padding: 11px 14px;
    border-bottom: 1px solid var(--border);
    background: #F8FAFC;
    color: #334155 !important;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-align: left;
    text-transform: uppercase;
    white-space: nowrap;
}

table.data-table tbody td {
    padding: 10px 14px;
    border-bottom: 1px solid #F1F5F9;
    color: var(--ink) !important;
    vertical-align: top;
}

table.data-table tbody tr:nth-child(even) {
    background: #FBFCFE;
}

table.data-table tbody tr:hover {
    background: #F5F3FF;
}

table.data-table tbody tr:last-child td {
    border-bottom: none;
}

/* ------------------------------------------------------------------ */
/* Native Streamlit widgets                                            */
/* ------------------------------------------------------------------ */
[data-testid="stWidgetLabel"] p,
[data-testid="stTextArea"] label p,
[data-testid="stSelectbox"] label p,
[data-testid="stSlider"] label p {
    color: var(--ink) !important;
    font-size: 0.84rem;
    font-weight: 600;
}

[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
    font-size: 0.8rem;
    line-height: 1.45;
}

[data-testid="stTextArea"] textarea {
    padding: 14px 16px;
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    background: #FFFFFF;
    color: var(--ink) !important;
    font-size: 0.95rem;
    line-height: 1.58;
    caret-color: var(--primary);
}

[data-testid="stTextArea"] textarea:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(67, 56, 202, 0.12);
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #94A3B8 !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    min-height: 42px;
    border-color: var(--border-strong);
    border-radius: 10px;
    background: #FFFFFF;
    color: var(--ink) !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 4px;
}

[data-testid="stButton"] button {
    min-height: 42px;
    padding: 0.55rem 1rem;
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    background: #FFFFFF;
    color: var(--ink) !important;
    font-weight: 700;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}

[data-testid="stButton"] button p,
[data-testid="stButton"] button div,
[data-testid="stButton"] button span {
    color: inherit !important;
    font-weight: inherit !important;
}

[data-testid="stButton"] button:hover {
    transform: translateY(-1px);
    border-color: #A5B4FC;
    box-shadow: 0 7px 18px rgba(15, 23, 42, 0.08);
}

[data-testid="stButton"] button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    border: none !important;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 9px 22px rgba(67, 56, 202, 0.24);
}

[data-testid="stButton"] button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 12px 28px rgba(67, 56, 202, 0.3);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--border);
}

[data-testid="stTabs"] button[role="tab"] {
    padding: 10px 16px;
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: 10px 10px 0 0;
    background: #FFFFFF;
    color: var(--body) !important;
    font-weight: 600;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    border-color: #C7D2FE;
    background: #EEF2FF;
    color: var(--primary) !important;
}

[data-testid="stExpander"] {
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #FFFFFF;
    box-shadow: var(--shadow-sm);
}

[data-testid="stExpander"] summary {
    color: var(--ink) !important;
    font-weight: 700;
}

[data-testid="stExpander"] summary:hover {
    color: var(--primary) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 10px 12px;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    background: #FFFFFF;
    box-shadow: var(--shadow-sm);
}

[data-testid="stDataFrame"] {
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #FFFFFF;
    box-shadow: var(--shadow-sm);
}

[data-testid="stAlert"] {
    border-radius: 12px;
    font-size: 0.88rem;
}

[data-testid="stJson"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #F8FAFC;
}

pre {
    border-radius: 12px !important;
}

[data-testid="stCaptionContainer"] {
    margin-top: 2px;
}

/* ------------------------------------------------------------------ */
/* Responsive                                                          */
/* ------------------------------------------------------------------ */
@media (max-width: 1100px) {
    .block-container {
        padding: 1.2rem 1.4rem 3.4rem;
    }
    .hero {
        padding: 36px 34px;
    }
    .pipeline-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .pipeline-step:not(:nth-child(4n)):not(:last-child)::after {
        display: none;
    }
}

@media (max-width: 760px) {
    .block-container {
        padding: 3.6rem 0.85rem 2.6rem;
    }
    .hero {
        padding: 28px 22px;
        border-radius: 18px;
    }
    .hero-title {
        font-size: 2rem;
    }
    .hero-sub {
        font-size: 0.95rem;
    }
    .pipeline-grid {
        grid-template-columns: 1fr;
    }
    .pipeline-step:not(:nth-child(4n)):not(:last-child)::after {
        display: none;
    }
    .kpi-value {
        font-size: 1.9rem;
    }
    .page-title {
        font-size: 1.7rem;
    }
    .section-title {
        font-size: 1.25rem;
    }
    .info-card,
    .surface-card {
        padding: 20px;
    }
    .timeline-item {
        padding-left: 48px;
    }
    .timeline::before {
        left: 17px;
    }
    .timeline-dot {
        left: 3px;
        width: 28px;
        height: 28px;
    }
}
</style>
"""
