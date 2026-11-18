"""Build the ABSA Studio guide PDF from the Markdown source.

The Markdown dialect is intentionally small but sufficient for the guide:
headings, paragraphs, bold/italic/code spans, bullet and numbered lists,
pipe tables, fenced code, blockquote callouts, images, two-column image grids,
display formulas and explicit page breaks.
"""

from __future__ import annotations

import os
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
MD_PATH = os.path.join(BASE, "ABSA_Studio_Project_Guide.md")
PDF_PATH = os.path.join(BASE, "ABSA_Studio_Project_Guide.pdf")

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
FONT_DIR = "/System/Library/Fonts/Supplemental"
pdfmetrics.registerFont(TTFont("Arial", os.path.join(FONT_DIR, "Arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", os.path.join(FONT_DIR, "Arial Bold.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", os.path.join(FONT_DIR, "Arial Italic.ttf")))
pdfmetrics.registerFont(
    TTFont("Arial-BoldItalic", os.path.join(FONT_DIR, "Arial Bold Italic.ttf"))
)
pdfmetrics.registerFontFamily(
    "Arial",
    normal="Arial",
    bold="Arial-Bold",
    italic="Arial-Italic",
    boldItalic="Arial-BoldItalic",
)
MONO = os.path.join(
    ROOT, ".venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf"
)
pdfmetrics.registerFont(TTFont("Mono", os.path.join(MONO, "DejaVuSansMono.ttf")))

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY = colors.HexColor("#0F172A")
INK = colors.HexColor("#1E293B")
BODY = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748B")
INDIGO = colors.HexColor("#4338CA")
DEEP_INDIGO = colors.HexColor("#312E81")
PURPLE = colors.HexColor("#7C3AED")
BORDER = colors.HexColor("#E2E8F0")
SOFT = colors.HexColor("#F8FAFC")
LAVENDER = colors.HexColor("#EEF2FF")
PALE_INDIGO = colors.HexColor("#C7D2FE")

PAGE_W, PAGE_H = A4
LM, RM = 16 * mm, 16 * mm
TM, BM = 18 * mm, 16 * mm
CONTENT_W = PAGE_W - LM - RM
CONTENT_H = PAGE_H - TM - BM

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
ss = getSampleStyleSheet()

BODY_STYLE = ParagraphStyle(
    "BodyText",
    fontName="Arial",
    fontSize=10.0,
    leading=12.85,
    textColor=BODY,
    alignment=TA_LEFT,
    spaceAfter=2.9,
    spaceBefore=0,
    allowWidows=0,
)
LEAD_STYLE = ParagraphStyle(
    "Lead", parent=BODY_STYLE, fontSize=10.4, leading=14.1, textColor=INK,
    spaceAfter=4,
)
H1_KICKER = ParagraphStyle(
    "ChapterKicker", fontName="Arial-Bold", fontSize=7.8, leading=9.5,
    textColor=INDIGO, spaceBefore=0, spaceAfter=0.5,
)
H1 = ParagraphStyle(
    "ChapterTitle", fontName="Arial-Bold", fontSize=19.5, leading=22,
    textColor=NAVY, spaceBefore=0, spaceAfter=1,
)
H2 = ParagraphStyle(
    "SectionTitle", fontName="Arial-Bold", fontSize=11.8, leading=14,
    textColor=DEEP_INDIGO, spaceBefore=6.5, spaceAfter=2, keepWithNext=1,
)
H3 = ParagraphStyle(
    "SubsectionTitle", fontName="Arial-Bold", fontSize=10.2, leading=12.4,
    textColor=INDIGO, spaceBefore=5, spaceAfter=1.5, keepWithNext=1,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY_STYLE, leftIndent=11, bulletIndent=2,
    spaceAfter=1.2, alignment=TA_LEFT,
)
QA = ParagraphStyle(
    "QA", parent=BODY_STYLE, spaceBefore=3.2, spaceAfter=0.8,
)
CAPTION = ParagraphStyle(
    "Caption", fontName="Arial-Italic", fontSize=7.7, leading=9.7,
    textColor=MUTED, alignment=TA_CENTER, spaceBefore=1.5, spaceAfter=4,
)
CALLOUT = ParagraphStyle(
    "Callout", fontName="Arial", fontSize=9.4, leading=12.7,
    textColor=INK, alignment=TA_LEFT,
)
CODE = ParagraphStyle(
    "Code", fontName="Mono", fontSize=7.9, leading=10.3, textColor=INK,
)
FORMULA = ParagraphStyle(
    "Formula", fontName="Arial-Italic", fontSize=10.4, leading=13.4,
    textColor=DEEP_INDIGO, alignment=TA_CENTER,
)
TOC_TITLE = ParagraphStyle(
    "TOCTitle", fontName="Arial-Bold", fontSize=23, leading=27, textColor=NAVY,
    spaceAfter=2,
)
TOC_SUB = ParagraphStyle(
    "TOCSub", fontName="Arial", fontSize=9.5, leading=12.4, textColor=MUTED,
    spaceAfter=6,
)

TABLE_BODY = ParagraphStyle(
    "TableBody", fontName="Arial", fontSize=8.05, leading=10.0,
    textColor=INK, alignment=TA_LEFT,
)
TABLE_HEAD = ParagraphStyle(
    "TableHead", fontName="Arial-Bold", fontSize=8.0, leading=9.8,
    textColor=colors.white,
)
TABLE_NUM = ParagraphStyle(
    "TableNum", parent=TABLE_BODY, alignment=TA_RIGHT,
)

# ---------------------------------------------------------------------------
# Inline Markdown
# ---------------------------------------------------------------------------
_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def inline(text: str) -> str:
    """Convert a small, safe subset of Markdown to ReportLab markup."""
    placeholders = {}

    def keep_code(match):
        key = f"@@CODE{len(placeholders)}@@"
        code = escape(match.group(1))
        placeholders[key] = (
            f'<font name="Mono" size="8.5" color="#312E81">{code}</font>'
        )
        return key

    text = _CODE_RE.sub(keep_code, text)
    text = escape(text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    # Markdown links: keep the visible text only.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def plain(text: str) -> str:
    return re.sub(r"[*`]", "", text)


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------
def parse_markdown(path: str):
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()

    # Strip YAML front matter.
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            raw = parts[2]

    lines = raw.splitlines()
    blocks = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "\\newpage":
            blocks.append(("pagebreak", None))
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", "\n".join(code_lines)))
            continue

        if stripped.startswith(":::cols"):
            i += 1
            images = []
            while i < n and lines[i].strip() != ":::":
                match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", lines[i].strip())
                if match:
                    images.append((match.group(1), match.group(2)))
                i += 1
            i += 1
            blocks.append(("cols", images))
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            blocks.append((f"h{min(level, 3)}", text))
            i += 1
            continue

        if stripped.startswith(">"):
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            blocks.append(("callout", " ".join(quote)))
            continue

        if stripped.startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                if not re.match(r"^\|[\s:|-]+\|$", row):
                    cells = [cell.strip() for cell in row.strip("|").split("|")]
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue

        match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)(?:\{width=([0-9.]+)%\})?$", stripped)
        if match:
            blocks.append(
                ("image", (match.group(1), match.group(2),
                           float(match.group(3)) if match.group(3) else 100.0))
            )
            i += 1
            continue

        if stripped.startswith("$$") and stripped.endswith("$$"):
            blocks.append(("formula", stripped.strip("$").strip()))
            i += 1
            continue

        if stripped.startswith("- "):
            items = []
            while i < n and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append(("bullets", items))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            blocks.append(("numbers", items))
            continue

        # Paragraph: collect until blank or a new block marker.
        para = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if not nxt:
                break
            if (
                nxt.startswith(("#", ">", "|", "```", ":::", "![", "$$", "- "))
                or re.match(r"^\d+\.\s+", nxt)
                or nxt == "\\newpage"
            ):
                break
            para.append(nxt)
            i += 1
        blocks.append(("paragraph", " ".join(para)))

    return blocks


# ---------------------------------------------------------------------------
# Flowable helpers
# ---------------------------------------------------------------------------
def _callout_table(text: str):
    paragraph = Paragraph(inline(text), CALLOUT)
    table = Table([[paragraph]], colWidths=[CONTENT_W])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LAVENDER),
                ("LINEBEFORE", (0, 0), (0, -1), 3, INDIGO),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def callout_flowable(text: str):
    # A single table cell cannot split across pages. Long quotes (for example
    # the live-demo script) are therefore split into sentence-aligned bands so
    # they flow naturally without leaving a blank half-page.
    if len(text) <= 620:
        return [Spacer(1, 2), _callout_table(text), Spacer(1, 6)]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > 520:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)

    out = [Spacer(1, 2)]
    for index, chunk in enumerate(chunks):
        out.append(_callout_table(chunk))
        if index < len(chunks) - 1:
            out.append(Spacer(1, 1.5))
    out.append(Spacer(1, 6))
    return out


def code_flowable(text: str):
    body = XPreformatted(escape(text), CODE)
    table = Table([[body]], colWidths=[CONTENT_W])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, PALE_INDIGO),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return [Spacer(1, 2), table, Spacer(1, 8)]


def formula_flowable(text: str):
    body = Paragraph(inline(text), FORMULA)
    table = Table([[body]], colWidths=[CONTENT_W])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return [Spacer(1, 2), table, Spacer(1, 8)]


def table_flowable(rows):
    if not rows:
        return []
    ncols = len(rows[0])
    rows = [row + [""] * (ncols - len(row)) for row in rows]
    font_size = TABLE_BODY.fontSize

    def text_width(value):
        return pdfmetrics.stringWidth(plain(value), "Arial", font_size)

    def longest_word(value):
        words = re.split(r"\s+", plain(value).strip())
        return max((text_width(word) for word in words), default=0)

    widths = []
    for col in range(ncols):
        values = [plain(row[col]) for row in rows]
        header_w = text_width(values[0]) + 8
        word_w = max(longest_word(value) for value in values) + 9
        body = values[1:] or values
        average_w = sum(text_width(value) for value in body) / len(body)
        desired = max(header_w, word_w, min(average_w * 0.55, 52 * mm))
        desired = min(desired, 58 * mm)
        widths.append(max(desired, 10 * mm))

    scale = CONTENT_W / sum(widths)
    widths = [width * scale for width in widths]
    # Guarantee enough room for the longest unbreakable token in each column.
    for _ in range(4):
        deficits = []
        for col, width in enumerate(widths):
            need = max(longest_word(row[col]) for row in rows) + 9
            deficits.append(max(0, min(need, 62 * mm) - width))
        if sum(deficits) <= 0.5:
            break
        flex = [max(0, width - 9 * mm) for width in widths]
        flex_total = sum(flex)
        if flex_total <= 0:
            break
        widths = [
            width + sum(deficits) * (flex_i / flex_total)
            for width, flex_i in zip(widths, flex)
        ]
        scale = CONTENT_W / sum(widths)
        widths = [width * scale for width in widths]
    widths = [width * CONTENT_W / sum(widths) for width in widths]

    data = []
    for row_index, row in enumerate(rows):
        cells = []
        for col_index, cell in enumerate(row):
            style = TABLE_HEAD if row_index == 0 else TABLE_BODY
            if row_index > 0 and re.match(r"^[-+]?[0-9.,]+$", plain(cell)):
                style = TABLE_NUM if row_index > 0 else style
            cells.append(Paragraph(inline(cell), style))
        data.append(cells)

    table = Table(
        data,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), DEEP_INDIGO),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.2),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, BORDER),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            style.append(("BACKGROUND", (0, row_index), (-1, row_index), SOFT))
    table.setStyle(TableStyle(style))
    return [Spacer(1, 1.5), table, Spacer(1, 6)]


def image_flowable(caption: str, path: str, width_pct: float = 100.0):
    if not os.path.isabs(path):
        path = os.path.join(BASE, path)
    with PILImage.open(path) as handle:
        px_w, px_h = handle.size
    max_w = CONTENT_W * width_pct / 100.0
    max_h = 105 * mm
    scale = min(max_w / px_w, max_h / px_h)
    draw_w, draw_h = px_w * scale, px_h * scale
    image = Image(path, width=draw_w, height=draw_h)
    image.hAlign = "CENTER"
    caption_flow = Paragraph(inline(caption), CAPTION)
    return [KeepTogether([Spacer(1, 1.5), image, caption_flow])]


def cols_flowable(images):
    if not images:
        return []
    gap = 4.5 * mm
    per_row = 2 if len(images) > 1 else 1
    cell_w = (CONTENT_W - gap * (per_row - 1)) / per_row
    rows = []
    for start in range(0, len(images), per_row):
        cells = []
        for caption, path in images[start:start + per_row]:
            if not os.path.isabs(path):
                path = os.path.join(BASE, path)
            with PILImage.open(path) as handle:
                px_w, px_h = handle.size
            max_h = 59 * mm
            scale = min(cell_w / px_w, max_h / px_h)
            image = Image(path, width=px_w * scale, height=px_h * scale)
            image.hAlign = "CENTER"
            cells.append([image, Paragraph(inline(caption), CAPTION)])
        while len(cells) < per_row:
            cells.append([Spacer(1, 1), Paragraph("", CAPTION)])
        rows.append(cells)
    table = Table(rows, colWidths=[cell_w] * per_row, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), gap / 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), gap / 2),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return [Spacer(1, 1.5), table, Spacer(1, 4)]


# ---------------------------------------------------------------------------
# Document template
# ---------------------------------------------------------------------------
class GuideDoc(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        self.chapter_count = 0
        self.current_chapter = "ABSA Studio"

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "ChapterTitle":
            number = getattr(flowable, "_chapter_number", None)
            title = flowable.getPlainText()
            if number is None:
                self.chapter_count += 1
                number = self.chapter_count
            self.current_chapter = title
            key = f"chapter-{int(number):02d}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=0, closed=False)
            self.notify(
                "TOCEntry",
                (0, f"{int(number)}   {title}", self.page),
            )


def draw_cover(c: pdfcanvas.Canvas, doc):
    width, height = A4
    c.saveState()
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    band_bottom = height - 122 * mm
    c.linearGradient(
        0,
        band_bottom,
        width,
        height,
        [colors.HexColor("#0B1220"), colors.HexColor("#312E81"), colors.HexColor("#6D28D9")],
        positions=[0.0, 0.58, 1.0],
    )

    c.setFillColor(colors.white)
    c.setFillAlpha(0.055)
    c.circle(width - 12 * mm, height - 18 * mm, 52 * mm, stroke=0, fill=1)
    c.circle(width - 2 * mm, height - 92 * mm, 40 * mm, stroke=0, fill=1)
    c.circle(18 * mm, height - 110 * mm, 26 * mm, stroke=0, fill=1)
    c.setFillAlpha(1)

    x = 19 * mm
    c.setFillColor(colors.HexColor("#C7D2FE"))
    c.setFont("Arial-Bold", 8.3)
    c.drawString(
        x,
        height - 36 * mm,
        "NLP COURSEWORK PROJECT  ·  STUDY & PRESENTATION GUIDE",
        charSpace=1.9,
    )

    c.setFillColor(colors.white)
    c.setFont("Arial-Bold", 40)
    c.drawString(x, height - 55 * mm, "ABSA STUDIO")

    c.setStrokeColor(colors.HexColor("#A5B4FC"))
    c.setLineWidth(1.2)
    c.line(x, height - 61.5 * mm, x + 34 * mm, height - 61.5 * mm)

    c.setFillColor(colors.HexColor("#E0E7FF"))
    c.setFont("Arial", 19.5)
    c.drawString(x, height - 75 * mm, "Complete Project Documentation")
    c.drawString(x, height - 85 * mm, "& Presentation Guide")

    c.setFillColor(colors.HexColor("#C7D2FE"))
    c.setFont("Arial", 9.6)
    c.drawString(
        x,
        height - 97 * mm,
        "Aspect-Based Sentiment Analysis of Restaurant Reviews",
    )
    c.setFont("Arial", 8.7)
    c.drawString(
        x,
        height - 104 * mm,
        "Architecture · NLP pipeline · algorithms · implementation · evaluation · practical demo",
    )

    # Technology chips
    chips = ["SemEval-2014", "pLSA + TF-ICF", "TF-IDF · LR/SVM", "GloVe BiLSTM", "Streamlit"]
    chip_y = height - 120 * mm
    c.setFont("Arial-Bold", 7.8)
    chip_x = x
    for chip in chips:
        text_w = pdfmetrics.stringWidth(chip, "Arial-Bold", 7.8)
        w = text_w + 10 * mm
        c.setFillColor(colors.HexColor("#EEF2FF"))
        c.roundRect(chip_x, chip_y, w, 6.6 * mm, 3.3 * mm, stroke=0, fill=1)
        c.setFillColor(INDIGO)
        c.drawCentredString(chip_x + w / 2, chip_y + 2.2 * mm, chip)
        chip_x += w + 3 * mm

    # Body section
    c.setFillColor(colors.HexColor("#E0E7FF"))
    c.setFont("Arial-Bold", 9.8)
    c.drawString(x, height - 138 * mm, "INSIDE THIS GUIDE")
    c.setStrokeColor(colors.HexColor("#4338CA"))
    c.setLineWidth(0.7)
    c.line(x, height - 141 * mm, width - x, height - 141 * mm)

    cards = [
        ("PROJECT", "ABSA Studio — aspect-based sentiment analysis of restaurant reviews."),
        ("DATASET", "SemEval-2014 Task 4 · 3,041 train + 800 test sentences · 5 aspects."),
        ("MODELS", "pLSA + TF-ICF · TF-IDF logistic regression/SVM · GloVe BiLSTM."),
        ("EVIDENCE", "0.835 aspect micro-F1 · 0.665 sentiment macro-F1 · 19 tests."),
    ]
    card_w = (width - 2 * x - 5 * mm) / 2
    card_h = 24 * mm
    for index, (title, body) in enumerate(cards):
        col = index % 2
        row = index // 2
        cx = x + col * (card_w + 5 * mm)
        cy = height - (147 * mm) - row * (card_h + 3.5 * mm)
        c.setFillColor(SOFT)
        c.setStrokeColor(BORDER)
        c.setLineWidth(0.6)
        c.roundRect(cx, cy - card_h, card_w, card_h, 2.6 * mm, stroke=1, fill=1)
        c.setFillColor(INDIGO)
        c.setFont("Arial-Bold", 7.2)
        c.drawString(cx + 5 * mm, cy - 6.8 * mm, title)
        c.setFillColor(BODY)
        c.setFont("Arial", 8.3)
        words = body.split()
        lines, line = [], ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if pdfmetrics.stringWidth(candidate, "Arial", 8.3) <= card_w - 10 * mm:
                line = candidate
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        for line_index, text in enumerate(lines[:2]):
            c.drawString(cx + 5 * mm, cy - 13.5 * mm - line_index * 4.0 * mm, text)

    # Bottom statistics
    stats = [("12", "chapters"), ("30", "viva questions"), ("35–40", "minutes of study")]
    col_w = (width - 2 * x) / 3
    for index, (value, label) in enumerate(stats):
        sx = x + index * col_w
        c.setFillColor(colors.white)
        c.setFont("Arial-Bold", 17)
        c.drawString(sx, 78 * mm, value)
        c.setFillColor(colors.HexColor("#C7D2FE"))
        c.setFont("Arial", 8.4)
        c.drawString(sx, 72 * mm, label)

    c.setStrokeColor(colors.HexColor("#4338CA"))
    c.setLineWidth(0.7)
    c.line(x, 60 * mm, width - x, 60 * mm)
    c.setFillColor(colors.HexColor("#A5B4FC"))
    c.setFont("Arial", 7.4)
    c.drawString(
        x,
        51 * mm,
        "Built from repository source, saved models and evaluation artefacts · November 2026",
    )
    c.drawRightString(
        width - x,
        43 * mm,
        "Project presentation & viva preparation",
    )
    c.restoreState()


def draw_header_footer(c: pdfcanvas.Canvas, doc):
    if doc.page == 1:
        return
    c.saveState()
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(LM, PAGE_H - 13.5 * mm, PAGE_W - RM, PAGE_H - 13.5 * mm)
    c.setFillColor(MUTED)
    c.setFont("Arial-Bold", 7.0)
    c.drawString(
        LM,
        PAGE_H - 11.5 * mm,
        "ABSA STUDIO  ·  PROJECT GUIDE",
        charSpace=1.2,
    )
    c.setFont("Arial", 7.2)
    c.drawRightString(
        PAGE_W - RM,
        PAGE_H - 11.5 * mm,
        "NLP coursework · SemEval-2014 restaurant domain",
    )

    c.setStrokeColor(BORDER)
    c.line(LM, 12.5 * mm, PAGE_W - RM, 12.5 * mm)
    c.setFillColor(MUTED)
    c.setFont("Arial", 7.2)
    c.drawString(LM, 8.8 * mm, "Aspect-Based Sentiment Analysis of Restaurant Reviews")
    c.setFont("Arial-Bold", 7.6)
    c.drawRightString(PAGE_W - RM, 8.8 * mm, f"Page {doc.page}")
    c.restoreState()


# ---------------------------------------------------------------------------
# Story construction
# ---------------------------------------------------------------------------
def chapter_opener(text: str):
    match = re.match(r"Chapter\s+(\d+)\s+[—-]\s+(.+)", text)
    if match:
        number = int(match.group(1))
        title = match.group(2)
        kicker = Paragraph(f"CHAPTER {number:02d}", H1_KICKER)
        heading = Paragraph(escape(title), H1)
        heading._toc_title = title
        heading._chapter_number = number
    else:
        kicker = Paragraph("", H1_KICKER)
        heading = Paragraph(inline(text), H1)
    rule = HRFlowable(
        width="100%", thickness=1.0, color=PALE_INDIGO,
        spaceBefore=1, spaceAfter=8,
    )
    return [kicker, heading, rule]


def build_story(blocks):
    story = []
    after_heading = False

    for kind, value in blocks:
        if kind == "h1":
            story.append(CondPageBreak(90 * mm))
            if story:
                story.append(Spacer(1, 6))
            story.extend(chapter_opener(value))
            after_heading = True
        elif kind == "h2":
            story.append(Paragraph(inline(value), H2))
            after_heading = False
        elif kind == "h3":
            story.append(Paragraph(inline(value), H3))
            after_heading = False
        elif kind == "paragraph":
            style = LEAD_STYLE if after_heading else (
                QA if value.startswith("**Q") else BODY_STYLE
            )
            story.append(Paragraph(inline(value), style))
            after_heading = False
        elif kind == "bullets":
            for item in value:
                story.append(
                    Paragraph(inline(item), BULLET, bulletText="•")
                )
            story.append(Spacer(1, 3))
            after_heading = False
        elif kind == "numbers":
            for index, item in enumerate(value, start=1):
                story.append(
                    Paragraph(inline(item), BULLET, bulletText=f"{index}.")
                )
            story.append(Spacer(1, 3))
            after_heading = False
        elif kind == "callout":
            story.extend(callout_flowable(value))
        elif kind == "table":
            story.extend(table_flowable(value))
        elif kind == "code":
            story.extend(code_flowable(value))
        elif kind == "formula":
            story.extend(formula_flowable(value))
        elif kind == "image":
            caption, path, pct = value
            story.extend(image_flowable(caption, path, pct))
        elif kind == "cols":
            story.extend(cols_flowable(value))
        elif kind == "pagebreak":
            story.append(PageBreak())
    return story


def toc_page(blocks, story):
    story.append(Spacer(1, 2))
    story.append(Paragraph("Contents", TOC_TITLE))
    story.append(
        Paragraph(
            "Twelve chapters, one practical demonstration and thirty viva questions — "
            "designed to be read once and revised quickly.",
            TOC_SUB,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=PALE_INDIGO,
                            spaceBefore=0, spaceAfter=8))

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCEntry", fontName="Arial", fontSize=10.6, leading=17,
            textColor=NAVY, leftIndent=0, firstLineIndent=0,
            spaceBefore=1.2, spaceAfter=1.2,
        )
    ]
    toc.dotsMinLevel = 0
    story.append(toc)

    story.append(Spacer(1, 10))
    story.extend(
        callout_flowable(
            "**How to use this guide.** Read Chapters 1–4 for the conceptual frame, "
            "Chapters 5–7 for the implementation, and Chapters 8–9 for the demo and "
            "evidence. Chapter 10 is the script to follow live; Chapter 11 is for rapid "
            "revision before the viva. Every number and file name in this guide was "
            "checked against the repository."
        )
    )

    story.append(Paragraph("Project at a glance", H2))
    rows = [
        ["Item", "What it is in this repository"],
        ["Task", "Aspect detection + per-aspect 3-class sentiment (English restaurant reviews)"],
        ["Dataset", "SemEval-2014 Task 4 · 3,041 train / 800 test · 5 aspects"],
        ["Aspect detection", "TF-IDF + one-vs-rest LR (supervised) · pLSA + TF-ICF (unsupervised)"],
        ["Sentiment", "Per-aspect TF-IDF + LR, calibrated SVM, GloVe-initialised BiLSTM"],
        ["Best results", "Aspect micro-F1 0.835 · sentiment macro-F1 0.665 (TF-IDF + LR)"],
        ["Interface", "Six-page Streamlit app · engine cached once · no retraining"],
        ["Quality", "19 passing tests · reproducible scripts · saved artefacts"],
    ]
    story.extend(table_flowable(rows))

    story.append(PageBreak())


def main():
    blocks = parse_markdown(MD_PATH)

    doc = GuideDoc(
        PDF_PATH,
        pagesize=A4,
        leftMargin=LM,
        rightMargin=RM,
        topMargin=TM,
        bottomMargin=BM,
        title="ABSA Studio: Complete Project Documentation & Presentation Guide",
        author="ABSA Studio",
        subject="Aspect-Based Sentiment Analysis of Restaurant Reviews",
        creator="ReportLab",
    )

    frame = Frame(
        LM, BM, CONTENT_W, CONTENT_H,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="body",
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[frame], onPage=draw_cover),
            PageTemplate(id="body", frames=[frame], onPage=draw_header_footer),
        ]
    )

    story = [Spacer(1, 1), NextPageTemplate("body"), PageBreak()]
    toc_page(blocks, story)
    story.extend(build_story(blocks))
    while story and isinstance(story[-1], Spacer):
        story.pop()

    doc.multiBuild(story)
    print("Wrote", PDF_PATH)


if __name__ == "__main__":
    main()
