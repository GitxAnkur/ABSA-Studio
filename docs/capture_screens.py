"""Capture annotated screenshots from the running Streamlit app.

The app must be running on http://localhost:8501.  Screenshots are taken at
2x device scale so the PDF stays crisp when the image is reduced to page width.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE, "assets")
os.makedirs(ASSETS, exist_ok=True)

URL = "http://localhost:8501"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def add_markers(image_path: str, markers, output_path: str) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 21)
    for number, (x, y) in markers:
        x, y = x * 2, y * 2
        draw.ellipse((x - 21, y - 21, x + 21, y + 21), fill="white")
        draw.ellipse((x - 17, y - 17, x + 17, y + 17), fill="#4338CA")
        text = str(number)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text(
            (x - (right - left) / 2, y - (bottom - top) / 2 - 2),
            text,
            font=font,
            fill="white",
        )
    image.save(output_path, optimize=True)


def rel(box, clip):
    return box["x"] - clip["x"], box["y"] - clip["y"]


def safe_box(locator):
    try:
        if locator.count() == 0:
            return None
        return locator.first.bounding_box()
    except Exception:
        return None


def markers_for(boxes, clip):
    points = []
    for box in boxes:
        if not box:
            continue
        x, y = rel(box, clip)
        if x > clip["width"] - 40 or y > clip["height"] - 40:
            continue
        px = max(12, x - 24)
        py = max(12, y + 22)
        points.append((px, py))
    return [(index + 1, point) for index, point in enumerate(points)]


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(
            viewport={"width": 1440, "height": 1300}, device_scale_factor=2
        )
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(4500)
        main = page.locator('[data-testid="stMain"]').bounding_box()

        # --- Overview -----------------------------------------------------
        clip = {"x": main["x"], "y": 20, "width": main["width"], "height": 600}
        raw = os.path.join(ASSETS, "_raw_overview.png")
        page.screenshot(path=raw, clip=clip)
        boxes = [
            safe_box(page.locator(".hero")),
            safe_box(page.locator(".kpi-card")),
        ]
        add_markers(raw, markers_for(boxes, clip),
                    os.path.join(ASSETS, "fig_ui_overview.png"))

        # --- Analyse Review ----------------------------------------------
        page.locator(
            '[data-testid="stSidebar"] label[data-testid="stRadioOption"]',
            has=page.locator("p", has_text="Analyse Review"),
        ).first.click()
        page.wait_for_timeout(3000)
        page.get_by_role("button", name="Analyse review").first.click()
        page.wait_for_timeout(6500)
        page.evaluate(
            "document.querySelector('main[data-testid=\"stMain\"]').scrollTo(0, 0)"
        )
        page.wait_for_timeout(700)

        clip2 = {"x": main["x"], "y": 45, "width": main["width"], "height": 1170}
        raw = os.path.join(ASSETS, "_raw_analyse.png")
        page.screenshot(path=raw, clip=clip2)
        boxes = [
            safe_box(page.locator(".verdict-card")),
            safe_box(page.locator(".aspect-card")),
            safe_box(page.locator(".panel-title")),
        ]
        add_markers(raw, markers_for(boxes, clip2),
                    os.path.join(ASSETS, "fig_ui_analyse.png"))

        # --- Model Performance -------------------------------------------
        page.locator(
            '[data-testid="stSidebar"] label[data-testid="stRadioOption"]',
            has=page.locator("p", has_text="Model Performance"),
        ).first.click()
        page.wait_for_timeout(3500)
        page.evaluate(
            "document.querySelector('main[data-testid=\"stMain\"]').scrollTo(0, 0)"
        )
        page.wait_for_timeout(700)

        clip3 = {"x": main["x"], "y": 60, "width": main["width"], "height": 640}
        raw = os.path.join(ASSETS, "_raw_performance.png")
        page.screenshot(path=raw, clip=clip3)
        boxes = [
            safe_box(page.locator(".kpi-card")),
            safe_box(page.locator(".table-wrap")),
            safe_box(page.locator("canvas")),
        ]
        add_markers(raw, markers_for(boxes, clip3),
                    os.path.join(ASSETS, "fig_ui_performance.png"))

        browser.close()

    for name in ("_raw_overview.png", "_raw_analyse.png", "_raw_performance.png"):
        path = os.path.join(ASSETS, name)
        if os.path.exists(path):
            os.remove(path)
    print("Screenshots written to", ASSETS)


if __name__ == "__main__":
    run()
