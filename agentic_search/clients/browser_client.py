from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image
TEMP_DIR = os.path.abspath("./temp")


def _safe_name(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return h


def _cap_long_image(path: str, max_ratio: float = 10.0) -> Tuple[Tuple[int, int], Tuple[int, int], bool]:
    """
    If long side > short side * max_ratio, crop the excess.
    Vertical image: keep top region
    Horizontal image: keep left region
    Returns: (orig_size, new_size, truncated)
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        orig = (w, h)

        short_side = max(1, min(w, h))
        long_side = max(w, h)
        max_long = int(short_side * max_ratio)

        truncated = False
        if long_side > max_long:
            truncated = True
            if h >= w:
                # Vertical image, crop height
                new_h = max_long
                im = im.crop((0, 0, w, new_h))
            else:
                # Horizontal image, crop width
                new_w = max_long
                im = im.crop((0, 0, new_w, h))

            im.save(path)

        new_size = im.size
        return orig, new_size, truncated


def browse_web_page(
    url: str,
    save_dir: str = TEMP_DIR,
    viewport_width: int = 1440,
    viewport_height: int = 2200,
    timeout_ms: int = 60000,
    max_aspect_ratio: float = 10.0,
) -> Dict:
    """
    Open a webpage in a browser, capture title, and save a full-page screenshot.
    If the screenshot is too long, crop it so the long side is at most
    max_aspect_ratio times the short side.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    base = _safe_name(url)
    screenshot_path = os.path.join(save_dir, f"{base}.png")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise ImportError(
            "browse_web_page requires Playwright. Install it with: "
            "pip install -e .[browser] && python -m playwright install chromium"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            # Try waiting for network idle (max 5s), degrade on timeout
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass  # Timeout degrade, continue to screenshot

            title = page.title() or ""
            final_url = page.url or url

            page.screenshot(path=screenshot_path, full_page=True)

        finally:
            browser.close()

    orig_size, final_size, truncated = _cap_long_image(
        screenshot_path,
        max_ratio=max_aspect_ratio,
    )

    return {
        "title": title,
        "url": url,
        "final_url": final_url,
        "screenshot_path": screenshot_path,
        "orig_image_size": list(orig_size),
        "final_image_size": list(final_size),
        "truncated": truncated,
    }
