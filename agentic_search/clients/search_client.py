from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError

from agentic_search.exceptions import ApiError
from agentic_search.utils.text_utils import require_env

# SerpAPI configuration
SERP_API_KEY = os.getenv("SERP_API_KEY", "")
SERP_API_BASE = "https://serpapi.com/search"

# Images saved under ./temp by default
TEMP_IMAGE_DIR = os.path.abspath("./temp")

SUPPORTED_RASTER_MIME = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
    "image/bmp",
}

UNSTABLE_OR_UNSUPPORTED_MIME = {
    "image/avif",
    "image/svg+xml",
}


def _serpapi_request(params: dict) -> dict:
    """
    Send a request to the SerpAPI endpoint.
    All SerpAPI calls share the same base URL; the `engine` parameter
    selects the backend (google, google_images, google_lens, etc.).
    """
    api_key = require_env("SERP_API_KEY", SERP_API_KEY)
    params["api_key"] = api_key
    params.setdefault("source", "python")

    resp = requests.get(SERP_API_BASE, params=params, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise ApiError(f"SerpAPI HTTP error: {e}, body={resp.text}") from e

    data = resp.json()

    # SerpAPI returns an "error" field when something goes wrong
    if "error" in data:
        raise ApiError(f"SerpAPI error: {data['error']}")

    return data


def first_nonempty(d: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = d.get(key)
        if value not in (None, "", []):
            return str(value).strip()
    return ""


WEB_URL_KEYS = ["url", "link", "source_page", "sourceUrl", "pageUrl", "href"]
TITLE_KEYS = ["title", "name", "page_title"]
IMAGE_KEYS = ["image_url", "imageUrl", "image", "original", "url"]


def _iter_dict_nodes(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_dict_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_dict_nodes(item)


def normalize_image_results(data: Dict[str, Any], limit: int = 8) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()

    for node in _iter_dict_nodes(data):
        image_url = first_nonempty(node, IMAGE_KEYS)
        title = first_nonempty(node, TITLE_KEYS)
        page_url = first_nonempty(node, WEB_URL_KEYS)

        if not image_url:
            continue
        if not (image_url.startswith("http://") or image_url.startswith("https://")):
            continue
        if image_url in seen:
            continue

        seen.add(image_url)
        out.append(
            {
                "title": title,
                "image_url": image_url,
                "page_url": page_url,
            }
        )
        if len(out) >= limit:
            break

    return out


def _request_headers(image_url: str) -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36 "
            "AgenticSearchImageFetcher/1.0"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Referer": image_url,
        "Connection": "close",
    }


def _sniff_mime(content: bytes) -> str:
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):
        return "image/gif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    if content.startswith(b"BM"):
        return "image/bmp"
    if content.startswith(b"<svg") or b"<svg" in content[:256].lower():
        return "image/svg+xml"
    return "application/octet-stream"


def _extension_from_mime(mime_type: str) -> str:
    mime_type = (mime_type or "").split(";")[0].strip().lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/avif": ".avif",
        "image/svg+xml": ".svg",
    }
    return mapping.get(mime_type) or mimetypes.guess_extension(mime_type) or ".jpg"


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _validate_with_pil(local_path: str) -> None:
    with Image.open(local_path) as im:
        im.verify()


def _looks_like_login_gate(title: str, content: str, final_url: str) -> bool:
    s = f"{title}\n{content}\n{final_url}".lower()
    bad_patterns = [
        "log in",
        "login",
        "sign up",
        "create new account",
        "see instagram photos and videos",
        "facebook helps you connect and share",
    ]
    return any(p in s for p in bad_patterns)


def save_image_via_browser(
    image_url: str,
    save_dir: str = TEMP_IMAGE_DIR,
    timeout_ms: int = 30000,
) -> str:
    """
    Open URL in browser and take a screenshot to save as PNG.
    Suitable for AVIF / SVG / unstable direct downloads.
    """
    from playwright.sync_api import sync_playwright

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(save_dir, f"{_sha1(image_url)}.png")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1600})
        try:
            page.goto(image_url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(1200)

            title = page.title() or ""
            final_url = page.url or ""
            content = page.content() or ""

            if _looks_like_login_gate(title, content, final_url):
                raise ValueError(f"browser fallback reached login/sign-up gate: {final_url}")

            page.screenshot(path=out_path, full_page=True)
        finally:
            browser.close()

    _validate_with_pil(out_path)
    return out_path


def download_single_image(
    image_url: str,
    save_dir: str = TEMP_IMAGE_DIR,
    connect_timeout: float = 10.0,
    read_timeout: float = 20.0,
    max_bytes: int = 20 * 1024 * 1024,
    max_retries: int = 3,
    allow_browser_fallback: bool = True,
) -> str:
    """
    Download a single image_url.
    - Normal public image: requests download
    - AVIF / SVG / PIL cannot open: optional browser screenshot fallback -> PNG
    - Browser fallback at most once
    """
    if not image_url or not (image_url.startswith("http://") or image_url.startswith("https://")):
        raise ValueError(f"invalid image_url: {image_url}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    last_err = None
    data = b""
    content_type = ""

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(
                image_url,
                headers=_request_headers(image_url),
                stream=True,
                timeout=(connect_timeout, read_timeout),
                allow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()

                chunks: List[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"image too large: {total} bytes")
                    chunks.append(chunk)

            data = b"".join(chunks)
            break

        except ValueError:
            raise
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < max_retries:
                sleep_s = 2 ** (attempt - 1)
                print(
                    f"[download_single_image retry {attempt}/{max_retries}] "
                    f"url={image_url} error={repr(e)} sleep={sleep_s}s",
                    flush=True,
                )
                time.sleep(sleep_s)
            else:
                data = b""
                break

    # requests completely failed: optional browser screenshot fallback
    if not data:
        if allow_browser_fallback:
            return save_image_via_browser(image_url=image_url, save_dir=save_dir)
        raise last_err or ValueError(f"failed to download image: {image_url}")

    sniffed = _sniff_mime(data[:256])
    mime_type = sniffed if sniffed.startswith("image/") else content_type

    # HTML / non-image: optional browser screenshot
    if not mime_type.startswith("image/"):
        if allow_browser_fallback:
            return save_image_via_browser(image_url=image_url, save_dir=save_dir)
        raise ValueError(f"not an image payload: {mime_type}")

    # AVIF / SVG: use browser screenshot instead, output png
    if mime_type in UNSTABLE_OR_UNSUPPORTED_MIME:
        if allow_browser_fallback:
            return save_image_via_browser(image_url=image_url, save_dir=save_dir)
        raise ValueError(f"unsupported image mime for direct use: {mime_type}")

    # Normal raster image: save as original format
    url_hash = _sha1(image_url)
    ext = _extension_from_mime(mime_type)
    local_path = os.path.join(save_dir, f"{url_hash}{ext}")

    if not (os.path.exists(local_path) and os.path.getsize(local_path) > 0):
        with open(local_path, "wb") as f:
            f.write(data)

    # PIL validation failed: allow at most one browser fallback
    try:
        _validate_with_pil(local_path)
        return local_path
    except (UnidentifiedImageError, OSError) as e:
        if allow_browser_fallback:
            return save_image_via_browser(image_url=image_url, save_dir=save_dir)
        raise e

def images_search(
    query: str,
    num: int = 10,
    country: str = "us",
    locale: str = "en",
    location: str = "United States",
    page: int = 1,
) -> dict:
    """
    Search for images using SerpAPI Google Images engine.

    Returns:
    - raw_search_result: raw JSON from SerpAPI
    - normalized_results: normalized top-k results
    - downloaded_images: successfully downloaded images
    - download_errors: download failure details
    """
    params = {
        "engine": "google_images",
        "q": query,
        "num": num,
        "gl": country,
        "hl": locale,
        "location": location,
        "start": (page - 1) * num + 1 if page > 1 else 0,
    }

    raw = _serpapi_request(params)
    candidates = normalize_image_results(raw, limit=num)

    downloaded_images: List[Dict[str, Any]] = []
    download_errors: List[Dict[str, Any]] = []

    for rank, cand in enumerate(candidates[:num], start=1):
        image_url = cand.get("image_url", "")
        try:
            local_path = download_single_image(
                image_url=image_url,
                save_dir=TEMP_IMAGE_DIR,
                allow_browser_fallback=True,
            )
            downloaded_images.append(
                {
                    "rank": rank,
                    "title": cand.get("title", ""),
                    "page_url": cand.get("page_url", ""),
                    "remote_url": image_url,
                    "local_path": local_path,
                }
            )
        except Exception as e:
            download_errors.append(
                {
                    "rank": rank,
                    "title": cand.get("title", ""),
                    "page_url": cand.get("page_url", ""),
                    "remote_url": image_url,
                    "error": repr(e),
                }
            )

    return {
        "query": query,
        "requested_top_k": num,
        "raw_search_result": raw,
        "normalized_results": candidates[:num],
        "downloaded_images": downloaded_images,
        "download_errors": download_errors,
    }


def lens_search(image_url: str, hl: str = "en", gl: str = "us") -> dict:
    """
    Reverse image search using SerpAPI Google Lens engine.

    Args:
        image_url: Publicly accessible URL of the image to search with.
        hl: Language code (default: "en").
        gl: Country code (default: "us").

    Returns:
        Raw SerpAPI response with lens results.
    """
    params = {
        "engine": "google_lens",
        "url": image_url,
        "hl": hl,
        "gl": gl,
    }
    return _serpapi_request(params)


def web_search(
    query: str,
    num: int = 10,
    country: str = "us",
    locale: str = "en",
    location: str = "United States",
    page: int = 1,
) -> dict:
    """
    Web search using SerpAPI Google Search engine.

    Args:
        query: Search query string.
        num: Number of results (max 10 for organic results per page on SerpAPI).
        country: Country code for gl parameter.
        locale: Language code for hl parameter.
        location: Location string (e.g. "United States").
        page: 1-based page number.

    Returns:
        Raw SerpAPI response containing organic results, knowledge graph, etc.
    """
    params = {
        "engine": "google",
        "q": query,
        "num": num,
        "gl": country,
        "hl": locale,
        "location": location,
        "start": (page - 1) * num + 1 if page > 1 else 0,
    }
    return _serpapi_request(params)


def fetch_webpage_text(url: str, max_body_chars: int = 1200) -> Tuple[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    main_node = soup.find("article") or soup.find("main") or soup.find("body")
    body_text = main_node.get_text(separator="\n", strip=True) if main_node else ""
    body_text = re.sub(r"\n{2,}", "\n", body_text)
    body_text = re.sub(r"[ \t]{2,}", " ", body_text).strip()
    return title, body_text[:max_body_chars]


def t2t_from_webpage(url: str, max_body_chars: int = 1200, timeout: int = 20) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    main_node = soup.find("article") or soup.find("main") or soup.find("body")
    body_text = main_node.get_text(separator="\n", strip=True) if main_node else ""
    body_text = re.sub(r"\n{2,}", "\n", body_text)
    body_text = re.sub(r"[ \t]{2,}", " ", body_text).strip()
    return f"{title}\n\n{body_text[:max_body_chars]}".strip()
