from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agentic_search.exceptions import ImagePreparationError
from agentic_search.utils.text_utils import clean_text

IMAGE_HTTP_RETRIES = 2
IMAGE_MAX_BYTES = 20 * 1024 * 1024
IMAGE_CONNECT_TIMEOUT = 10
IMAGE_READ_TIMEOUT = 20
IMAGE_FAST_CONNECT_TIMEOUT = 4
IMAGE_FAST_READ_TIMEOUT = 8
WIKIMEDIA_THUMB_WIDTHS = [256, 512, 1024]
SAVE_IMAGE_DIR = os.getenv("AGENTIC_SEARCH_IMAGE_DIR", "/tmp/agentic_search_images")


def build_resilient_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=IMAGE_HTTP_RETRIES,
        connect=IMAGE_HTTP_RETRIES,
        read=IMAGE_HTTP_RETRIES,
        backoff_factor=1.0,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


IMAGE_HTTP_SESSION = build_resilient_session()


def is_wikimedia_original_upload(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return (
        parsed.netloc.lower() == "upload.wikimedia.org"
        and "/wikipedia/commons/" in parsed.path
        and "/wikipedia/commons/thumb/" not in parsed.path
    )


def wikimedia_thumb_to_original(image_url: str) -> str:
    image_url = clean_text(image_url, 2000)
    if not image_url or "upload.wikimedia.org" not in image_url or "/thumb/" not in image_url:
        return image_url
    try:
        prefix, tail = image_url.split("/thumb/", 1)
        parts = tail.split("/")
        if len(parts) < 4:
            return image_url
        hash1, hash2, filename = parts[0], parts[1], parts[2]
        return f"{prefix}/{hash1}/{hash2}/{filename}"
    except Exception:
        return image_url


def derive_wikimedia_thumbnail_urls(image_url: str, widths: Optional[List[int]] = None) -> List[str]:
    if not is_wikimedia_original_upload(image_url):
        return []
    widths = widths or WIKIMEDIA_THUMB_WIDTHS
    parsed = urlparse(image_url)
    path = parsed.path.strip("/")
    parts = path.split("/")
    if len(parts) < 5 or parts[0] != "wikipedia" or parts[1] != "commons":
        return []
    shard1 = parts[2]
    shard2 = parts[3]
    filename = "/".join(parts[4:])
    filename_leaf = parts[-1]
    out = []
    for w in widths:
        thumb_path = f"/wikipedia/commons/thumb/{shard1}/{shard2}/{filename}/{w}px-{filename_leaf}"
        out.append(f"{parsed.scheme}://{parsed.netloc}{thumb_path}")
    return out


def build_image_url_candidates(image_url: str, thumbnail_url: str = "") -> List[str]:
    urls: List[str] = []
    for u in [image_url, thumbnail_url]:
        u = clean_text(u, 1200)
        if not u:
            continue
        original_u = wikimedia_thumb_to_original(u)
        urls.append(original_u if original_u else u)

    deduped: List[str] = []
    seen = set()
    for u in urls:
        u = clean_text(u, 1200)
        if not u or u in seen:
            continue
        seen.add(u)
        deduped.append(u)
    return deduped


def _normalize_image_url(image_url: str) -> str:
    image_url = clean_text(image_url, 2000)
    if not image_url:
        return ""
    if image_url.startswith("//"):
        return "https:" + image_url
    return image_url


def _sniff_mime_from_bytes(content: bytes) -> str:
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
    return "application/octet-stream"


def _is_image_mime(mime_type: str) -> bool:
    return bool(mime_type) and mime_type.lower().startswith("image/")


def _extension_from_mime(mime_type: str) -> str:
    ext = mimetypes.guess_extension((mime_type or "").split(";")[0].strip())
    return ext or ".jpg"


def _write_bytes_atomically(path: str, data: bytes) -> None:
    tmp_path = path + ".part"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


def _decode_data_url(image_url: str) -> Tuple[bytes, str]:
    m = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", image_url, flags=re.I | re.S)
    if not m:
        raise ValueError("Invalid data URL")
    mime_type = (m.group(1) or "image/jpeg").strip()
    is_b64 = bool(m.group(2))
    payload = m.group(3)
    data = base64.b64decode(payload) if is_b64 else unquote(payload).encode("utf-8")
    return data, mime_type


def _build_request_headers(image_url: str) -> Dict[str, str]:
    parsed = urlparse(image_url)
    referer_host = parsed.scheme + "://" + (parsed.netloc or "")
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": referer_host or "https://www.google.com/",
        "Connection": "close",
    }


def _image_download_timeout_for_url(image_url: str) -> Tuple[int, int]:
    host = (urlparse(image_url).netloc or "").lower()
    if host in {"upload.wikimedia.org", "commons.wikimedia.org", "encrypted-tbn0.gstatic.com", "lookaside.fbsbx.com"}:
        return (IMAGE_FAST_CONNECT_TIMEOUT, IMAGE_FAST_READ_TIMEOUT)
    return (IMAGE_CONNECT_TIMEOUT, IMAGE_READ_TIMEOUT)


def _download_via_requests(
    image_url: str,
    timeout: Tuple[int, int],
    max_bytes: int,
    session: requests.Session,
) -> Tuple[bytes, str]:
    headers = _build_request_headers(image_url)
    with session.get(
        image_url,
        headers=headers,
        stream=True,
        timeout=timeout,
        allow_redirects=True,
    ) as resp:
        resp.raise_for_status()
        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
        chunks: List[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Image too large during download: {image_url}")
            chunks.append(chunk)
        data = b"".join(chunks)

    sniffed = _sniff_mime_from_bytes(data[:32])
    final_mime = sniffed if _is_image_mime(sniffed) else content_type
    if not _is_image_mime(final_mime):
        raise ValueError(f"Downloaded payload is not an image: {image_url}")
    return data, final_mime


def _download_via_urllib(image_url: str, timeout: int, max_bytes: int) -> Tuple[bytes, str]:
    headers = _build_request_headers(image_url)
    req = Request(image_url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"Image too large during download: {image_url}")
        content_type = (resp.headers.get_content_type() or "").strip()
    sniffed = _sniff_mime_from_bytes(data[:32])
    final_mime = sniffed if _is_image_mime(sniffed) else content_type
    if not _is_image_mime(final_mime):
        raise ValueError(f"Downloaded payload is not an image: {image_url}")
    return data, final_mime


def download_image_to_local(
    image_url: str,
    save_dir: str,
    timeout: Optional[Tuple[int, int]] = None,
    max_bytes: int = IMAGE_MAX_BYTES,
    session: Optional[requests.Session] = None,
) -> Tuple[str, str]:
    image_url = _normalize_image_url(image_url)
    if not image_url:
        raise ValueError("Empty image_url")

    os.makedirs(save_dir, exist_ok=True)
    session = session or IMAGE_HTTP_SESSION
    timeout = timeout or _image_download_timeout_for_url(image_url)
    url_hash = hashlib.sha1(image_url.encode("utf-8")).hexdigest()

    if image_url.startswith("data:"):
        data, mime_type = _decode_data_url(image_url)
        if not _is_image_mime(mime_type):
            mime_type = _sniff_mime_from_bytes(data[:32])
        if not _is_image_mime(mime_type):
            raise ValueError("data URL is not an image")
        local_path = os.path.join(save_dir, f"{url_hash}{_extension_from_mime(mime_type)}")
        if not (os.path.exists(local_path) and os.path.getsize(local_path) > 0):
            _write_bytes_atomically(local_path, data)
        return local_path, mime_type

    if image_url.startswith("file://"):
        local_src = image_url[7:]
        if not os.path.exists(local_src):
            raise ValueError(f"Local file does not exist: {local_src}")
        with open(local_src, "rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"Local image too large: {local_src}")
        mime_type = _sniff_mime_from_bytes(data[:32])
        if not _is_image_mime(mime_type):
            mime_type = mimetypes.guess_type(local_src)[0] or mime_type
        if not _is_image_mime(mime_type):
            raise ValueError(f"Local file is not an image: {local_src}")
        local_path = os.path.join(save_dir, f"{url_hash}{_extension_from_mime(mime_type)}")
        if not (os.path.exists(local_path) and os.path.getsize(local_path) > 0):
            _write_bytes_atomically(local_path, data)
        return local_path, mime_type

    data = None
    mime_type = ""
    last_err: Optional[Exception] = None

    try:
        data, mime_type = _download_via_requests(
            image_url=image_url,
            timeout=timeout,
            max_bytes=max_bytes,
            session=session,
        )
    except Exception as e:
        last_err = e

    if data is None:
        try:
            data, mime_type = _download_via_urllib(
                image_url=image_url,
                timeout=max(timeout),
                max_bytes=max_bytes,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to download image: {image_url}; last_err={repr(last_err)}; fallback_err={repr(e)}"
            ) from e

    if not _is_image_mime(mime_type):
        raise ValueError(f"Downloaded content is not image mime: {mime_type}")

    local_path = os.path.join(save_dir, f"{url_hash}{_extension_from_mime(mime_type)}")
    if not (os.path.exists(local_path) and os.path.getsize(local_path) > 0):
        _write_bytes_atomically(local_path, data)
    return local_path, mime_type


def local_image_to_data_url(local_path: str, mime_type: Optional[str] = None) -> str:
    mime_type = mime_type or mimetypes.guess_type(local_path)[0] or "image/jpeg"
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def pil_image_to_data_url(image: Image.Image, mime_type: str = "image/png") -> str:
    buf = BytesIO()
    fmt = "PNG" if mime_type.endswith("png") else "JPEG"
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def prepare_mllm_image_input(
    image_url: str,
    thumbnail_url: str = "",
    save_dir: str = SAVE_IMAGE_DIR,
) -> Tuple[str, str, str]:
    tried = []
    candidates = build_image_url_candidates(image_url=image_url, thumbnail_url=thumbnail_url)

    for u in candidates:
        try:
            local_path, mime_type = download_image_to_local(
                u,
                save_dir=save_dir,
                session=IMAGE_HTTP_SESSION,
            )
            data_url = local_image_to_data_url(local_path, mime_type=mime_type)
            return local_path, data_url, u
        except Exception as e:
            tried.append(f"{u} -> {repr(e)}")

    raise ImagePreparationError(
        "Failed to prepare local image for MLLM: " + " | ".join(tried[:5])
    )


def image_input_to_data_url(image_input) -> str:
    if isinstance(image_input, Image.Image):
        return pil_image_to_data_url(image_input)
    if isinstance(image_input, str):
        if image_input.startswith("data:"):
            return image_input
        if image_input.startswith("http://") or image_input.startswith("https://") or image_input.startswith("file://"):
            _, data_url, _ = prepare_mllm_image_input(image_input)
            return data_url
        if os.path.exists(image_input):
            return local_image_to_data_url(image_input)
    raise ValueError(f"Unsupported image input: {type(image_input)}")
