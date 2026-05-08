# -*- coding: utf-8 -*-
# pip install requests
# Use --contact_email or INTERLV_CONTACT_EMAIL to set your own Wikimedia-compliant contact address.

import argparse
import json
import mimetypes
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

import requests


RETRY_STATUS = {429, 500, 502, 503, 504}

_thread_local = threading.local()

WIKI_API_SEMAPHORE = threading.Semaphore(4)
IMAGE_DOWNLOAD_SEMAPHORE = threading.Semaphore(1)

CONTACT_EMAIL_PLACEHOLDER = "your_email@example.com"
CONTACT_EMAIL = os.getenv("INTERLV_CONTACT_EMAIL", CONTACT_EMAIL_PLACEHOLDER)


def set_contact_email(email: Optional[str]) -> None:
    """Set the contact email used in the Wikimedia User-Agent header."""
    global CONTACT_EMAIL
    if email:
        CONTACT_EMAIL = email.strip() or CONTACT_EMAIL_PLACEHOLDER
    # Existing thread-local sessions may already contain an old User-Agent.
    if hasattr(_thread_local, "session"):
        delattr(_thread_local, "session")


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(
            {
                # Replace the contact email with your own address.
                "User-Agent": f"wikidata-wikipedia-image-refill/1.0 (contact: {CONTACT_EMAIL})",
                "Accept": "*/*",
            }
        )
        _thread_local.session = s
    return _thread_local.session




def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_existing_image(image_dir: Path, qid: str) -> Optional[Path]:
    files = list(image_dir.glob(f"{qid}.*"))
    return files[0] if files else None


def parse_retry_after_seconds(resp: Optional[requests.Response]) -> Optional[int]:
    if resp is None:
        return None
    v = resp.headers.get("Retry-After")
    if not v:
        return None
    v = v.strip()
    return int(v) if v.isdigit() else None


def backoff_seconds(attempt: int, is_429: bool = False) -> int:
    if is_429:
        base = [15, 30, 60]
    else:
        base = [3, 6, 12]
    idx = min(attempt - 1, len(base) - 1)
    return int(base[idx] + random.uniform(0, 1.5))


def request_with_retry(
    url: str,
    *,
    params: Optional[dict] = None,
    stream: bool = False,
    retries: int = 3,
    connect_timeout: int = 15,
    read_timeout: int = 180,
    log_prefix: str = "",
    semaphore: Optional[threading.Semaphore] = None,
) -> requests.Response:
    last_err = None

    for attempt in range(1, retries + 1):
        resp = None
        try:
            if semaphore is None:
                resp = get_session().get(
                    url,
                    params=params,
                    stream=stream,
                    timeout=(connect_timeout, read_timeout),
                    allow_redirects=True,
                )
            else:
                with semaphore:
                    resp = get_session().get(
                        url,
                        params=params,
                        stream=stream,
                        timeout=(connect_timeout, read_timeout),
                        allow_redirects=True,
                    )

            if resp.status_code in RETRY_STATUS:
                wait_s = parse_retry_after_seconds(resp)
                if wait_s is None:
                    wait_s = backoff_seconds(attempt, is_429=(resp.status_code == 429))

                print(
                    f"[RETRY]{log_prefix} status={resp.status_code} "
                    f"attempt={attempt}/{retries} wait={wait_s}s url={resp.url}"
                )
                last_err = requests.HTTPError(
                    f"{resp.status_code} Client Error for url: {resp.url}",
                    response=resp,
                )
                resp.close()
                if attempt < retries:
                    time.sleep(wait_s)
                    continue
                break

            resp.raise_for_status()
            return resp

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.HTTPError,
        ) as e:
            last_err = e
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass

            if attempt < retries:
                wait_s = backoff_seconds(attempt, is_429=False)
                print(
                    f"[RETRY]{log_prefix} error={type(e).__name__} "
                    f"attempt={attempt}/{retries} wait={wait_s}s url={url}"
                )
                time.sleep(wait_s)
                continue
            break

    raise last_err


def read_ent_links(path: Path) -> Dict[str, str]:
    """
    ent_links format:
      http://dbpedia.org/resource/Chișinău    http://www.wikidata.org/entity/Q21197
    return:
      { "Q21197": "http://dbpedia.org/resource/Chișinău" }
    """
    if not path.exists():
        raise FileNotFoundError(f"ent_links file not found: {path}")

    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()

            if len(parts) != 2:
                print(f"[WARN] bad ent_links line {line_no}: {line}")
                continue

            dbpedia_url, wikidata_url = parts
            m = re.search(r"\bQ\d+\b", wikidata_url)
            if not m:
                print(f"[WARN] no qid in ent_links line {line_no}: {line}")
                continue

            qid = m.group(0)
            result[qid] = dbpedia_url

    return result


def scan_missing_qids(text_dir: Path, image_dir: Path) -> List[str]:
    
    qids = []
    seen = set()

    json_files = sorted(text_dir.glob("*.json"))
    print(f"[INFO] found text json files: {len(json_files)}")

    for path in json_files:
        qid = path.stem
        if not re.fullmatch(r"Q\d+", qid):
            try:
                obj = load_json(path)
                qid = obj.get("id", "")
            except Exception:
                qid = path.stem

        if not re.fullmatch(r"Q\d+", qid):
            continue

        missing = False

        try:
            obj = load_json(path)
            if not isinstance(obj, dict):
                obj = {}
        except Exception:
            obj = {}

        image_file = obj.get("image_file")
        if image_file:
            p = image_dir / image_file
            if not p.exists():
                missing = True
        else:
            missing = True

        existing = find_existing_image(image_dir, qid)
        if existing is None:
            missing = True
        else:
            pass

        if missing and qid not in seen:
            seen.add(qid)
            qids.append(qid)

    return qids




def _path_exists_from_cwd_or_file(path_value: str, owner_file: Path) -> bool:
    p = Path(path_value)
    if p.is_absolute():
        return p.exists()
    return (Path.cwd() / p).exists() or (owner_file.parent / p).exists()


def scan_missing_qids_from_kgdb(kgdb_file: Path, image_dir: Path) -> List[str]:
    """
    Scan a JSONL/JSON KGDB file for entities whose image is missing.

    Supported fields:
      - qid or id
      - local_image_path or image_path
    """
    if not kgdb_file.exists():
        raise FileNotFoundError(f"kgdb file not found: {kgdb_file}")

    raw = kgdb_file.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if kgdb_file.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        obj = json.loads(raw)
        rows = obj.get("records", obj) if isinstance(obj, dict) else obj
        if not isinstance(rows, list):
            raise ValueError("kgdb_file must be JSONL, a JSON list, or a JSON object with a records list")

    qids: List[str] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid") or row.get("id") or "")
        if not re.fullmatch(r"Q\d+", qid):
            continue

        image_path = row.get("local_image_path") or row.get("image_path") or ""
        has_image_path = isinstance(image_path, str) and bool(image_path.strip())
        path_exists = has_image_path and _path_exists_from_cwd_or_file(image_path, kgdb_file)
        existing_by_qid = find_existing_image(image_dir, qid) is not None

        if not path_exists and not existing_by_qid and qid not in seen:
            seen.add(qid)
            qids.append(qid)

    return qids


def dbpedia_url_to_resource_name(dbpedia_url: str) -> str:
    return unquote(dbpedia_url.rstrip("/").split("/")[-1])


def dbpedia_resource_to_wikipedia_title(dbpedia_url: str) -> str:
    resource_name = dbpedia_url_to_resource_name(dbpedia_url)
    return resource_name.replace("_", " ")


def site_api_url(site: str) -> str:
    lang = site[:-4]  # enwiki -> en
    return f"https://{lang}.wikipedia.org/w/api.php"


def get_pageimage(api_url: str, title: str, thumb_width: int = 512) -> Optional[dict]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageimages",
        "piprop": "thumbnail|name",
        "pithumbsize": thumb_width,
        "redirects": 1,
        "titles": title,
    }
    resp = request_with_retry(
        api_url,
        params=params,
        retries=3,
        connect_timeout=15,
        read_timeout=120,
        log_prefix=f"[PAGEIMAGE][{title}]",
        semaphore=WIKI_API_SEMAPHORE,
    )
    try:
        data = resp.json()
    finally:
        resp.close()

    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        thumb = page.get("thumbnail")
        if thumb and thumb.get("source"):
            return {
                "image_url": thumb["source"],
                "image_name": page.get("pageimage") or "pageimage",
                "method": "pageimages",
            }
    return None


def get_parse_images(api_url: str, title: str) -> List[str]:
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "images",
        "redirects": 1,
    }
    resp = request_with_retry(
        api_url,
        params=params,
        retries=3,
        connect_timeout=15,
        read_timeout=120,
        log_prefix=f"[PARSEIMG][{title}]",
        semaphore=WIKI_API_SEMAPHORE,
    )
    try:
        data = resp.json()
    finally:
        resp.close()

    parse = data.get("parse", {})
    images = parse.get("images", [])
    return images if isinstance(images, list) else []


def search_wikipedia_titles(api_url: str, query: str, limit: int = 5) -> List[str]:
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "srwhat": "title",
    }
    resp = request_with_retry(
        api_url,
        params=params,
        retries=3,
        connect_timeout=15,
        read_timeout=120,
        log_prefix=f"[SEARCH][{query}]",
        semaphore=WIKI_API_SEMAPHORE,
    )
    try:
        data = resp.json()
    finally:
        resp.close()

    hits = data.get("query", {}).get("search", [])
    out = []
    for x in hits:
        title = x.get("title")
        if title:
            out.append(title)
    return out


def get_imageinfo_url(api_url: str, filename: str, thumb_width: int = 512) -> Optional[str]:
    params = {
        "action": "query",
        "format": "json",
        "titles": f"File:{filename}",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": thumb_width,
    }
    resp = request_with_retry(
        api_url,
        params=params,
        retries=3,
        connect_timeout=15,
        read_timeout=120,
        log_prefix=f"[IMAGEINFO][{filename}]",
        semaphore=WIKI_API_SEMAPHORE,
    )
    try:
        data = resp.json()
    finally:
        resp.close()

    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        infos = page.get("imageinfo", [])
        if infos:
            return infos[0].get("thumburl") or infos[0].get("url")
    return None


def is_probably_bad_page_image(filename: str) -> bool:
    name = filename.lower().strip()

    bad_exts = (".svg", ".ogg", ".ogv", ".webm", ".mp3", ".wav", ".pdf")
    if name.endswith(bad_exts):
        return True

    bad_keywords = [
        "wikidata-logo",
        "wikimedia",
        "commons-logo",
        "question_book",
        "help_icon",
        "symbol",
        "disambig",
        "stub",
        "edit-clear",
        "ambox",
        "system-search",
        "magnify-clip",
        "nuvola",
        "flag of",
        "coat of arms",
        "emblem",
        "logo",
        "icon",
    ]
    return any(kw in name for kw in bad_keywords)


def score_page_image(filename: str, page_title: str) -> int:
    
    name = filename.lower()
    title = page_title.lower().replace("_", " ")

    score = 100

    if name.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")):
        score -= 20

    title_tokens = [t for t in re.split(r"[^a-z0-9]+", title) if t]
    hit = 0
    for tok in title_tokens[:8]:
        if tok and tok in name:
            hit += 1
    score -= min(hit * 5, 30)

    good_keywords = [
        "cover",
        "poster",
        "front",
        "book",
        "novel",
        "album",
        "portrait",
        "photo",
        "title",
    ]
    for kw in good_keywords:
        if kw in name:
            score -= 6

    if len(name) < 8:
        score += 8

    return score


def guess_ext_from_name_or_type(name: str, content_type: Optional[str] = None) -> str:
    ext = os.path.splitext(name)[1]
    if ext:
        return ext.lower()

    if content_type:
        content_type = content_type.split(";")[0].strip().lower()
        ext = mimetypes.guess_extension(content_type)
        if ext == ".jpe":
            ext = ".jpg"
        if ext:
            return ext

    return ".jpg"


def download_binary(url: str, out_path: Path, log_prefix: str = "") -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    last_err = None

    for attempt in range(1, 4):
        resp = None
        try:
            with IMAGE_DOWNLOAD_SEMAPHORE:
                resp = get_session().get(
                    url,
                    stream=True,
                    timeout=(20, 240),
                    allow_redirects=True,
                )

            if resp.status_code in {429, 500, 502, 503, 504}:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_s = int(retry_after)
                else:
                    wait_s = [15, 30, 60][attempt - 1]

                print(f"[RETRY][IMG]{log_prefix} status={resp.status_code} attempt={attempt}/3 wait={wait_s}s final_url={resp.url}")
                last_err = requests.HTTPError(f"{resp.status_code} Client Error for url: {resp.url}", response=resp)
                resp.close()

                if attempt < 3:
                    time.sleep(wait_s)
                    continue
                break

            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)

            tmp_path.replace(out_path)
            resp.close()
            return content_type

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.HTTPError,
        ) as e:
            last_err = e
            try:
                if resp is not None:
                    resp.close()
            except Exception:
                pass

            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

            if attempt < 3:
                wait_s = [5, 10, 20][attempt - 1]
                print(f"[RETRY][IMG]{log_prefix} error={type(e).__name__} attempt={attempt}/3 wait={wait_s}s url={url}")
                time.sleep(wait_s)
                continue
            break

    raise last_err

def update_text_json_if_exists(
    text_dir: Path,
    qid: str,
    image_file_name: str,
    image_source_method: Optional[str],
    image_candidates: Optional[List[str]],
):
    text_path = text_dir / f"{qid}.json"
    if not text_path.exists():
        return

    try:
        obj = load_json(text_path)
        if not isinstance(obj, dict):
            obj = {"id": qid}
    except Exception:
        obj = {"id": qid}

    obj["image_file"] = image_file_name
    obj["image_source_method"] = image_source_method
    if image_candidates is not None:
        obj["image_candidates"] = image_candidates[:20]

    if "id" not in obj:
        obj["id"] = qid

    save_json(text_path, obj)


def build_title_candidates_from_entlink(dbpedia_url: str) -> List[str]:
    title = dbpedia_resource_to_wikipedia_title(dbpedia_url)
    resource_name = dbpedia_url_to_resource_name(dbpedia_url)

    candidates = []
    for x in [
        title,
        resource_name.replace("_", " "),
        resource_name,
    ]:
        x = x.strip()
        if x and x not in candidates:
            candidates.append(x)
    return candidates


def try_pick_from_exact_or_search_title(api_url: str, title: str, thumb_width: int) -> Optional[dict]:
    pageimg = get_pageimage(api_url, title, thumb_width=thumb_width)
    if pageimg and pageimg.get("image_url"):
        return {
            "image_url": pageimg["image_url"],
            "image_name": pageimg.get("image_name") or "pageimage",
            "source_method": "wikipedia_pageimages",
            "image_candidates": None,
            "used_title": title,
        }

    try:
        images = get_parse_images(api_url, title)
    except Exception:
        images = []

    images = [x for x in images if isinstance(x, str) and x.strip()]
    images = [x.strip() for x in images if not is_probably_bad_page_image(x)]

    if images:
        images.sort(key=lambda x: score_page_image(x, title))
        for filename in images[:8]:
            img_url = None

            try:
                img_url = get_imageinfo_url(api_url, filename, thumb_width=thumb_width)
            except Exception:
                img_url = None

            if not img_url:
                try:
                    img_url = get_imageinfo_url(
                        "https://commons.wikimedia.org/w/api.php",
                        filename,
                        thumb_width=thumb_width,
                    )
                except Exception:
                    img_url = None

            if img_url:
                return {
                    "image_url": img_url,
                    "image_name": filename,
                    "source_method": "wikipedia_parse_images",
                    "image_candidates": images[:20],
                    "used_title": title,
                }

    try:
        search_hits = search_wikipedia_titles(api_url, title, limit=5)
    except Exception:
        search_hits = []

    for hit_title in search_hits:
        if hit_title == title:
            continue

        pageimg = get_pageimage(api_url, hit_title, thumb_width=thumb_width)
        if pageimg and pageimg.get("image_url"):
            return {
                "image_url": pageimg["image_url"],
                "image_name": pageimg.get("image_name") or "pageimage",
                "source_method": "wikipedia_search_pageimages",
                "image_candidates": [hit_title],
                "used_title": hit_title,
            }

        try:
            images = get_parse_images(api_url, hit_title)
        except Exception:
            images = []

        images = [x for x in images if isinstance(x, str) and x.strip()]
        images = [x.strip() for x in images if not is_probably_bad_page_image(x)]

        if images:
            images.sort(key=lambda x: score_page_image(x, hit_title))
            for filename in images[:5]:
                img_url = None
                try:
                    img_url = get_imageinfo_url(api_url, filename, thumb_width=thumb_width)
                except Exception:
                    img_url = None

                if not img_url:
                    try:
                        img_url = get_imageinfo_url(
                            "https://commons.wikimedia.org/w/api.php",
                            filename,
                            thumb_width=thumb_width,
                        )
                    except Exception:
                        img_url = None

                if img_url:
                    return {
                        "image_url": img_url,
                        "image_name": filename,
                        "source_method": "wikipedia_search_parse_images",
                        "image_candidates": [hit_title] + images[:20],
                        "used_title": hit_title,
                    }

    return None


def process_one_qid(
    qid: str,
    ent_map: Dict[str, str],
    image_dir: Path,
    text_dir: Path,
    thumb_width: int,
    force: bool = False,
):
    existing = find_existing_image(image_dir, qid)
    if existing and not force:
        update_text_json_if_exists(
            text_dir=text_dir,
            qid=qid,
            image_file_name=existing.name,
            image_source_method="existing_local_file",
            image_candidates=None,
        )
        return {
            "qid": qid,
            "status": "skipped_has_image",
            "detail": existing.name,
        }

    dbpedia_url = ent_map.get(qid)
    if not dbpedia_url:
        return {
            "qid": qid,
            "status": "missing_ent_link",
            "detail": "qid not found in ent_links",
        }

    api_url = "https://en.wikipedia.org/w/api.php"
    title_candidates = build_title_candidates_from_entlink(dbpedia_url)

    picked = None
    last_titles_tried = []

    for title in title_candidates:
        last_titles_tried.append(title)
        print(f"[TITLE] qid={qid} try={title}")
        picked = try_pick_from_exact_or_search_title(api_url, title, thumb_width=thumb_width)
        if picked:
            break

    if not picked:
        return {
            "qid": qid,
            "status": "still_no_image",
            "detail": {
                "dbpedia_url": dbpedia_url,
                "titles_tried": last_titles_tried,
            },
        }

    image_name = picked["image_name"] or f"{qid}.jpg"
    img_url = picked["image_url"]

    ext = guess_ext_from_name_or_type(image_name, None)
    target_path = image_dir / f"{qid}{ext}"

    old_existing = find_existing_image(image_dir, qid)
    if old_existing and old_existing != target_path:
        try:
            old_existing.unlink()
        except Exception:
            pass

    try:
        content_type = download_binary(img_url, target_path, log_prefix=f"[{qid}]")
    except Exception as e:
        return {
            "qid": qid,
            "status": "image_download_failed",
            "detail": str(e),
        }

    better_ext = guess_ext_from_name_or_type(image_name, content_type)
    better_path = image_dir / f"{qid}{better_ext}"
    if better_path != target_path:
        try:
            if better_path.exists():
                better_path.unlink()
            target_path.rename(better_path)
            target_path = better_path
        except Exception:
            pass

    update_text_json_if_exists(
        text_dir=text_dir,
        qid=qid,
        image_file_name=target_path.name,
        image_source_method=picked.get("source_method"),
        image_candidates=picked.get("image_candidates"),
    )

    return {
        "qid": qid,
        "status": "downloaded",
        "detail": {
            "image_file": target_path.name,
            "source_method": picked.get("source_method"),
            "used_title": picked.get("used_title"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ent_links_file", required=True, help="ent_links or ent_links.txt path")
    parser.add_argument("--kgdb_file", default=None, help="Optional KGDB JSON/JSONL path. If set, missing images are scanned from this file.")
    parser.add_argument("--text_dir", default="./data/text", help="text json directory; used when --kgdb_file is not set")
    parser.add_argument("--image_dir", default="./data/image", help="图片目录")
    parser.add_argument("--workers", type=int, default=4, help="线程数")
    parser.add_argument("--thumb_width", type=int, default=512, help="缩略图宽度，默认 512")
    parser.add_argument("--force", action="store_true", help="即使已有图片也重新下载")
    parser.add_argument("--scan_only", action="store_true", help="只扫描缺图 qid，不下载")
    parser.add_argument("--missing_qids_file", default="./data/missing_image_qids.txt", help="扫描出的缺图 qid")
    parser.add_argument("--error_file", default="./data/refill_from_missing_errors.json", help="错误输出 json")
    parser.add_argument("--still_no_image_file", default="./data/still_no_image.txt", help="QIDs that still have no image after download")
    parser.add_argument("--contact_email", default=os.getenv("INTERLV_CONTACT_EMAIL", CONTACT_EMAIL_PLACEHOLDER), help="Contact email used in the Wikimedia User-Agent. Replace with your own email.")
    args = parser.parse_args()

    set_contact_email(args.contact_email)
    if CONTACT_EMAIL == CONTACT_EMAIL_PLACEHOLDER:
        print("[WARN] Using placeholder contact email. Pass --contact_email you@example.com or set INTERLV_CONTACT_EMAIL.")

    ent_links_file = Path(args.ent_links_file)
    kgdb_file = Path(args.kgdb_file) if args.kgdb_file else None
    text_dir = Path(args.text_dir)
    image_dir = Path(args.image_dir)
    missing_qids_file = Path(args.missing_qids_file)
    error_file = Path(args.error_file)
    still_no_image_file = Path(args.still_no_image_file)

    image_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    ent_map = read_ent_links(ent_links_file)
    if kgdb_file is not None:
        missing_qids = scan_missing_qids_from_kgdb(kgdb_file, image_dir)
        print(f"[INFO] scanned KGDB file: {kgdb_file}")
    else:
        missing_qids = scan_missing_qids(text_dir, image_dir)
        print(f"[INFO] scanned text directory: {text_dir}")

    # Keep only qids that exist in ent_links.
    missing_qids = [qid for qid in missing_qids if qid in ent_map]

    print(f"[INFO] ent_links loaded: {len(ent_map)}")
    print(f"[INFO] missing qids after scan: {len(missing_qids)}")

    with open(missing_qids_file, "w", encoding="utf-8") as f:
        for qid in missing_qids:
            f.write(qid + "\n")

    print(f"[INFO] missing_qids saved -> {missing_qids_file}")

    if args.scan_only:
        print("[DONE] scan_only enabled, no download performed.")
        return

    if still_no_image_file.exists():
        still_no_image_file.unlink()

    stats = {
        "downloaded": 0,
        "skipped_has_image": 0,
        "missing_ent_link": 0,
        "image_download_failed": 0,
        "still_no_image": 0,
    }

    errors: Dict[str, Dict[str, object]] = {
        "missing_ent_link": {},
        "image_download_failed": {},
        "still_no_image": {},
    }

    done = 0
    total = len(missing_qids)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_one_qid,
                qid,
                ent_map,
                image_dir,
                text_dir,
                args.thumb_width,
                args.force,
            ): qid
            for qid in missing_qids
        }

        for fut in as_completed(futures):
            result = fut.result()
            qid = result["qid"]
            status = result["status"]
            detail = result["detail"]

            done += 1
            stats[status] = stats.get(status, 0) + 1

            if status in errors:
                errors[status][qid] = detail

            if status == "still_no_image":
                with open(still_no_image_file, "a", encoding="utf-8") as f:
                    f.write(f"{qid}\n")

            if done % 20 == 0 or done == total:
                print(
                    f"[INFO] {done}/{total} | "
                    f"downloaded={stats['downloaded']} "
                    f"skipped={stats['skipped_has_image']} "
                    f"missing_ent_link={stats['missing_ent_link']} "
                    f"img_fail={stats['image_download_failed']} "
                    f"still_no_image={stats['still_no_image']}"
                )

    out = {
        "stats": stats,
        "errors": errors,
    }
    save_json(error_file, out)

    print("[DONE]")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[DONE] still_no_image -> {still_no_image_file}")
    print(f"[DONE] errors -> {error_file}")


if __name__ == "__main__":
    main()
