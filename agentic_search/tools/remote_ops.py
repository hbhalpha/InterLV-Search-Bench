from __future__ import annotations

import os
from typing import Any, Dict

import requests

from agentic_search.clients.search_client import lens_search
from agentic_search.tools.base import BaseSkill
from agentic_search.types import SkillResult


# ---------------------------------------------------------------------------
# Imgur anonymous upload helpers
# ---------------------------------------------------------------------------
# Upload images to Imgur to obtain a public URL.
# Register a free application at https://api.imgur.com to get a Client ID.
#
# Environment variable:
#   IMGUR_CLIENT_ID – required for image uploads

IMGUR_API_URL = "https://api.imgur.com/3/image"
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID", "")


def upload_to_imgur(local_path: str, **kwargs) -> dict:
    """
    Upload a local image file to Imgur via anonymous upload.

    Args:
        local_path: Path to the local image file to upload.

    Returns:
        dict with at least {"object_name": str, "imgur_url": str, "delete_hash": str}
        on success.

    Raises:
        FileNotFoundError: If the local file does not exist.
        ValueError: If IMGUR_CLIENT_ID is not set.
        requests.HTTPError: If the upload request fails.
    """
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    client_id = os.getenv("IMGUR_CLIENT_ID", IMGUR_CLIENT_ID)
    if not client_id:
        raise ValueError(
            "IMGUR_CLIENT_ID environment variable is required. "
            "Register a free application at https://api.imgur.com to get one."
        )

    headers = {"Authorization": f"Client-ID {client_id}"}

    with open(local_path, "rb") as f:
        resp = requests.post(
            IMGUR_API_URL,
            headers=headers,
            files={"image": f},
            timeout=60,
        )
        resp.raise_for_status()

    data = resp.json()
    if not data.get("success"):
        raise ValueError(f"Imgur upload failed: {data.get('data', {})}")

    imgur_data = data["data"]
    return {
        "object_name": os.path.basename(local_path),
        "imgur_url": imgur_data["link"],
        "delete_hash": imgur_data.get("deletehash", ""),
        "status": "uploaded",
    }


def get_remote_url(local_path: str, **kwargs) -> str:
    """
    Upload a local image to Imgur and return the public URL.

    Args:
        local_path: Path to the local image file.

    Returns:
        Public Imgur URL string, or empty string on failure.
    """
    try:
        result = upload_to_imgur(local_path, **kwargs)
        return result.get("imgur_url", "")
    except Exception as e:
        print(f"[get_remote_url] Failed to upload to Imgur: {e}")
        return ""


def upload_and_get_url(local_path: str, **kwargs) -> dict:
    """
    Upload a local image file to Imgur and return both upload info and the public URL.
    """
    upload_info = upload_to_imgur(local_path, **kwargs)
    return {**upload_info, "signed_url": upload_info["imgur_url"]}


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

class UploadToRemoteSkill(BaseSkill):
    name = "upload_to_remote"
    description = (
        "Upload a local image file to Imgur for a public URL. "
        "Args: local_path."
    )

    def run(self, **kwargs) -> SkillResult:
        local_path = kwargs.get("local_path") or kwargs.get("path")
        if not local_path:
            raise ValueError("upload_to_remote requires local_path")
        data = upload_to_imgur(local_path=local_path)
        return SkillResult(ok=True, skill_name=self.name, output=data)


class SearchImageFromLocalSkill(BaseSkill):
    name = "search_image_from_local"
    description = (
        "Upload a local image to Imgur, generate a public URL, "
        "then run Google Lens search on that URL. "
        "Args: local_path, hl?(optional), gl?(optional)."
    )

    def run(self, **kwargs) -> SkillResult:
        local_path = kwargs.get("local_path") or kwargs.get("path")
        if not local_path:
            raise ValueError("search_image_from_local requires local_path")

        uploaded = upload_and_get_url(local_path=local_path)
        imgur_url = uploaded.get("imgur_url") or uploaded.get("signed_url")
        if not imgur_url:
            return SkillResult(
                ok=False,
                skill_name=self.name,
                output={"upload": uploaded, "lens": None},
                error="Upload succeeded but no Imgur URL was generated.",
            )

        lens_result = lens_search(
            image_url=imgur_url,
            hl=kwargs.get("hl", "en"),
            gl=kwargs.get("gl", "us"),
        )
        return SkillResult(
            ok=True,
            skill_name=self.name,
            output={
                "upload": uploaded,
                "imgur_url": imgur_url,
                "lens": lens_result,
            },
        )
