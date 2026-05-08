from __future__ import annotations

import os
from typing import Optional

from agentic_search.local_kb import LocalKBManager
from agentic_search.tools.base import BaseSkill
from agentic_search.types import SkillResult
from agentic_search.utils.image_utils import download_image_to_local

_LOCAL_MANAGER: Optional[LocalKBManager] = None


def get_local_manager() -> LocalKBManager:
    global _LOCAL_MANAGER
    if _LOCAL_MANAGER is None:
        _LOCAL_MANAGER = LocalKBManager.from_env()
    return _LOCAL_MANAGER


def _ensure_local_image(image: str) -> str:
    if image.startswith("http://") or image.startswith("https://") or image.startswith("file://"):
        local_path, _ = download_image_to_local(image, save_dir="/tmp/agentic_search_local_retrieval")
        return local_path
    return image


class LocalTextSearchSkill(BaseSkill):
    name = "local_text_search"
    description = "Local text-to-text retrieval over the local KB. Args: query, top_k, max_text_chars."

    def run(self, **kwargs) -> SkillResult:
        mgr = get_local_manager()
        rows = mgr.text_search(
            query=kwargs["query"],
            top_k=kwargs.get("top_k", 5),
            max_text_chars=kwargs.get("max_text_chars", 1200),
            include_image_path=bool(kwargs.get("include_image_path", False)),
        )
        return SkillResult(
            ok=True,
            skill_name=self.name,
            output={
                "source": "local_kb",
                "query": kwargs["query"],
                "results": rows,
            },
        )


class LocalImageSearchSkill(BaseSkill):
    name = "local_image_search"
    description = "Local image-to-image retrieval over the local KB. Args: image, top_k, max_text_chars."

    def run(self, **kwargs) -> SkillResult:
        mgr = get_local_manager()
        image_path = _ensure_local_image(kwargs["image"])
        rows = mgr.image_search(
            image_path=image_path,
            top_k=kwargs.get("top_k", 5),
            max_text_chars=kwargs.get("max_text_chars", 1200),
            include_image_path=bool(kwargs.get("include_image_path", False)),
        )
        return SkillResult(
            ok=True,
            skill_name=self.name,
            output={
                "source": "local_kb",
                "query_image_provided": True,
                "results": rows,
            },
        )


class LocalTextToImageSearchSkill(BaseSkill):
    name = "local_text_to_image_search"
    description = "Local text-to-image retrieval over the local KB. Args: query, top_k, max_text_chars."

    def run(self, **kwargs) -> SkillResult:
        mgr = get_local_manager()
        rows = mgr.text_to_image_search(
            query=kwargs["query"],
            top_k=kwargs.get("top_k", 5),
            max_text_chars=kwargs.get("max_text_chars", 1200),
            include_image_path=bool(kwargs.get("include_image_path", False)),
        )
        return SkillResult(
            ok=True,
            skill_name=self.name,
            output={
                "source": "local_kb",
                "query": kwargs["query"],
                "results": rows,
            },
        )


def local_retrieval_enabled() -> bool:
    return bool(os.getenv("AGENTIC_SEARCH_LOCAL_INDEX_DIR") and os.getenv("AGENTIC_SEARCH_LOCAL_EMBED_MODEL"))
