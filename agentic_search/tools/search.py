from __future__ import annotations

from agentic_search.clients.search_client import fetch_webpage_text, images_search, lens_search, web_search
from agentic_search.tools.base import BaseSkill
from agentic_search.types import SkillResult


class WebSearchSkill(BaseSkill):
    name = "web_search"
    description = "Text-to-web search. Args: query, num, country, locale, location, page."

    def run(self, **kwargs) -> SkillResult:
        data = web_search(
            query=kwargs["query"],
            num=kwargs.get("num", 10),
            country=kwargs.get("country", "us"),
            locale=kwargs.get("locale", "en"),
            location=kwargs.get("location", "United States"),
            page=kwargs.get("page", 1),
        )
        return SkillResult(ok=True, skill_name=self.name, output=data)


class ImageSearchSkill(BaseSkill):
    name = "image_search"
    description = (
        "Text-to-image search. Returns raw search JSON plus top-k downloaded images saved under ./temp "
        "if download succeeds. Failed downloads are tolerated and only recorded as errors. "
        "Args: query, num, country, locale, location, page."
    )

    def run(self, **kwargs) -> SkillResult:
        data = images_search(
            query=kwargs["query"],
            num=kwargs.get("num", 10),
            country=kwargs.get("country", "us"),
            locale=kwargs.get("locale", "en"),
            location=kwargs.get("location", "United States"),
            page=kwargs.get("page", 1),
        )
        return SkillResult(ok=True, skill_name=self.name, output=data)


class LensSearchSkill(BaseSkill):
    name = "lens_search"
    description = "Image-to-search via lens. Args: image_url, hl, gl."

    def run(self, **kwargs) -> SkillResult:
        data = lens_search(
            image_url=kwargs["image_url"],
            hl=kwargs.get("hl", "en"),
            gl=kwargs.get("gl", "us"),
        )
        return SkillResult(ok=True, skill_name=self.name, output=data)


class FetchWebpageTextSkill(BaseSkill):
    name = "fetch_webpage_text"
    description = "Fetch title + cleaned webpage body text. Args: url, max_body_chars."

    def run(self, **kwargs) -> SkillResult:
        title, body = fetch_webpage_text(
            url=kwargs["url"],
            max_body_chars=kwargs.get("max_body_chars", 1200),
        )
        return SkillResult(ok=True, skill_name=self.name, output={"title": title, "body": body})
