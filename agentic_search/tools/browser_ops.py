from __future__ import annotations

from agentic_search.clients.browser_client import browse_web_page
from agentic_search.tools.base import BaseSkill
from agentic_search.types import SkillResult


class BrowseWebPageSkill(BaseSkill):
    name = "browse_web_page"
    description = (
        "Open a webpage in a browser and return its title plus a full-page screenshot. "
        "If the page is too long, the screenshot is capped so the long side is at most "
        "10x the short side. "
        "Args: url, viewport_width(optional), viewport_height(optional), timeout_ms(optional)."
    )

    def run(self, **kwargs) -> SkillResult:
        try:
            output = browse_web_page(
                url=kwargs["url"],
                save_dir=kwargs.get("save_dir", "./temp"),
                viewport_width=kwargs.get("viewport_width", 1440),
                viewport_height=kwargs.get("viewport_height", 2200),
                timeout_ms=kwargs.get("timeout_ms", 60000),
                max_aspect_ratio=kwargs.get("max_aspect_ratio", 10.0),
            )
            return SkillResult(
                ok=True,
                skill_name=self.name,
                output=output,
            )
        except Exception as e:
            return SkillResult(
                ok=False,
                skill_name=self.name,
                output=None,
                error=repr(e),
            )
