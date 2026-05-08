from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

from agentic_search.models.base import BaseModel
from agentic_search.tools.base import BaseSkill
from agentic_search.tools.image_ops import ImageCropSkill
from agentic_search.tools.python_eval import PythonEvalSkill
from agentic_search.tools.summarize import SummarizeTextSkill


class SkillRegistry:
    def __init__(self, skills: Optional[Iterable[BaseSkill]] = None):
        self._skills: Dict[str, BaseSkill] = {}
        if skills:
            for skill in skills:
                self.register(skill)

    def register(self, skill: BaseSkill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> BaseSkill:
        if name not in self._skills:
            raise KeyError(f"Skill not registered: {name}")
        return self._skills[name]

    def exists(self, name: str) -> bool:
        return name in self._skills

    def specs(self) -> str:
        return "\n".join(skill.spec() for skill in self._skills.values())

    def run(self, name: str, **kwargs):
        return self.get(name).run(**kwargs)


def default_skill_registry(model: BaseModel, mode: str = "web") -> SkillRegistry:
    """Create the default skill registry.

    Args:
        model: The language model instance used by summarize skill.
        mode: Search mode. "local" registers local KB retrieval tools;
              any other value (default "web") registers web search tools.
    """
    registry = SkillRegistry()

    if mode == "local":
        from agentic_search.tools.local_retrieval import (
            LocalTextSearchSkill,
            LocalImageSearchSkill,
            LocalTextToImageSearchSkill,
        )
        registry.register(LocalTextSearchSkill())
        registry.register(LocalImageSearchSkill())
        registry.register(LocalTextToImageSearchSkill())
    else:
        from agentic_search.tools.composite_ops import CropAndSearchSkill
        from agentic_search.tools.remote_ops import UploadToRemoteSkill, SearchImageFromLocalSkill
        from agentic_search.tools.search import FetchWebpageTextSkill, ImageSearchSkill, LensSearchSkill, WebSearchSkill
        from agentic_search.tools.browser_ops import BrowseWebPageSkill

        registry.register(WebSearchSkill())
        registry.register(ImageSearchSkill())
        registry.register(LensSearchSkill())
        registry.register(FetchWebpageTextSkill())
        registry.register(UploadToRemoteSkill())
        registry.register(SearchImageFromLocalSkill())
        registry.register(BrowseWebPageSkill())
        registry.register(CropAndSearchSkill(registry=registry))

    # Common tools shared by both modes
    registry.register(PythonEvalSkill())
    registry.register(ImageCropSkill())
    registry.register(SummarizeTextSkill(model=model))
    return registry
