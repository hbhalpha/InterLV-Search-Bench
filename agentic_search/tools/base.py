from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from agentic_search.types import SkillResult


class BaseSkill(ABC):
    name: str = "base_skill"
    description: str = "Base skill"

    @abstractmethod
    def run(self, **kwargs) -> SkillResult:
        raise NotImplementedError

    def spec(self) -> str:
        return f"- {self.name}: {self.description}"
