from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from PIL import Image

ImageLike = Union[str, Image.Image]
ActionType = Literal["query", "code", "clip", "tool", "done", "text"]


@dataclass
class ParsedAction:
    action_type: ActionType
    content: str = ""
    tool_name: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class SkillResult:
    ok: bool
    skill_name: str
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
