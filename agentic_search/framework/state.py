from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentState:
    query: str
    observations: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    done: bool = False
    running_memory: str = ""
