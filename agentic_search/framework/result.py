from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StepTrace:
    iteration: int
    prompt: str
    model_output: str
    parsed_actions: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    running_memory: str = ""


@dataclass
class AgentRunResult:
    final_answer: str
    done: bool
    iterations: int
    trace: List[StepTrace] = field(default_factory=list)
