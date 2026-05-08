from __future__ import annotations

import contextlib
import io
from typing import Any, Dict

from agentic_search.tools.base import BaseSkill
from agentic_search.types import SkillResult


class PythonEvalSkill(BaseSkill):
    name = "python_eval"
    description = "Execute short Python code and capture stdout. Args: code."

    def run(self, **kwargs) -> SkillResult:
        code = kwargs["code"]
        stdout = io.StringIO()
        local_vars: Dict[str, Any] = {}
        global_vars: Dict[str, Any] = {"__builtins__": __builtins__}
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, global_vars, local_vars)
            return SkillResult(
                ok=True,
                skill_name=self.name,
                output={
                    "stdout": stdout.getvalue(),
                    "locals": {k: repr(v) for k, v in local_vars.items() if not k.startswith("__")},
                },
            )
        except Exception as e:
            return SkillResult(ok=False, skill_name=self.name, output=None, error=repr(e))
