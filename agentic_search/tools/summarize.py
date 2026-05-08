from __future__ import annotations

from agentic_search.models.base import BaseModel
from agentic_search.tools.base import BaseSkill
from agentic_search.types import SkillResult


class SummarizeTextSkill(BaseSkill):
    name = "summarize_text"
    description = "Summarize text with the loaded LLM. Args: text, instruction(optional)."

    def __init__(self, model: BaseModel):
        self.model = model

    def run(self, **kwargs) -> SkillResult:
        text = kwargs["text"]
        instruction = kwargs.get("instruction", "Summarize the following text in 3-6 concise bullet points.")
        prompt = f"{instruction}\n\nText:\n{text}"
        summary = self.model.generate_response(prompt)
        return SkillResult(ok=True, skill_name=self.name, output={"summary": summary})
