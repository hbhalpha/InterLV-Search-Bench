"""Minimal example for registering your own model backend.

Run from the repository root:
    pip install -e .
    python examples/custom_backend.py
"""

from agentic_search import AgentMode, AgenticSearchFramework, register_model_backend, load_model
from agentic_search.models import BaseModel
from agentic_search.tools import default_skill_registry


class MyModel(BaseModel):
    """Replace this class with your own local model or private API wrapper."""

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        # Example: load tokenizer/model/client here.
        # self.client = YourClient(...)

    def generate_response(self, text: str, images=None, **kwargs) -> str:
        # Return plain model text. The framework parser will read tags such as
        # <query>...</query>, <tool>...</tool>, <code>...</code>, and <done>...</done>.
        return "<done>This is a custom backend smoke test.</done>"


def main() -> None:
    register_model_backend("my_backend", MyModel)

    model = load_model("my-model-name-or-path", backend="my_backend")
    registry = default_skill_registry(model=model)
    agent = AgenticSearchFramework(
        model=model,
        skill_registry=registry,
        mode=AgentMode.UNTIL_DONE,
        max_iters=3,
    )

    result = agent.run("Say hello")
    print(result.final_answer)


if __name__ == "__main__":
    main()
