import argparse
import json

from agentic_search import AgentMode, AgenticSearchFramework, load_model
from agentic_search.prompts import get_system_prompt
from agentic_search.tools import default_skill_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--mode", default="until_done", choices=["until_done", "fixed_depth"])
    parser.add_argument("--search-mode", default="web", choices=["web", "local"],
                        help="Search mode: 'web' for internet search, 'local' for local KB retrieval.")
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    mode = AgentMode.UNTIL_DONE if args.mode == "until_done" else AgentMode.FIXED_DEPTH
    model = load_model(args.model, backend=args.backend)
    registry = default_skill_registry(model=model, mode=args.search_mode)
    system_prompt = get_system_prompt(mode=args.search_mode)
    agent = AgenticSearchFramework(
        model=model, skill_registry=registry, mode=mode,
        max_iters=args.max_iters, system_prompt=system_prompt,
    )

    result = agent.run(args.query)
    print("FINAL ANSWER:")
    print(result.final_answer)
    print("\nTRACE JSON:")
    print(json.dumps([t.__dict__ for t in result.trace], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
