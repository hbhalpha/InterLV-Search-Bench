import argparse
import json

from agentic_search import AgentMode, AgenticSearchFramework, load_model
from agentic_search.framework.evaluator import BenchmarkRunner
from agentic_search.prompts import get_system_prompt
from agentic_search.tools import default_skill_registry


def simple_exact_match(pred: str, ex: dict) -> dict:
    gold = (ex.get("gold") or "").strip().lower()
    pred = (pred or "").strip().lower()
    return {"exact_match": float(pred == gold)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--mode", default="until_done", choices=["until_done", "fixed_depth"])
    parser.add_argument("--search-mode", default="web", choices=["web", "local"],
                        help="Search mode: 'web' for internet search, 'local' for local KB retrieval.")
    parser.add_argument("--max-iters", type=int, default=6)
    args = parser.parse_args()

    mode = AgentMode.UNTIL_DONE if args.mode == "until_done" else AgentMode.FIXED_DEPTH
    model = load_model(args.model, backend=args.backend)
    registry = default_skill_registry(model=model, mode=args.search_mode)
    system_prompt = get_system_prompt(mode=args.search_mode)
    agent = AgenticSearchFramework(
        model=model, skill_registry=registry, mode=mode,
        max_iters=args.max_iters, system_prompt=system_prompt,
    )
    runner = BenchmarkRunner(agent=agent, scorer=simple_exact_match)
    outputs = runner.run_jsonl(args.input, args.output)
    print(json.dumps({"num_examples": len(outputs), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
