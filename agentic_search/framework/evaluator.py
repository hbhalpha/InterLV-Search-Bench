from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from agentic_search.framework.agent import AgenticSearchFramework


class BenchmarkRunner:
    """A simple evaluation harness for agentic-search experiments."""

    def __init__(self, agent: AgenticSearchFramework, scorer: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None):
        self.agent = agent
        self.scorer = scorer

    def run_examples(self, examples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for ex in examples:
            query = ex["query"]
            run_result = self.agent.run(query)
            row: Dict[str, Any] = {
                "id": ex.get("id"),
                "query": query,
                "gold": ex.get("gold"),
                "prediction": run_result.final_answer,
                "done": run_result.done,
                "iterations": run_result.iterations,
                "trace": [asdict(t) for t in run_result.trace],
            }
            if self.scorer:
                row["score"] = self.scorer(run_result.final_answer, ex)
            outputs.append(row)
        return outputs

    def run_jsonl(self, input_path: str | Path, output_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
        input_path = Path(input_path)
        examples = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        outputs = self.run_examples(examples)
        if output_path:
            output_path = Path(output_path)
            with output_path.open("w", encoding="utf-8") as f:
                for row in outputs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return outputs
