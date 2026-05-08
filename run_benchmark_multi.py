#!/usr/bin/env python3
import argparse
import fcntl
import json
import multiprocessing
import os
from dataclasses import asdict
from pathlib import Path

from agentic_search import AgentMode, AgenticSearchFramework, load_model
from agentic_search.tools import default_skill_registry
from agentic_search.prompts import get_system_prompt


def load_raw_dataset(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ex in data:
        rows.append({
            "id": ex["id"],
            "question": ex["question_en"],
        })
    return rows


def clean_trace(trace):
    cleaned = []
    for step in trace:
        d = asdict(step)
        cleaned.append({
            "iteration": d["iteration"],
            "model_output": d["model_output"],
            "parsed_actions": d["parsed_actions"],
            "observations": d["observations"],
        })
    return cleaned


def make_safe_filename(name: str) -> str:
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    safe = name
    for ch in bad_chars:
        safe = safe.replace(ch, "__")
    return safe


def _write_jsonl_row(filepath, row):
    """Thread-safe append of a JSONL row to file."""
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with open(filepath, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def _recover_temp_files(outdir, safe_model_name, ans_path, result_path, memory_path):
    """Recover leftover temp files from a previous crash before starting."""
    outdir = Path(outdir)
    existing_ids = set()
    if ans_path.exists():
        with ans_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "id" in row:
                        existing_ids.add(row["id"])
                except json.JSONDecodeError:
                    continue

    recovered = 0
    for pattern_suffix in ["_ans.jsonl", "_result.jsonl", "_memory.jsonl"]:
        target_path = {
            "_ans.jsonl": ans_path,
            "_result.jsonl": result_path,
            "_memory.jsonl": memory_path,
        }[pattern_suffix]
        if target_path is None:
            continue

        temp_files = sorted(outdir.glob(f"_{safe_model_name}_w*{pattern_suffix}"))
        for tf in temp_files:
            with tf.open("r", encoding="utf-8") as f_in:
                with target_path.open("a", encoding="utf-8") as f_out:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            if row.get("id") not in existing_ids:
                                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                                existing_ids.add(row.get("id"))
                                recovered += 1
                        except json.JSONDecodeError:
                            continue
            tf.unlink()

    if recovered:
        print(f"[RECOVER] Recovered {recovered} rows from leftover temp files.")


def _worker_process(batch, worker_id, model_name, backend, max_iters, use_running_memory, ans_path, result_path, memory_path, mode_str, search_mode="web"):
    """Each worker independently creates model/agent instances and processes its batch."""
    from agentic_search import AgentMode, AgenticSearchFramework, load_model
    from agentic_search.tools import default_skill_registry
    from agentic_search.prompts import get_system_prompt

    ans_path = Path(ans_path)
    result_path = Path(result_path)
    memory_path = Path(memory_path) if memory_path else None

    mode = AgentMode.UNTIL_DONE if mode_str == "until_done" else AgentMode.FIXED_DEPTH
    model = load_model(model_name, backend=backend)
    registry = default_skill_registry(model=model, mode=search_mode)
    system_prompt = get_system_prompt(mode=search_mode)
    agent = AgenticSearchFramework(
        model=model,
        skill_registry=registry,
        mode=mode,
        max_iters=max_iters,
        use_running_memory=use_running_memory,
        system_prompt=system_prompt,
    )

    total = len(batch)

    for idx, ex in enumerate(batch, 1):
        print(f"[W{worker_id}][{idx}/{total}] id={ex['id']}")
        print(ex["question"])

        try:
            run_result = agent.run(ex["question"])
            final_answer = (run_result.final_answer or "").strip()

            print(f"[W{worker_id}][FINAL ANSWER] {final_answer}")

            for step in run_result.trace:
                print(f"\n===== [W{worker_id}] ITERATION {step.iteration} =====")
                for action, obs in zip(step.parsed_actions, step.observations):
                    print("\n[STEP ACTION]")
                    print(json.dumps(action, ensure_ascii=False, indent=2))
                    print("\n[STEP RESULT]")
                    print(json.dumps(obs, ensure_ascii=False, indent=2))

            ans_row = {
                "id": ex["id"],
                "question": ex["question"],
                "prediction": final_answer,
            }

            result_row = {
                "id": ex["id"],
                "question": ex["question"],
                "final_answer": final_answer,
                "done": run_result.done,
                "iterations": run_result.iterations,
                "steps": clean_trace(run_result.trace),
            }

        except Exception as e:
            print(f"[W{worker_id}][ERROR] id={ex['id']} failed: {type(e).__name__}: {e}")
            run_result = None

            ans_row = {
                "id": ex["id"],
                "question": ex["question"],
                "prediction": "",
                "error": f"{type(e).__name__}: {e}",
            }

            result_row = {
                "id": ex["id"],
                "question": ex["question"],
                "final_answer": "",
                "done": False,
                "iterations": 0,
                "steps": [],
                "error": f"{type(e).__name__}: {e}",
            }

        _write_jsonl_row(ans_path, ans_row)

        if memory_path is not None:
            final_memory = ""
            if run_result is not None and run_result.trace:
                final_memory = run_result.trace[-1].running_memory
            memory_row = {
                "id": ex["id"],
                "running_memory": final_memory,
            }
            _write_jsonl_row(memory_path, memory_row)

        _write_jsonl_row(result_path, result_row)

    print(f"[W{worker_id}] Done. Processed {total} items.")


def _merge_temp_files(outdir, safe_model_name, num_workers, final_ans_path, final_result_path, final_memory_path):
    """No-op: workers now write directly to final files."""
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original dataset JSON file.")
    parser.add_argument("--model", required=True, help="Model name, also used in output filename.")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--mode", default="until_done", choices=["until_done", "fixed_depth"])
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--search-mode", default="web", choices=["web", "local"],
                        help="Search mode: 'web' for internet search, 'local' for local KB retrieval.")
    parser.add_argument("--use-running-memory", action="store_true", default=False,
                        help="Enable running memory mode: use an extra model call per round to maintain a compressed memory, "
                             "and reduce observation window from 8 to 3.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    safe_model_name = make_safe_filename(args.model)

    question_path = outdir / "question_en.jsonl"
    ans_path = outdir / f"{safe_model_name}_ans.jsonl"
    result_path = outdir / f"{safe_model_name}_result.jsonl"
    memory_path = outdir / f"{safe_model_name}_memory.jsonl" if args.use_running_memory else None

    question_path.parent.mkdir(parents=True, exist_ok=True)
    ans_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_raw_dataset(Path(args.input))

    # Write question_en.jsonl
    with question_path.open("w", encoding="utf-8") as fq:
        for ex in dataset:
            fq.write(json.dumps({
                "id": ex["id"],
                "question": ex["question"],
            }, ensure_ascii=False) + "\n")
        fq.flush()

    total = len(dataset)

    # ---- recover leftover temp files from previous crash ----
    _recover_temp_files(
        outdir=outdir,
        safe_model_name=safe_model_name,
        ans_path=ans_path,
        result_path=result_path,
        memory_path=memory_path,
    )
    # ---- end recover logic ----

    # ---- checkpoint / resume logic ----
    completed_ids = set()
    if ans_path.exists():
        with ans_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "id" in row:
                        completed_ids.add(row["id"])
                except json.JSONDecodeError:
                    continue

    if completed_ids:
        print(f"\n{'='*60}")
        print(f"[RESUME] Found {len(completed_ids)} completed answers, "
              f"resuming from next unanswered question.")
        print(f"{'='*60}\n")
    # ---- end checkpoint logic ----

    # Filter out already completed items
    todo_items = [ex for ex in dataset if ex["id"] not in completed_ids]

    if not todo_items:
        print("All questions already completed. Nothing to do.")
    else:
        num_workers = args.num_workers
        print(f"[INFO] {len(todo_items)} questions to process with {num_workers} worker(s).")

        if num_workers <= 1:
            # Serial mode: call worker directly without Pool
            _worker_process(
                batch=todo_items,
                worker_id=0,
                model_name=args.model,
                backend=args.backend,
                max_iters=args.max_iters,
                use_running_memory=args.use_running_memory,
                ans_path=str(ans_path),
                result_path=str(result_path),
                memory_path=str(memory_path) if memory_path else None,
                mode_str=args.mode,
                search_mode=args.search_mode,
            )
        else:
            # Split todo_items into num_workers batches
            batches = [[] for _ in range(num_workers)]
            for i, item in enumerate(todo_items):
                batches[i % num_workers].append(item)

            # Remove empty batches
            worker_args = [
                (batch, wid, args.model, args.backend, args.max_iters, args.use_running_memory, str(ans_path), str(result_path), str(memory_path) if memory_path else None, args.mode, args.search_mode)
                for wid, batch in enumerate(batches)
                if len(batch) > 0
            ]

            actual_workers = len(worker_args)
            print(f"[INFO] Launching {actual_workers} workers...")

            with multiprocessing.Pool(actual_workers) as pool:
                pool.starmap(_worker_process, worker_args)

            print(f"[INFO] All workers finished.")

    # Final summary
    final_completed = set()
    if ans_path.exists():
        with ans_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "id" in row:
                        final_completed.add(row["id"])
                except json.JSONDecodeError:
                    continue

    newly_answered = len(final_completed) - len(completed_ids)
    print(json.dumps({
        "question_file": str(question_path),
        "ans_file": str(ans_path),
        "result_file": str(result_path),
        "num_examples": total,
        "num_resumed": len(completed_ids),
        "num_newly_answered": newly_answered,
        "model_name": args.model,
        "safe_model_name": safe_model_name,
        "num_workers": args.num_workers,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
