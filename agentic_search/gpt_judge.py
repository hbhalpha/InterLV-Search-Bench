#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from agentic_search.models.factory import load_model


def load_records(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def normalize_id(x) -> str:
    return str(x)


def get_question(row: dict) -> str:
    for k in ["question_en", "question", "question_zh"]:
        if k in row and row[k] not in [None, ""]:
            return str(row[k]).strip()
    return ""


def get_gold_answer(row: dict) -> str:
    for k in ["answer_en", "answer", "answer_zh", "page_title"]:
        if k in row and row[k] not in [None, ""]:
            return str(row[k]).strip()
    return ""


def parse_yes_no(text: str) -> str:
    text = str(text or "").strip().upper()
    m = re.search(r"\b(YES|NO)\b", text)
    if not m:
        return "NO"
    return m.group(1)


def extract_final_answer(pred: str) -> str:
    """
    Extract the final answer from prediction text, removing reasoning/thinking process.
    Supports common chain-of-thought tag formats.
    """
    # Remove <think>...</think> or <thinking>...</thinking> blocks
    pred = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", pred, flags=re.DOTALL | re.IGNORECASE)

    # Remove <reasoning>...</reasoning> blocks
    pred = re.sub(r"<reasoning>.*?</reasoning>", "", pred, flags=re.DOTALL | re.IGNORECASE)

    # Extract content from <answer>...</answer> or <final_answer>...</final_answer>
    m = re.search(r"<(?:final_)?answer>(.*?)</(?:final_)?answer>", pred, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Extract content after "Final Answer:" / "最终答案："
    m = re.search(r"(?:final\s+answer|最终答案)[:\s：]+(.+)", pred, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    return pred.strip()


def build_prompt(question: str, gold: str, pred: str) -> str:
    return f"""
You are an answer-equivalence judge.

You are given:
- QUESTION
- GOLD_ANSWER
- PREDICTED_ANSWER

IMPORTANT: The PREDICTED_ANSWER field may contain the model's internal reasoning or chain-of-thought process before the actual answer. You MUST ignore any reasoning, thinking steps, or intermediate analysis. Focus ONLY on the final answer stated by the model —  the conclusion after all reasoning.

Judge whether the final answer in PREDICTED_ANSWER should count as correct for this QUESTION, using GOLD_ANSWER as the reference.

Guidelines:
- Use the QUESTION only to understand answer context and granularity.
- Do NOT use outside knowledge.
- Do NOT invent hidden requirements not supported by QUESTION, GOLD_ANSWER, and PREDICTED_ANSWER.
- Be reasonably permissive.
- Ignore all reasoning steps, thinking processes, or intermediate conclusions — evaluate only the final answer.

Return YES if the predicted final answer is clearly the same answer as the gold answer in context, including:
- same answer in different wording
- alias / synonym / alternative surface form
- numeral vs word form
- full sentence containing the answer
- a more specific form of the same answer
- harmless extra descriptive words

Return NO if the predicted final answer is clearly a different answer.

Examples:
QUESTION: what Roman numeral marks the day on the tablet?
GOLD_ANSWER: IV
PREDICTED_ANSWER: The day is marked by IV.
YES

QUESTION: how many helmets appear on that reverse?
GOLD_ANSWER: Two
PREDICTED_ANSWER: Two women.
YES

QUESTION: what object is in its talons?
GOLD_ANSWER: A boomerang.
PREDICTED_ANSWER: Each kangaroo holds a boomerang.
YES

QUESTION: what is the floral emblem?
GOLD_ANSWER: A rose.
PREDICTED_ANSWER: The floral emblem is the White Rose of York.
YES

QUESTION: what year?
GOLD_ANSWER: 1970
PREDICTED_ANSWER: 1990
NO

Output exactly one token:
YES
or
NO

QUESTION: {question}
GOLD_ANSWER: {gold}
PREDICTED_ANSWER: {pred}
""".strip()


def iter_with_progress(rows: List[dict], desc: str):
    if tqdm is None:
        print(f"[START] {desc} total={len(rows)}")
        for i, row in enumerate(rows, 1):
            if i % 10 == 0 or i == len(rows):
                print(f"[{desc}] {i}/{len(rows)}")
            yield row
    else:
        yield from tqdm(rows, desc=desc, ncols=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Ground-truth dataset (.json / .jsonl)")
    parser.add_argument("--pred", required=True, help="Prediction file (.jsonl)")
    parser.add_argument("--judge-model", required=True, help="Judge model name")
    parser.add_argument("--out", default=None, help="Output jsonl path")
    parser.add_argument("--base-url", default=None, help="Optional base URL override")
    parser.add_argument("--api-key", default=None, help="Optional API key override")
    args = parser.parse_args()

    # If base_url / api_key are passed via CLI, set them as env vars for the model factory
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    dataset_path = Path(args.dataset)
    pred_path = Path(args.pred)
    out_path = Path(args.out) if args.out else pred_path.with_name(pred_path.stem + "_judge.jsonl")

    dataset_rows = load_records(dataset_path)
    dataset_by_id: Dict[str, dict] = {normalize_id(x["id"]): x for x in dataset_rows}

    pred_rows = load_records(pred_path)

    existing_ids = set()
    if out_path.exists():
        for row in load_records(out_path):
            if "id" in row:
                existing_ids.add(normalize_id(row["id"]))

    judge_model = load_model(args.judge_model, backend="openai")

    rows_to_run = []
    skipped = 0
    for row in pred_rows:
        rid = normalize_id(row.get("id"))
        if rid in existing_ids:
            skipped += 1
            continue
        rows_to_run.append(row)

    judged = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for row in iter_with_progress(rows_to_run, desc=f"judge:{pred_path.stem}"):
            rid = normalize_id(row.get("id"))
            gold_row = dataset_by_id.get(rid, {})

            question = get_question(gold_row)
            gold = get_gold_answer(gold_row)
            pred_raw = str(row.get("prediction", "")).strip()
            pred = extract_final_answer(pred_raw)

            prompt = build_prompt(question=question, gold=gold, pred=pred)
            judge_raw = judge_model.generate_response(prompt)
            verdict = parse_yes_no(judge_raw)

            out_row = {
                "id": row.get("id"),
                "question": question,
                "prediction": pred,
                "prediction_raw": pred_raw,
                "gold_answer": gold,
                "judge": verdict,
                "judge_mode": "llm_with_question_non_strict",
                "judge_raw": judge_raw,
            }

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()
            judged += 1

    summary = {
        "dataset": str(dataset_path),
        "pred": str(pred_path),
        "output": str(out_path),
        "judge_model": args.judge_model,
        "num_dataset_rows": len(dataset_rows),
        "num_pred_rows": len(pred_rows),
        "num_skipped_existing": skipped,
        "num_new_judged": judged,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
