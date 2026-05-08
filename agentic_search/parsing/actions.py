from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from agentic_search.exceptions import ActionParseError
from agentic_search.types import ParsedAction


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Top-level JSON is not an object.")
    except (json.JSONDecodeError, ValueError):
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")

    depth = 0
    end = None
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError(f"Could not find balanced JSON object in model output: {text[:200]}")

    snippet = text[start:end]
    return json.loads(snippet)


TAG_PATTERNS = {
    "done": re.compile(r"<done>(.*?)</done>", re.I | re.S),
    "query": re.compile(r"<query>(.*?)</query>", re.I | re.S),
    "code": re.compile(r"<code>(.*?)</code>", re.I | re.S),
    "clip": re.compile(r"<clip>(.*?)</clip>", re.I | re.S),
    "tool": re.compile(r"<tool\s+name=[\"\']([^\"\']+)[\"\']\s*>(.*?)</tool>", re.I | re.S),
}


def parse_actions(text: str) -> List[ParsedAction]:
    text = text.strip()
    matches: List[Tuple[int, ParsedAction]] = []

    for m in TAG_PATTERNS["done"].finditer(text):
        matches.append((m.start(), ParsedAction(action_type="done", content=m.group(1).strip(), raw=m.group(0))))

    for m in TAG_PATTERNS["query"].finditer(text):
        body = m.group(1).strip()
        payload = extract_json_object(body) if "{" in body else {"query": body}
        matches.append((m.start(), ParsedAction(action_type="query", content=body, payload=payload, raw=m.group(0))))

    for m in TAG_PATTERNS["code"].finditer(text):
        code = m.group(1).strip()
        matches.append((m.start(), ParsedAction(action_type="code", content=code, payload={"code": code}, raw=m.group(0))))

    for m in TAG_PATTERNS["clip"].finditer(text):
        body = m.group(1).strip()
        payload = extract_json_object(body) if "{" in body else {"image": body}
        matches.append((m.start(), ParsedAction(action_type="clip", content=body, payload=payload, raw=m.group(0))))

    for m in TAG_PATTERNS["tool"].finditer(text):
        tool_name = m.group(1).strip()
        body = m.group(2).strip()
        payload = extract_json_object(body) if body else {}
        matches.append(
            (
                m.start(),
                ParsedAction(
                    action_type="tool",
                    content=body,
                    tool_name=tool_name,
                    payload=payload,
                    raw=m.group(0),
                ),
            )
        )

    if matches:
        matches.sort(key=lambda x: x[0])
        return [item[1] for item in matches]

    if text:
        return [ParsedAction(action_type="text", content=text, raw=text)]

    raise ActionParseError("Empty model output; no actions found.")
