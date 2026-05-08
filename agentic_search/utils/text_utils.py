from __future__ import annotations

import json
import os
import re
from typing import Any, Dict


def require_env(name: str, value: str | None = None) -> str:
    value = value or os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def clean_text(text: Any, max_len: int = 4000) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def safe_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def to_pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
