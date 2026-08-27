"""从 system 消息解析工具列表，并按 predictor 结果剪枝。"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Tuple


def get_system_message(sample: Dict) -> str:
    for msg in sample.get("conversations") or []:
        if msg.get("from") == "system":
            value = msg.get("value", "")
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ""


def extract_tools_from_system(system_value: str) -> List[Dict[str, Any]]:
    marker = "Specifically, you have access to the following APIs:"
    if marker not in system_value:
        return []
    tool_str = system_value.split(marker, 1)[1].strip()
    tools = None
    try:
        parsed = ast.literal_eval(tool_str)
        if isinstance(parsed, list):
            tools = parsed
    except Exception:
        pass
    if tools is None:
        try:
            parsed = json.loads(tool_str)
            if isinstance(parsed, list):
                tools = parsed
        except Exception:
            pass
    if not tools:
        return []
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(t)
    return out


def prune_tools(tools: List[Dict[str, Any]], pred_function: str | None) -> Tuple[List[Dict], List[Dict]]:
    if not pred_function:
        return list(tools), []
    kept, dropped = [], []
    for t in tools:
        if t.get("name") == pred_function:
            kept.append(t)
        else:
            dropped.append(t)
    if not kept:
        return list(tools), []
    return kept, dropped
