"""从 ToolBench conversations 抽出 user 侧文本（供 modular prompt）。"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def build_conversation_text(sample: Dict[str, Any]) -> str:
    """去掉最后一轮 assistant（gold），拼接 user 轮次。"""
    conversations: List[Dict] = list(sample.get("conversations") or [])
    if conversations and conversations[-1].get("from") == "assistant":
        conversations = conversations[:-1]
    user_lines: List[str] = []
    for turn in conversations:
        if turn.get("from") != "user":
            continue
        raw = turn.get("value", "")
        if isinstance(raw, dict):
            raw = json.dumps(raw, ensure_ascii=False)
        user_lines.append(str(raw))
    return "\n".join(user_lines)
