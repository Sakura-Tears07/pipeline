"""Pipeline ToolBench 样本 → find_model listwise 格式。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_FIND_MODEL = _PIPELINE_ROOT.parent / "find_model"
if str(_FIND_MODEL) not in sys.path:
    sys.path.insert(0, str(_FIND_MODEL))

# 与 find_model/src/sft_preprocess.INVALID_TOOLS 对齐
INVALID_TOOLS = {
    "",
    "Finish",
    "invalid_hallucination_function_name",
}


def _normalize_name_list(raw: Any) -> List[str]:
    names: List[str] = []
    seen = set()
    for item in raw or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        if not name or name in INVALID_TOOLS or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def resolve_candidate_names(sample: Dict[str, Any]) -> List[str]:
    """优先用划分数据里的 candidates（训练同款）；否则从 system 解析并过滤 Finish。"""
    if sample.get("candidates"):
        return _normalize_name_list(sample["candidates"])

    from common.prune import extract_tools_from_system, get_system_message

    tools = extract_tools_from_system(get_system_message(sample))
    return _normalize_name_list(tools)


def cap_listwise_candidates(lw: dict, max_candidates: int) -> dict:
    """推理时限制候选数 ≤ max_candidates（保留 gold 若其在截断范围外）。"""
    cands = list(lw.get("candidates") or [])
    if len(cands) <= max_candidates:
        return lw

    gold = str(lw.get("target_name") or "").strip()
    kept = cands[:max_candidates]
    if gold and not any(c["name"] == gold for c in kept):
        for c in cands[max_candidates:]:
            if c["name"] == gold:
                kept[-1] = c
                break

    out = dict(lw)
    out["candidates"] = kept
    out["label_index"] = next((i for i, c in enumerate(kept) if c["name"] == gold), -1)
    return out


def pipeline_sample_to_listwise(
    sample: Dict[str, Any],
    *,
    max_candidates: Optional[int] = None,
) -> dict:
    from src.sft_listwise import record_to_listwise

    names = resolve_candidate_names(sample)
    record = {
        "conversations": list(sample.get("conversations") or []),
        "candidates": names,
        "target": sample.get("target") or {},
        "sample_id": str(sample.get("sample_id") or sample.get("id") or ""),
        "split": sample.get("split"),
        "source": sample.get("source"),
    }
    lw = record_to_listwise(record, exclude_last_assistant=True)
    if max_candidates is not None and max_candidates > 0:
        lw = cap_listwise_candidates(lw, max_candidates)
    return lw
