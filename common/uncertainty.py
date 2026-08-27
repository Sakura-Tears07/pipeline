"""Predictor 不确定性：adaptive 门控决定 pruned vs full。"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


def margin_top12(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    if len(scores) == 1:
        return float("inf")
    ordered = sorted((float(x) for x in scores), reverse=True)
    return ordered[0] - ordered[1]


def softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    m = max(float(x) for x in scores)
    exps = [math.exp(float(x) - m) for x in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def softmax_gap(scores: Sequence[float]) -> float:
    """top1_prob - top2_prob；越小越不确定。"""
    probs = softmax(scores)
    if not probs:
        return 0.0
    if len(probs) == 1:
        return 1.0
    ordered = sorted(probs, reverse=True)
    return ordered[0] - ordered[1]


def top1_prob(scores: Sequence[float]) -> float:
    probs = softmax(scores)
    return max(probs) if probs else 0.0


def compute_uncertainty(
    scores: Sequence[float],
    *,
    metric: str = "margin_top12",
) -> Dict[str, float]:
    m = margin_top12(scores)
    gap = softmax_gap(scores)
    p1 = top1_prob(scores)
    out = {
        "margin_top12": float(m if math.isfinite(m) else 0.0),
        "softmax_gap": float(gap),
        "top1_prob": float(p1),
    }
    if metric == "softmax_gap":
        out["value"] = out["softmax_gap"]
    elif metric == "top1_prob":
        out["value"] = out["top1_prob"]
    else:
        out["value"] = out["margin_top12"]
    return out


def should_use_full(
    scores: Sequence[float],
    *,
    metric: str = "margin_top12",
    tau: float = 2.0,
) -> Tuple[bool, Dict[str, float]]:
    """不确定 → True（退回 full）；置信 → False（用 pruned）。

    约定：metric 越大越自信；value < tau 则不确定。
    """
    u = compute_uncertainty(scores, metric=metric)
    use_full = float(u["value"]) < float(tau)
    return use_full, u
