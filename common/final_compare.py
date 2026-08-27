"""八组最终实验对比：A pruned/full、B adaptive/full、C ori/dev × adaptive/full。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PIPELINE_ROOT = Path(__file__).resolve().parents[1]

GROUPS: List[Tuple[str, Path]] = [
    ("A/pruned", PIPELINE_ROOT / "A" / "output" / "pruned"),
    ("A/full", PIPELINE_ROOT / "A" / "output" / "full"),
    ("B/adaptive", PIPELINE_ROOT / "B" / "output" / "adaptive"),
    ("B/full", PIPELINE_ROOT / "B" / "output" / "full"),
    ("C/ori/adaptive", PIPELINE_ROOT / "C" / "output" / "ori" / "adaptive"),
    ("C/ori/full", PIPELINE_ROOT / "C" / "output" / "ori" / "full"),
    ("C/dev/adaptive", PIPELINE_ROOT / "C" / "output" / "dev" / "adaptive"),
    ("C/dev/full", PIPELINE_ROOT / "C" / "output" / "dev" / "full"),
]


def _load_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _cell(dir_path: Path) -> Optional[Dict[str, Any]]:
    summary = _load_json(dir_path / "runtime_summary.json")
    if not isinstance(summary, dict):
        return None
    recs = _load_json(dir_path / "runtime_outputs.json")
    lat = summary.get("latency_ms") or {}
    reuse = summary.get("cache_reuse") or {}
    chosen_tok = 0.0
    if isinstance(recs, list) and recs:
        chosen_tok = _avg(
            [float((r.get("prompt_stats") or {}).get("chosen_prompt_tokens") or 0) for r in recs]
        )
    acc = summary.get("function_prediction_accuracy")
    return {
        "n": int(summary.get("num_samples") or 0),
        "n_attempted": int(summary.get("num_attempted") or summary.get("num_samples") or 0),
        "n_oom": int(summary.get("n_oom") or (summary.get("vram") or {}).get("n_empty_or_error") or 0),
        "acc": None if acc is None else float(acc),
        "prefill": float(lat.get("prefill_mean") or 0),
        "ttft": float(lat.get("ttft_mean") or 0),
        "predict": float(lat.get("predict_mean") or 0),
        "queue_wait": float(lat.get("queue_wait_mean") or 0),
        "cache_oh": float(reuse.get("cache_overhead_mean_ms") or 0),
        "schema_reuse": float(reuse.get("schema_reuse_rate") or 0),
        "e2e": float(lat.get("e2e_strategy_mean") or lat.get("e2e_mean") or 0),
        "chosen_tok": chosen_tok,
        "peak_vram": float((summary.get("vram") or {}).get("peak_vram_mb_avg") or 0),
        "empty_rate": float((summary.get("vram") or {}).get("empty_rate") or 0),
        "dir": str(dir_path),
    }


def _acc_s(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4f}"


def _rel(new: float, base: float) -> str:
    if base <= 0 or new <= 0:
        return "—"
    pct = (1.0 - new / base) * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


_TABLE_HEAD = (
    "| 组 | n 成功/尝试 | acc | chosen tok | predict | queue_wait | cache oh | "
    "**prefill** | **TTFT** | e2e | schema reuse | **增量峰值 MB** | 空生成 |"
)
_TABLE_SEP = (
    "|----|-------------|-----|------------|---------|------------|----------|"
    "-------------|----------|-----|--------------|-----------------|--------|"
)


def _row(name: str, st: Optional[Dict[str, Any]]) -> str:
    if not st:
        return f"| {name} | — | — | — | — | — | — | — | — | — | — | — | — |"
    return (
        f"| {name} | **{st['n']}**/{st['n_attempted']} | {_acc_s(st['acc'])} | {st['chosen_tok']:.0f} | "
        f"{st['predict']:.1f} | {st['queue_wait']:.1f} | {st['cache_oh']:.1f} | "
        f"**{st['prefill']:.1f}** | **{st['ttft']:.1f}** | {st['e2e']:.1f} | "
        f"{st['schema_reuse']:.2%} | **{st['peak_vram']:.1f}** | {st['empty_rate']:.2%} |"
    )


PIPELINE_SPECS: Dict[str, Dict[str, Any]] = {
    "A": {
        "title": "Pipeline A 汇总：Qwen3-32B + modular KV（qwen_tools）",
        "groups": [
            ("pruned", PIPELINE_ROOT / "A" / "output" / "pruned"),
            ("full", PIPELINE_ROOT / "A" / "output" / "full"),
        ],
        "note": "pruned = 预测后单工具前缀；full = 完整工具前缀。两组独立进程。",
    },
    "B": {
        "title": "Pipeline B 汇总：Qwen3-32B + modular KV（adaptive / full）",
        "groups": [
            ("adaptive", PIPELINE_ROOT / "B" / "output" / "adaptive"),
            ("full", PIPELINE_ROOT / "B" / "output" / "full"),
        ],
        "note": "adaptive = 不确定度门控（margin_top12，sticky full）；full = 独立完整前缀基线。",
    },
    "C": {
        "title": "Pipeline C 汇总：CodeLlama-34B + modular KV（ori / dev）",
        "groups": [
            ("ori/adaptive", PIPELINE_ROOT / "C" / "output" / "ori" / "adaptive"),
            ("ori/full", PIPELINE_ROOT / "C" / "output" / "ori" / "full"),
            ("dev/adaptive", PIPELINE_ROOT / "C" / "output" / "dev" / "adaptive"),
            ("dev/full", PIPELINE_ROOT / "C" / "output" / "dev" / "full"),
        ],
        "note": "ori = 原始 PromptCache；dev = sidecar / 按层批量装载。json_only schema。",
    },
}


def write_pipeline_report(pipeline_name: str) -> Path:
    """A/B/C 各自一份汇总 report（各组明细仍在 output/<mode>/report.md）。"""
    key = str(pipeline_name or "").strip().upper()
    spec = PIPELINE_SPECS.get(key)
    if spec is None:
        raise KeyError(f"unknown pipeline {pipeline_name!r}, expected A/B/C")

    cells: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for name, d in spec["groups"]:
        cells.append((name, _cell(d)))

    lines = [
        f"# {spec['title']}",
        "",
        spec["note"],
        "",
        "各组独立 `report.md` 仍写在对应输出目录；本文件是该 pipeline 的横向汇总。",
        "OOM 空输出在各组 `runtime_oom.json`，不计入下方延时 / 显存均值。",
        "",
        _TABLE_HEAD,
        _TABLE_SEP,
    ]
    by_name: Dict[str, Optional[Dict[str, Any]]] = {}
    for name, st in cells:
        by_name[name] = st
        lines.append(_row(name, st))

    if key in ("A", "B"):
        strat_name = "pruned" if key == "A" else "adaptive"
        s, f = by_name.get(strat_name), by_name.get("full")
        lines += ["", f"## {strat_name} vs full", ""]
        if not s or not f:
            lines += ["- 缺一组结果，跑完两组后会自动补全。", ""]
        else:
            lines += [
                f"- prefill: {strat_name} **{s['prefill']:.1f} ms** → full **{f['prefill']:.1f} ms**"
                f"（策略相对 full {_rel(s['prefill'], f['prefill'])}）",
                f"- TTFT: {strat_name} **{s['ttft']:.1f} ms** → full **{f['ttft']:.1f} ms**"
                f"（{_rel(s['ttft'], f['ttft'])}）",
                f"- 增量峰值显存: {strat_name} **{s['peak_vram']:.1f} MB** → full **{f['peak_vram']:.1f} MB**"
                f"（{_rel(s['peak_vram'], f['peak_vram'])}）",
                f"- 空生成率: {strat_name} {s['empty_rate']:.2%} / full {f['empty_rate']:.2%}",
                "",
            ]
    else:
        lines += ["", "## ori vs dev", ""]
        for mode in ("adaptive", "full"):
            ori, dev = by_name.get(f"ori/{mode}"), by_name.get(f"dev/{mode}")
            lines.append(f"### MODE={mode}")
            if not ori or not dev:
                lines += ["- 缺一组结果。", ""]
                continue
            lines += [
                f"- prefill: ori **{ori['prefill']:.1f} ms** → dev **{dev['prefill']:.1f} ms**"
                f"（dev 相对 ori {_rel(dev['prefill'], ori['prefill'])}）",
                f"- TTFT: ori **{ori['ttft']:.1f} ms** → dev **{dev['ttft']:.1f} ms**"
                f"（{_rel(dev['ttft'], ori['ttft'])}）",
                f"- 增量峰值显存: ori **{ori['peak_vram']:.1f} MB** → dev **{dev['peak_vram']:.1f} MB**"
                f"（{_rel(dev['peak_vram'], ori['peak_vram'])}）",
                "",
            ]

    lines += ["## 各组明细", ""]
    for name, d in spec["groups"]:
        lines.append(f"- {name}: `{d}`")
        lines.append(f"  - `{d / 'report.md'}`")
        lines.append(f"  - `{d / 'runtime_oom.json'}`")
    lines.append("")

    out = PIPELINE_ROOT / key / "output" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[pipeline] saved {out}", flush=True)
    return out


def write_all_pipeline_reports() -> List[Path]:
    return [write_pipeline_report(name) for name in ("A", "B", "C")]


def write_final_compare() -> Path:
    rows: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for name, d in GROUPS:
        rows.append((name, _cell(d)))

    lines = [
        "# 最终实验（8 组，全部 modular KV）",
        "",
        "每组独立进程。A/B = Qwen3-32B + `qwen_tools` schema；C = CodeLlama-34B + `json_only` schema。",
        "主看 **TTFT / prefill / cache overhead / 增量峰值显存**。",
        "",
        _TABLE_HEAD,
        _TABLE_SEP,
    ]
    by_name: Dict[str, Optional[Dict[str, Any]]] = {}
    for name, st in rows:
        by_name[name] = st
        lines.append(_row(name, st))

    lines += ["", "## A / B：策略 vs independent full", ""]
    for strat, full in (("A/pruned", "A/full"), ("B/adaptive", "B/full")):
        s, f = by_name.get(strat), by_name.get(full)
        lines.append(f"### {strat} vs {full}")
        if not s or not f:
            lines.append("- 缺一组结果。")
            lines.append("")
            continue
        lines += [
            f"- prefill: 策略 **{s['prefill']:.1f} ms** → full **{f['prefill']:.1f} ms**"
            f"（策略相对 full {_rel(s['prefill'], f['prefill'])}）",
            f"- TTFT: 策略 **{s['ttft']:.1f} ms** → full **{f['ttft']:.1f} ms**"
            f"（{_rel(s['ttft'], f['ttft'])}）",
            f"- 增量峰值显存: 策略 **{s['peak_vram']:.1f} MB** → full **{f['peak_vram']:.1f} MB**"
            f"（{_rel(s['peak_vram'], f['peak_vram'])}）",
            "",
        ]

    lines += ["", "## C：ori vs dev", ""]
    for mode in ("adaptive", "full"):
        ori, dev = by_name.get(f"C/ori/{mode}"), by_name.get(f"C/dev/{mode}")
        lines.append(f"### MODE={mode}")
        if not ori or not dev:
            lines.append("- 缺一组结果。")
            lines.append("")
            continue
        lines += [
            f"- prefill: ori **{ori['prefill']:.1f} ms** → dev **{dev['prefill']:.1f} ms**"
            f"（dev 相对 ori {_rel(dev['prefill'], ori['prefill'])}）",
            f"- TTFT: ori **{ori['ttft']:.1f} ms** → dev **{dev['ttft']:.1f} ms**"
            f"（{_rel(dev['ttft'], ori['ttft'])}）",
            f"- cache oh: ori **{ori['cache_oh']:.1f} ms** → dev **{dev['cache_oh']:.1f} ms**",
            f"- 增量峰值显存: ori **{ori['peak_vram']:.1f} MB** → dev **{dev['peak_vram']:.1f} MB**"
            f"（{_rel(dev['peak_vram'], ori['peak_vram'])}）",
            "",
        ]

    lines += [
        "> 正数表示更快 / 更省。C 上不要用 e2e 判断框架（decode 占主导）。",
        "",
        "## 路径",
        "",
    ]
    for name, d in GROUPS:
        lines.append(f"- {name}: `{d}`")
    lines.append("")

    out = PIPELINE_ROOT / "output" / "final_compare.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    print(f"[final] saved {out}", flush=True)
    write_all_pipeline_reports()
    return out


if __name__ == "__main__":
    write_final_compare()
