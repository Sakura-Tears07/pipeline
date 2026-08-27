#!/usr/bin/env python3
"""补跑单个 rank 分片，或合并已有 .runtime_shards_* 为完整输出。

与主 runtime 一样按 conversation 切 shard（保证 staged 前缀复用）。无两阶段 resume。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from common.runtime_online import (  # noqa: E402
    _avg,
    _pct,
    _shard_by_conversation,
    _worker,
    _write_ok_and_oom,
)


def _build_summary(outputs: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    e2e = [float((o.get("latency_ms") or {}).get("e2e") or 0) for o in outputs]
    e2e_strat = [float((o.get("latency_ms") or {}).get("e2e_strategy") or 0) for o in outputs]
    pref = [float((o.get("latency_ms") or {}).get("llm_prefill") or 0) for o in outputs]
    pred = [float((o.get("latency_ms") or {}).get("predict") or 0) for o in outputs]
    hidden = [float((o.get("latency_ms") or {}).get("hidden_predict") or 0) for o in outputs]
    qwait = [float((o.get("latency_ms") or {}).get("queue_wait") or 0) for o in outputs]
    matched = sum(1 for o in outputs if o.get("match_predictor"))

    summary: Dict[str, Any] = {
        "pipeline": meta["pipeline"],
        "runtime": "single_phase",
        "num_samples": len(outputs),
        "num_attempted": int(meta.get("num_attempted") or len(outputs)),
        "n_oom": int(meta.get("n_oom") or 0),
        "elapsed_sec": meta.get("elapsed_sec", 0.0),
        "throughput_req_per_sec": int(meta.get("num_attempted") or len(outputs))
        / max(float(meta.get("elapsed_sec") or 0.0), 1e-9),
        "num_gpus": meta["num_gpus"],
        "gpu_ids": meta["gpu_ids"],
        "mode": meta["mode"],
        "model": meta["model"],
        "predictor_mode": meta["predictor_backend"],
        "predictor_device": meta["predictor_device"],
        "prefetch_depth": int(meta["prefetch_depth"]),
        "llm_batch_size": int(meta["llm_batch_size"]),
        "use_modular_cache": True,
        "llm_model_class": meta["llm_model_class"],
        "framework": meta["framework"],
        "prompt_cache_root": meta["prompt_cache_root"],
        "load_in_4bit": bool(meta["load_in_4bit"]),
        "load_in_8bit": bool(meta["load_in_8bit"]),
        "max_context_tokens": meta["max_context_tokens"],
        "max_new_tokens": meta["max_new_tokens"],
        "function_prediction_accuracy": matched / max(len(outputs), 1),
        "matched": matched,
        "latency_ms": {
            "e2e_wall_mean": _avg(e2e),
            "e2e_strategy_mean": _avg(e2e_strat),
            "e2e_mean": _avg(e2e_strat),
            "e2e_p50": _pct(e2e_strat, 0.5),
            "e2e_p90": _pct(e2e_strat, 0.9),
            "e2e_p99": _pct(e2e_strat, 0.99),
            "predict_mean": _avg(pred),
            "hidden_predict_mean": _avg(hidden),
            "queue_wait_mean": _avg(qwait),
            "prefill_mean": _avg(pref),
        },
        "modes": {},
    }
    for arm in ("adaptive", "pruned", "full"):
        prefs = [
            float(((o.get("generations") or {}).get(arm) or {}).get("prefill_ms") or 0)
            for o in outputs
            if arm in (o.get("generations") or {})
        ]
        tots = [
            float(((o.get("generations") or {}).get(arm) or {}).get("total_ms") or 0)
            for o in outputs
            if arm in (o.get("generations") or {})
        ]
        if prefs:
            summary["modes"][arm] = {
                "n": len(prefs),
                "avg_prefill_ms": _avg(prefs),
                "avg_total_ms": _avg(tots),
            }
    if meta["mode"] == "adaptive":
        n_full = sum(1 for o in outputs if o.get("chosen_mode") == "full")
        n_pruned = sum(1 for o in outputs if o.get("chosen_mode") == "pruned")
        summary["adaptive"] = {
            "unc_metric": meta["unc_metric"],
            "unc_tau": meta["unc_tau"],
            "n_full_fallback": n_full,
            "n_pruned": n_pruned,
            "fallback_rate": n_full / max(len(outputs), 1),
            "note": "本进程只跑 adaptive；full 基线请另起 MODE=full",
        }
    pb = (meta["predictor_backend"] or "").strip().lower()
    if pb in ("none", "no", "off") or meta["mode"] == "full":
        summary["function_prediction_accuracy"] = None
        summary["matched"] = None
    if meta.get("recovered"):
        summary["recovered"] = True
        summary["recovered_note"] = meta.get("recovered_note", "")
    return summary


def _worker_kwargs_from_args(
    args: argparse.Namespace,
    *,
    rank: int,
    gpu_id: str,
    samples: List[Dict[str, Any]],
    global_indices: List[int],
    out_path: str,
) -> Dict[str, Any]:
    return dict(
        rank=rank,
        gpu_id=gpu_id,
        samples=samples,
        global_indices=global_indices,
        model=args.model,
        mode=args.mode,
        load_in_8bit=bool(args.load_in_8bit),
        load_in_4bit=bool(args.load_in_4bit),
        max_new_tokens=int(args.max_new_tokens),
        max_context_tokens=int(args.max_context_tokens),
        temperature=float(args.temperature),
        prompt_cache_root=args.prompt_cache_root,
        predictor_backend=args.predictor_backend,
        unc_metric=args.unc_metric,
        unc_tau=float(args.unc_tau),
        predictor_device=args.predictor_device,
        prefetch_depth=int(args.prefetch_depth),
        llm_batch_size=int(args.llm_batch_size),
        llm_model_class=args.llm_model_class,
        framework_label=args.framework,
        use_modular_cache=True,
        out_path=out_path,
        modular_schema_style=str(getattr(args, "modular_schema_style", "json_only")),
        sticky_full=bool(getattr(args, "sticky_full", True)),
        max_cached_schemas=int(getattr(args, "max_cached_schemas", 8)),
    )


def _stitch_rank_parts(shard_dir: Path, rank: int, recover_gpus: int) -> Path:
    """将 rank{N}_part*.json 按 _global_index 拼成 rank{N}.json。"""
    parts: List[Dict[str, Any]] = []
    for sub in range(recover_gpus):
        p = shard_dir / f"rank{rank}_part{sub}.json"
        if not p.is_file():
            raise FileNotFoundError(f"缺少子分片: {p}")
        parts.extend(json.loads(p.read_text(encoding="utf-8")))
    parts.sort(key=lambda x: int(x["_global_index"]))
    out_path = shard_dir / f"rank{rank}.json"
    out_path.write_text(json.dumps(parts, ensure_ascii=False), encoding="utf-8")
    print(f"[recover] stitched rank{rank} n={len(parts)} -> {out_path}", flush=True)
    return out_path


def cmd_run_rank(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    num_gpus = int(args.num_gpus)
    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if len(gpu_ids) != num_gpus:
        raise SystemExit(f"gpu_ids 长度 {len(gpu_ids)} != num_gpus {num_gpus}")
    rank = int(args.rank)
    if rank < 0 or rank >= num_gpus:
        raise SystemExit(f"rank 须在 [0, {num_gpus})")

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_path = shard_dir / f"rank{rank}.json"
    if out_path.is_file() and not args.force:
        raise SystemExit(f"已存在 {out_path}，加 --force 覆盖")

    shards, gidx_shards = _shard_by_conversation(data, num_gpus)
    shard = shards[rank]
    gidx = gidx_shards[rank]
    print(
        f"[recover] rank{rank} gpu={gpu_ids[rank]} samples={len(shard)} -> {out_path}",
        flush=True,
    )
    _worker(
        **_worker_kwargs_from_args(
            args,
            rank=rank,
            gpu_id=gpu_ids[rank],
            samples=shard,
            global_indices=gidx,
            out_path=str(out_path),
        )
    )


def cmd_run_rank_multi(args: argparse.Namespace) -> None:
    """把单个逻辑 rank 的数据切到多张 GPU 上并行补跑，再拼成 rank{N}.json。"""
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    num_gpus = int(args.num_gpus)
    recover_gpus = int(args.recover_gpus)
    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if len(gpu_ids) < recover_gpus:
        raise SystemExit(f"gpu_ids 数量 {len(gpu_ids)} < recover_gpus {recover_gpus}")
    rank = int(args.rank)
    if rank < 0 or rank >= num_gpus:
        raise SystemExit(f"rank 须在 [0, {num_gpus})")

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    final_path = shard_dir / f"rank{rank}.json"
    if final_path.is_file() and not args.force:
        raise SystemExit(f"已存在 {final_path}，加 --force 覆盖")

    shards, gidx_all = _shard_by_conversation(data, num_gpus)
    rank_samples = shards[rank]
    rank_gidx = gidx_all[rank]
    sub_shards, sub_local = _shard_by_conversation(rank_samples, recover_gpus)
    print(
        f"[recover] rank{rank} 多卡切分: total={len(rank_samples)} recover_gpus={recover_gpus}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    procs: List[mp.Process] = []
    t0 = time.perf_counter()
    for sub in range(recover_gpus):
        sub_samples = sub_shards[sub]
        out_path = shard_dir / f"rank{rank}_part{sub}.json"
        if not sub_samples:
            if not out_path.is_file():
                out_path.write_text("[]", encoding="utf-8")
            continue
        sub_gidx = [rank_gidx[i] for i in sub_local[sub]]
        if out_path.is_file() and not args.force:
            print(f"[recover] skip part{sub} (exists): {out_path}", flush=True)
            continue
        print(
            f"[recover] part{sub} gpu={gpu_ids[sub]} samples={len(sub_samples)} "
            f"gidx=[{sub_gidx[0]}..{sub_gidx[-1]}] -> {out_path}",
            flush=True,
        )
        p = ctx.Process(
            target=_worker,
            kwargs=_worker_kwargs_from_args(
                args,
                rank=sub,
                gpu_id=gpu_ids[sub],
                samples=sub_samples,
                global_indices=sub_gidx,
                out_path=str(out_path),
            ),
        )
        p.start()
        procs.append(p)

    failed = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True
            print(f"[recover] worker pid={p.pid} exitcode={p.exitcode}", flush=True)
    if failed:
        raise RuntimeError("rank 多卡补跑有 worker 失败")

    _stitch_rank_parts(shard_dir, rank, recover_gpus)
    elapsed = time.perf_counter() - t0
    print(f"[recover] rank{rank} 多卡补跑完成 elapsed={elapsed:.1f}s", flush=True)


def cmd_merge(args: argparse.Namespace) -> None:
    shard_dir = Path(args.shard_dir)
    num_gpus = int(args.num_gpus)
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    dataset_len = len(data)

    shard_paths = [shard_dir / f"rank{r}.json" for r in range(num_gpus)]
    missing = [str(p) for p in shard_paths if not p.is_file()]
    if missing:
        raise SystemExit(f"缺少分片: {missing}")

    merged: List[Optional[Dict[str, Any]]] = [None] * dataset_len
    for sp in shard_paths:
        part = json.loads(sp.read_text(encoding="utf-8"))
        for item in part:
            gidx = item.pop("_global_index")
            item.pop("_rank", None)
            merged[gidx] = item
    if any(x is None for x in merged):
        bad = sum(1 for x in merged if x is None)
        raise SystemExit(f"merge 不完整，缺 {bad}/{dataset_len} 条")

    outputs: List[Dict[str, Any]] = merged  # type: ignore
    for o in outputs:
        o["framework"] = args.framework
        o["prompt_cache_root"] = args.prompt_cache_root
        o["llm_batch_size"] = int(args.llm_batch_size)
        o["llm_model_class"] = args.llm_model_class
        o["use_modular_cache"] = True

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_attempted = len(outputs)
    ok, oom, oom_path = _write_ok_and_oom(out, outputs)
    outputs = ok

    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    meta = {
        "pipeline": args.pipeline,
        "mode": args.mode,
        "num_gpus": num_gpus,
        "gpu_ids": gpu_ids,
        "model": args.model,
        "predictor_backend": args.predictor_backend,
        "predictor_device": args.predictor_device,
        "prefetch_depth": int(args.prefetch_depth),
        "llm_batch_size": int(args.llm_batch_size),
        "llm_model_class": args.llm_model_class,
        "framework": args.framework,
        "prompt_cache_root": args.prompt_cache_root,
        "load_in_4bit": bool(args.load_in_4bit),
        "load_in_8bit": bool(args.load_in_8bit),
        "max_context_tokens": int(args.max_context_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "unc_metric": args.unc_metric,
        "unc_tau": float(args.unc_tau),
        "elapsed_sec": float(args.elapsed_sec or 0.0),
        "recovered": True,
        "recovered_note": args.recovered_note or "merged from partial runtime shards",
        "num_attempted": n_attempted,
        "n_oom": len(oom),
    }
    summary = _build_summary(outputs, meta)
    summary_path = out.parent / "runtime_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[recover] saved {out}", flush=True)
    print(f"[recover] saved {oom_path}", flush=True)
    print(f"[recover] saved {summary_path}", flush=True)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--gpu-ids", default="0,1,2,3")
    p.add_argument("--mode", default="adaptive")
    p.add_argument("--model", required=True)
    p.add_argument("--predictor-backend", default="qwen3_emb")
    p.add_argument("--predictor-device", default="cpu")
    p.add_argument("--unc-metric", default="margin_top12")
    p.add_argument("--unc-tau", type=float, default=1.4)
    p.add_argument("--prefetch-depth", type=int, default=4)
    p.add_argument("--llm-batch-size", type=int, default=1)
    p.add_argument("--modular-schema-style", default="json_only")
    p.add_argument("--llm-model-class", default="Llama2")
    p.add_argument("--framework", default="dev")
    p.add_argument("--prompt-cache-root", required=True)
    p.add_argument("--load-in-4bit", type=int, default=1)
    p.add_argument("--load-in-8bit", type=int, default=0)
    p.add_argument("--max-context-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--sticky-full", type=int, default=1)
    p.add_argument("--max-cached-schemas", type=int, default=32)


def main() -> None:
    parser = argparse.ArgumentParser(description="runtime 分片补跑 / 合并")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run-rank", help="补跑单个 rank 分片")
    _add_common_args(p_run)
    p_run.add_argument("--rank", type=int, required=True)
    p_run.add_argument("--shard-dir", required=True)
    p_run.add_argument("--force", action="store_true")
    p_run.set_defaults(func=cmd_run_rank)

    p_multi = sub.add_parser("run-rank-multi", help="单 rank 切多卡并行补跑")
    _add_common_args(p_multi)
    p_multi.add_argument("--rank", type=int, required=True)
    p_multi.add_argument("--shard-dir", required=True)
    p_multi.add_argument("--recover-gpus", type=int, default=4)
    p_multi.add_argument("--force", action="store_true")
    p_multi.set_defaults(func=cmd_run_rank_multi)

    p_merge = sub.add_parser("merge", help="合并 rank*.json 并写 summary")
    _add_common_args(p_merge)
    p_merge.add_argument("--shard-dir", required=True)
    p_merge.add_argument("--output", required=True)
    p_merge.add_argument("--pipeline", default="C")
    p_merge.add_argument("--elapsed-sec", type=float, default=0.0)
    p_merge.add_argument("--recovered-note", default="")
    p_merge.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.load_in_4bit = bool(int(args.load_in_4bit))
    args.load_in_8bit = bool(int(args.load_in_8bit))
    args.sticky_full = bool(int(getattr(args, "sticky_full", 1)))
    args.func(args)


if __name__ == "__main__":
    main()
