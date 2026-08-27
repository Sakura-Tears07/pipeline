#!/usr/bin/env python3
"""在线 runtime：单阶段 modular KV。

每组独立进程（A pruned/full、B adaptive/full、C ori/dev × adaptive/full）。
CPU 预取 predictor，与 LLM decode 重叠。MODE=full 不加载 predictor。
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _parse_gpu_ids(gpu_ids: str, num_gpus: int) -> List[str]:
    ids = [x.strip() for x in gpu_ids.split(",") if x.strip() != ""]
    if not ids:
        ids = [str(i) for i in range(num_gpus)]
    if len(ids) < num_gpus:
        raise ValueError(f"gpu-ids 数量 ({len(ids)}) < num-gpus ({num_gpus})")
    return ids[:num_gpus]


def _resolve_predictor_device(spec: str) -> "torch.device":
    import torch

    s = (spec or "cpu").strip().lower()
    if s in ("cpu", "overlap"):
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        return torch.device("cuda:0")
    return torch.device(s)


def _choose_mode(
    mode: str,
    scores: List[float],
    *,
    unc_metric: str,
    unc_tau: float,
) -> tuple[str, Optional[Dict[str, Any]]]:
    from common.uncertainty import should_use_full

    if mode == "pruned":
        return "pruned", None
    if mode == "full":
        return "full", None
    if mode == "adaptive":
        use_full, unc = should_use_full(scores, metric=unc_metric, tau=unc_tau)
        chosen = "full" if use_full else "pruned"
        return chosen, {"use_full": use_full, "tau": unc_tau, "metric": unc_metric, **unc}
    raise ValueError(f"unknown mode: {mode}")


def _apply_sticky_full(prep: Dict[str, Any], latched: set) -> None:
    """对话内一旦 fallback full，后续轮次不再切回 pruned（避免换臂重拷工具 L1）。"""
    if str(prep.get("_mode") or "") != "adaptive":
        return
    conv = str(prep.get("conversation_id") or "")
    gate = prep.get("uncertainty_gate")
    if not isinstance(gate, dict):
        gate = {}
        prep["uncertainty_gate"] = gate
    gate.setdefault("gate_chosen", prep.get("chosen_mode"))
    latched_now = bool(conv and conv in latched)
    if latched_now and prep.get("chosen_mode") != "full":
        prep["chosen_mode"] = "full"
        prep["_strategy_arm"] = "full"
        ps = prep.get("prompt_stats")
        if isinstance(ps, dict):
            ps["chosen_prompt_tokens"] = int(ps.get("full_prompt_tokens") or ps.get("chosen_prompt_tokens") or 0)
        gate["sticky_latched"] = True
    else:
        gate["sticky_latched"] = False
    if prep.get("chosen_mode") == "full" and conv:
        latched.add(conv)


def _apply_sticky_full_batch(preps: List[Dict[str, Any]], latched: set) -> None:
    for prep in preps:
        _apply_sticky_full(prep, latched)


def _strategy_arm(mode: str, chosen_mode: str) -> str:
    if mode == "adaptive":
        return chosen_mode
    return mode


def _conversation_key(sample: Dict[str, Any]) -> str:
    from common.prompt_cache_xml import conversation_key

    return conversation_key(sample)


def _conversation_step(sample: Dict[str, Any]) -> int:
    from common.prompt_cache_xml import conversation_step

    return conversation_step(sample)


def _shard_by_conversation(
    data: List[Dict[str, Any]], num_gpus: int
) -> Tuple[List[List[Dict[str, Any]]], List[List[int]]]:
    """同一对话的所有 step 放同一 GPU，并按 step 排序，便于 staged 前缀复用。"""
    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    order: List[str] = []
    for i, sample in enumerate(data):
        key = _conversation_key(sample)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((i, sample))
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_gpus)]
    gidx: List[List[int]] = [[] for _ in range(num_gpus)]
    loads = [0] * num_gpus
    for key in order:
        items = sorted(groups[key], key=lambda x: (_conversation_step(x[1]), x[0]))
        rank = min(range(num_gpus), key=lambda r: loads[r])
        for orig_i, sample in items:
            shards[rank].append(sample)
            gidx[rank].append(orig_i)
        loads[rank] += len(items)
    return shards, gidx


def _build_modular_prompts(
    full_tools: List[Dict[str, Any]],
    kept_tools: List[Dict[str, Any]],
    modular_turns: Optional[List[Tuple[str, str]]] = None,
    *,
    schema_style: str = "json_only",
    system_value: str = "",
    conv_user: str = "",
) -> tuple[str, str, Dict[str, Any]]:
    from common.prompt_cache_xml import (
        build_modular_prompt_text,
        build_qwen_modular_prompt_text,
        modular_schema_system_prefix,
        qwen_schema_system_prefix,
        schema_cache_name,
    )

    style = (schema_style or "json_only").strip().lower()
    if style in ("qwen", "qwen_tools", "find_model_qwen"):
        system_prefix = qwen_schema_system_prefix(system_value)
        full_names = [str(t.get("name") or "") for t in full_tools if t.get("name")]
        pruned_names = [str(t.get("name") or "") for t in kept_tools if t.get("name")]
        schema_name = schema_cache_name(system_prefix, full_tools)
        full_prompt = build_qwen_modular_prompt_text(
            full_names, conv_user, schema_name=schema_name
        )
        pruned_prompt = build_qwen_modular_prompt_text(
            pruned_names, conv_user, schema_name=schema_name
        )
    else:
        system_prefix = modular_schema_system_prefix()
        full_names = [str(t.get("name") or "") for t in full_tools if t.get("name")]
        pruned_names = [str(t.get("name") or "") for t in kept_tools if t.get("name")]
        schema_name = schema_cache_name(system_prefix, full_tools)
        turns = list(modular_turns or [])
        full_prompt = build_modular_prompt_text(
            full_names, turns, schema_name=schema_name
        )
        pruned_prompt = build_modular_prompt_text(
            pruned_names, turns, schema_name=schema_name
        )
    schema_meta = {
        "system_prefix": system_prefix,
        "full_tools": full_tools,
        "pruned_tools": kept_tools,
        "schema_name": schema_name,
        "schema_style": style,
    }
    return full_prompt, pruned_prompt, schema_meta


def _prepare_one_no_predictor(
    sample: Dict[str, Any],
    *,
    n_tokens,
    modular_schema_style: str = "json_only",
) -> Dict[str, Any]:
    """无 predictor（MODE=full）：只建 full modular prompt。"""
    from common.prune import extract_tools_from_system, get_system_message
    from common.prompts import build_conversation_text
    from common.prompt_cache_xml import extract_modular_turns

    t0 = time.perf_counter()
    conv_user = build_conversation_text(sample)
    system_value = get_system_message(sample)
    full_tools = extract_tools_from_system(system_value)
    modular_turns = extract_modular_turns(sample)
    full_prompt, _, schema_meta = _build_modular_prompts(
        full_tools,
        full_tools,
        modular_turns,
        schema_style=modular_schema_style,
        system_value=system_value,
        conv_user=conv_user,
    )
    predict_ms = (time.perf_counter() - t0) * 1000.0

    gold = (sample.get("target") or {}).get("function_call")
    full_tok = n_tokens(full_prompt)
    pruned_prompt = full_prompt

    out: Dict[str, Any] = {
        "id": sample.get("id"),
        "sample_id": sample.get("sample_id") or sample.get("id"),
        "conversation_id": _conversation_key(sample),
        "gold_function": gold,
        "match_predictor": None,
        "prediction": {
            "function_call": None,
            "raw": "",
            "backend": "none",
            "scores": [],
            "candidates": [],
            "uncertainty": {},
        },
        "chosen_mode": "full",
        "uncertainty_gate": None,
        "prompt_stats": {
            "full_prompt_tokens": full_tok,
            "pruned_prompt_tokens": full_tok,
            "chosen_prompt_tokens": full_tok,
            "token_reduction_ratio": 0.0,
        },
        "tool_stats": {
            "original_tool_count": 0,
            "pruned_tool_count": 0,
            "kept_tools": [],
            "dropped_tool_count": 0,
        },
        "_prompts": {"pruned": pruned_prompt, "full": full_prompt},
        "_strategy_arm": "full",
        "_mode": "full",
        "_predict_ms": predict_ms,
        "_t_req0": t0,
        "_pc_schema": schema_meta,
        "_use_modular_cache": True,
    }
    return out


def _prepare_one(
    sample: Dict[str, Any],
    *,
    predictor,
    n_tokens,
    mode: str,
    unc_metric: str,
    unc_tau: float,
    modular_schema_style: str = "json_only",
) -> Dict[str, Any]:
    """只做 predict + 建 prompt + 门控（可与 LLM 重叠）。"""
    if predictor is None or mode == "full":
        return _prepare_one_no_predictor(
            sample,
            n_tokens=n_tokens,
            modular_schema_style=modular_schema_style,
        )

    from common.prune import extract_tools_from_system, get_system_message, prune_tools
    from common.prompts import build_conversation_text
    from common.prompt_cache_xml import extract_modular_turns
    from common.uncertainty import compute_uncertainty

    t_pred0 = time.perf_counter()
    conv_user = build_conversation_text(sample)
    pred = predictor.predict_sample(sample)
    predict_ms = (time.perf_counter() - t_pred0) * 1000.0

    system_value = get_system_message(sample)
    full_tools = extract_tools_from_system(system_value)
    kept, dropped = prune_tools(full_tools, pred.function_call)
    modular_turns = extract_modular_turns(sample)
    full_prompt, pruned_prompt, schema_meta = _build_modular_prompts(
        full_tools,
        kept,
        modular_turns,
        schema_style=modular_schema_style,
        system_value=system_value,
        conv_user=conv_user,
    )

    scores = list(getattr(pred, "scores", None) or [])
    candidates = list(getattr(pred, "candidates", None) or [])
    unc = (getattr(pred, "extras", None) or {}).get("uncertainty") or compute_uncertainty(scores)
    chosen_mode, gate = _choose_mode(mode, scores, unc_metric=unc_metric, unc_tau=unc_tau)
    strat_arm = _strategy_arm(mode, chosen_mode)

    gold = (sample.get("target") or {}).get("function_call")
    full_tok = n_tokens(full_prompt)
    pruned_tok = n_tokens(pruned_prompt)
    chosen_tok = pruned_tok if chosen_mode == "pruned" else full_tok

    result: Dict[str, Any] = {
        "id": sample.get("id"),
        "sample_id": sample.get("sample_id") or sample.get("id"),
        "conversation_id": _conversation_key(sample),
        "gold_function": gold,
        "match_predictor": pred.function_call == gold,
        "prediction": {
            "function_call": pred.function_call,
            "raw": (pred.raw or "")[:2000],
            "backend": pred.backend,
            "scores": scores,
            "candidates": candidates,
            "uncertainty": unc,
        },
        "chosen_mode": chosen_mode,
        "uncertainty_gate": gate,
        "prompt_stats": {
            "full_prompt_tokens": full_tok,
            "pruned_prompt_tokens": pruned_tok,
            "chosen_prompt_tokens": chosen_tok,
            "token_reduction_ratio": ((full_tok - pruned_tok) / full_tok) if full_tok else 0.0,
        },
        "tool_stats": {
            "original_tool_count": len(full_tools),
            "pruned_tool_count": len(kept),
            "kept_tools": [t.get("name") for t in kept],
            "dropped_tool_count": len(dropped),
        },
        "_prompts": {"pruned": pruned_prompt, "full": full_prompt},
        "_strategy_arm": strat_arm,
        "_mode": mode,
        "_predict_ms": predict_ms,
        "_t_req0": t_pred0,
        "_pc_schema": schema_meta,
        "_use_modular_cache": True,
    }
    return result


def _gen_to_out(gen) -> tuple[Dict[str, Any], Dict[str, float]]:
    wall = float(gen.total_ms)
    g = {
        "text": gen.text,
        "prompt_tokens": gen.prompt_tokens,
        "new_tokens": gen.new_tokens,
        "backend": gen.backend,
        "prefill_ms": float(gen.prefill_ms),
        "total_ms": float(gen.total_ms),
        "wall_ms": wall,
        "schema_reused": bool(getattr(gen, "schema_reused", False)),
        "staged_reused": bool(getattr(gen, "staged_reused", False)),
        "cache_overhead_ms": float(getattr(gen, "cache_overhead_ms", 0.0) or 0.0),
        "cache_source": str(getattr(gen, "cache_source", "") or ""),
        "baseline_allocated_mb": getattr(gen, "baseline_allocated_mb", None),
        "absolute_peak_vram_mb": getattr(gen, "absolute_peak_vram_mb", None),
        "peak_vram_mb": getattr(gen, "peak_vram_mb", None),
    }
    lat = {
        "wall_ms": wall,
        "prefill_ms": float(gen.prefill_ms),
        "total_ms": float(gen.total_ms),
    }
    return g, lat


def _llm_arms_once(
    llm,
    preps: List[Dict[str, Any]],
    *,
    arm: Optional[str] = None,
) -> List[tuple[Dict[str, Any], Dict[str, float]]]:
    if not preps:
        return []
    out: List[tuple[Dict[str, Any], Dict[str, float]]] = []
    for prep in preps:
        a = arm or str(prep["_strategy_arm"])
        out.append(_gen_to_out(llm.generate_for_prep(prep, a)))
    return out


class _PreparePrefetch:
    """后台 CPU prepare（predict+prompt+门控），与 GPU LLM 重叠。"""

    def __init__(
        self,
        *,
        predictor,
        n_tokens,
        mode: str,
        unc_metric: str,
        unc_tau: float,
        depth: int,
        modular_schema_style: str = "json_only",
    ) -> None:
        self._kwargs = dict(
            predictor=predictor,
            n_tokens=n_tokens,
            mode=mode,
            unc_metric=unc_metric,
            unc_tau=unc_tau,
            modular_schema_style=modular_schema_style,
        )
        self._depth = max(0, int(depth))
        self._futures: Dict[int, Future] = {}
        self._executor = ThreadPoolExecutor(max_workers=1) if self._depth > 0 else None

    def schedule(self, local_i: int, sample: Dict[str, Any]) -> None:
        if self._executor is None or local_i in self._futures:
            return
        self._futures[local_i] = self._executor.submit(_prepare_one, sample, **self._kwargs)

    def get(self, local_i: int, sample: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
        """返回 (prep, queue_wait_ms, hidden_predict_ms)。"""
        if self._executor is None:
            prep = _prepare_one(sample, **self._kwargs)
            return prep, 0.0, 0.0

        if local_i not in self._futures:
            self.schedule(local_i, sample)
        t0 = time.perf_counter()
        prep = self._futures.pop(local_i).result()
        wait_ms = (time.perf_counter() - t0) * 1000.0
        predict_ms = float(prep.get("_predict_ms") or 0.0)
        hidden_ms = max(0.0, predict_ms - wait_ms)
        return prep, wait_ms, hidden_ms

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)


def _finalize_record(prep: Dict[str, Any]) -> Dict[str, Any]:
    mode = prep.pop("_mode")
    predict_ms = float(prep.pop("_predict_ms"))
    queue_wait_ms = float(prep.pop("_queue_wait_ms", 0.0) or 0.0)
    hidden_predict_ms = float(prep.pop("_hidden_predict_ms", 0.0) or 0.0)
    prep.pop("_t_req0", None)
    prep.pop("_prompts", None)
    prep.pop("_pc_schema", None)
    prep.pop("_use_modular_cache", None)
    prep.pop("_strategy_arm", None)
    chosen_mode = prep.get("chosen_mode")

    generations = prep.pop("generations", {})
    arm_lat = prep.pop("_arm_lat", {})

    if mode == "adaptive" and isinstance(chosen_mode, str) and chosen_mode in generations:
        if "adaptive" not in generations:
            generations["adaptive"] = dict(generations[chosen_mode])
            arm_lat["adaptive"] = dict(arm_lat.get(chosen_mode) or {})

    strat_key = "adaptive" if mode == "adaptive" else chosen_mode
    strat = arm_lat.get(strat_key) or arm_lat.get(str(chosen_mode)) or {}

    strat_llm = float(strat.get("wall_ms") or 0.0)

    if hidden_predict_ms > 0.0 or queue_wait_ms > 0.0:
        e2e_strategy = queue_wait_ms + strat_llm
        ttft_wait = queue_wait_ms
    else:
        e2e_strategy = predict_ms + strat_llm
        ttft_wait = predict_ms

    gstrat = generations.get(strat_key) or generations.get(str(chosen_mode)) or {}
    cache_oh = float(gstrat.get("cache_overhead_ms") or 0.0)
    prefill_ms = float(strat.get("prefill_ms") or 0.0)
    ttft_ms = ttft_wait + cache_oh + prefill_ms

    prep["generations"] = generations
    prep["latency_ms"] = {
        "predict": predict_ms,
        "queue_wait": queue_wait_ms,
        "hidden_predict": hidden_predict_ms,
        "llm_wall_strategy": strat_llm,
        "llm_wall": strat_llm,
        "llm_prefill": prefill_ms,
        "llm_total": float(strat.get("total_ms") or 0.0),
        "cache_overhead": cache_oh,
        "ttft": ttft_ms,
        "ttft_wait": ttft_wait,
        "e2e_strategy": e2e_strategy,
        "e2e": e2e_strategy,
        "arms": arm_lat,
        "stream": "single_phase",
    }
    prep["generation"] = {
        "text": gstrat.get("text"),
        "prompt_tokens": gstrat.get("prompt_tokens"),
        "new_tokens": gstrat.get("new_tokens"),
        "backend": gstrat.get("backend"),
        "mode": strat_key,
        "peak_vram_mb": gstrat.get("peak_vram_mb"),
        "absolute_peak_vram_mb": gstrat.get("absolute_peak_vram_mb"),
        "baseline_allocated_mb": gstrat.get("baseline_allocated_mb"),
    }
    return prep


def _worker(
    rank: int,
    gpu_id: str,
    samples: List[Dict[str, Any]],
    global_indices: List[int],
    *,
    model: str,
    mode: str,
    load_in_8bit: bool,
    load_in_4bit: bool,
    max_new_tokens: int,
    max_context_tokens: int,
    temperature: float,
    prompt_cache_root: str,
    predictor_backend: str,
    unc_metric: str,
    unc_tau: float,
    predictor_device: str,
    prefetch_depth: int,
    llm_batch_size: int,
    llm_model_class: str,
    framework_label: str,
    use_modular_cache: bool,
    out_path: str,
    max_cached_schemas: int = 8,
    modular_schema_style: str = "json_only",
    sticky_full: bool = True,
) -> None:
    time.sleep(rank * 3.0)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT))

    import torch

    from common.find_model_predictor import make_predictor
    from common.prompt_cache_llm import PromptCacheLLM, _pin_promptcache_root

    _pin_promptcache_root(prompt_cache_root)

    assert torch.cuda.device_count() == 1, (
        f"rank{rank} expected 1 visible GPU, got {torch.cuda.device_count()}"
    )
    pred_dev = _resolve_predictor_device(predictor_device)
    depth = max(0, int(prefetch_depth))
    # MODE=full 只拼 full prompt，不需要 predictor；公平 TTFT 也不应串行等预测
    pb = (predictor_backend or "qwen3_emb").strip().lower()
    if mode == "full" or pb in ("none", "no", "off"):
        predictor = None
        pred_label = "none"
    else:
        predictor = make_predictor(predictor_backend, pred_dev)
        pred_label = str(pred_dev)
        if samples:
            try:
                t_w = time.perf_counter()
                predictor.predict_sample(samples[0])
                if pred_dev.type == "cuda":
                    torch.cuda.synchronize()
                print(
                    f"[runtime][rank{rank}] predictor warmup {(time.perf_counter()-t_w)*1000:.1f}ms",
                    flush=True,
                )
            except Exception as exc:
                print(f"[runtime][rank{rank}] predictor warmup failed: {exc}", flush=True)
    print(
        f"[runtime][rank{rank}] gpu={gpu_id} mode={mode} framework={framework_label} "
        f"llm={torch.cuda.get_device_name(0)} predictor={pred_label} prefetch={depth} "
        f"llm_batch={llm_batch_size} modular={use_modular_cache} cache={prompt_cache_root} "
        f"max_cached_schemas={max_cached_schemas} sticky_full={sticky_full}",
        flush=True,
    )

    llm = PromptCacheLLM(
        model,
        prompt_cache_root=prompt_cache_root,
        gpu_id="0",
        max_new_tokens=max_new_tokens,
        max_context_tokens=max_context_tokens,
        temperature=temperature,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        batch_size=llm_batch_size,
        llm_model_class=llm_model_class,
        set_visible_devices=False,
        use_modular_cache=use_modular_cache,
        framework_label=framework_label,
        max_cached_schemas=max_cached_schemas,
        modular_schema_style=str(modular_schema_style),
    )

    def n_tokens(text: str) -> int:
        return llm.count_prompt_tokens(text)

    results: List[Dict[str, Any]] = []
    matched = 0
    sticky_latched: set = set()
    t_stream0 = time.perf_counter()
    def _run_single_phase() -> None:
        nonlocal matched
        buf: List[Dict[str, Any]] = []
        prefetcher = _PreparePrefetch(
            predictor=predictor,
            n_tokens=n_tokens,
            mode=mode,
            unc_metric=unc_metric,
            unc_tau=unc_tau,
            depth=depth,
            modular_schema_style=modular_schema_style,
        )
        for ahead in range(min(depth, len(samples))):
            prefetcher.schedule(ahead, samples[ahead])

        def _flush_buf() -> None:
            nonlocal matched
            if not buf:
                return
            if sticky_full:
                _apply_sticky_full_batch(buf, sticky_latched)
            for prep, (g, lat) in zip(buf, _llm_arms_once(llm, buf)):
                arm = prep["_strategy_arm"]
                prep["generations"] = {arm: g}
                prep["_arm_lat"] = {arm: lat}
                item = _finalize_record(prep)
                item["_global_index"] = prep.pop("_global_index")
                item["_rank"] = rank
                matched += int(bool(item.get("match_predictor")))
                results.append(item)
            buf.clear()

        try:
            for local_i, (gidx, sample) in enumerate(zip(global_indices, samples)):
                for ahead in range(1, depth + 1):
                    j = local_i + ahead
                    if j < len(samples):
                        prefetcher.schedule(j, samples[j])
                prep, queue_wait_ms, hidden_predict_ms = prefetcher.get(local_i, sample)
                prep["_queue_wait_ms"] = queue_wait_ms
                prep["_hidden_predict_ms"] = hidden_predict_ms
                prep["_global_index"] = gidx
                buf.append(prep)
                if len(buf) >= llm_batch_size:
                    _flush_buf()
                if (local_i + 1) % 50 == 0 or local_i + 1 == len(samples):
                    if local_i + 1 == len(samples):
                        _flush_buf()
                    elapsed = time.perf_counter() - t_stream0
                    pf_note = f" prefetch={depth}" if depth > 0 else ""
                    print(
                        f"[runtime][rank{rank}/gpu{gpu_id}] {local_i+1}/{len(samples)} "
                        f"acc={matched/(local_i+1):.4f} throughput={((local_i+1)/elapsed):.3f} req/s{pf_note}",
                        flush=True,
                    )
        finally:
            prefetcher.close()

    _run_single_phase()

    if predictor is not None:
        predictor.close()
    Path(out_path).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(
        f"[runtime][rank{rank}/gpu{gpu_id}] done matched={matched}/{len(samples)} -> {out_path}",
        flush=True,
    )


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round((len(ys) - 1) * q))))
    return ys[i]


def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _is_failed_generation(rec: Dict[str, Any]) -> bool:
    """OOM / 空生成：backend 带 :error，或 generation.text 为空。"""
    g = rec.get("generation") or {}
    if ":error" in str(g.get("backend") or ""):
        return True
    return not str(g.get("text") or "").strip()


def _split_ok_oom(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ok: List[Dict[str, Any]] = []
    oom: List[Dict[str, Any]] = []
    for rec in records:
        if _is_failed_generation(rec):
            oom.append(rec)
        else:
            ok.append(rec)
    return ok, oom


def _write_ok_and_oom(out: Path, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
    """主结果只留成功生成；OOM 空输出写到同目录 runtime_oom.json。"""
    ok, oom = _split_ok_oom(records)
    out.write_text(json.dumps(ok, indent=2, ensure_ascii=False), encoding="utf-8")
    oom_path = out.parent / "runtime_oom.json"
    oom_path.write_text(json.dumps(oom, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[runtime] saved {out} n_ok={len(ok)} | {oom_path} n_oom={len(oom)}",
        flush=True,
    )
    return ok, oom, oom_path


def _cache_reuse_lines(reuse: Dict[str, Any], *, heading: str) -> List[str]:
    if not reuse:
        return []
    src = reuse.get("cache_source") or {}
    src_bits = ", ".join(f"{k}={int(v)}" for k, v in sorted(src.items())) if src else ""
    lines = [
        heading,
        f"- schema L1 复用: **{float(reuse.get('schema_reuse_rate') or 0):.2%}** "
        f"({int(reuse.get('schema_reused') or 0)}/{int(reuse.get('n_generations') or 0)})",
        f"- staged 前缀复用: **{float(reuse.get('staged_reuse_rate') or 0):.2%}** "
        f"({int(reuse.get('staged_reused') or 0)}/{int(reuse.get('n_generations') or 0)})",
        f"- cache overhead mean: **{float(reuse.get('cache_overhead_mean_ms') or 0):.1f} ms** "
        f"（staged {float(reuse.get('cache_overhead_staged_mean_ms') or 0):.1f} / "
        f"fresh {float(reuse.get('cache_overhead_fresh_mean_ms') or 0):.1f}）",
    ]
    if src_bits:
        lines.append(f"- cache_source: {src_bits}")
    lines.append("")
    return lines


def _report_footer(cfg, out_dir: Path) -> List[str]:
    return [
        "## 文件",
        "",
        f"- `{cfg.RUNTIME_OUTPUTS_JSON}`（成功生成）",
        f"- `{out_dir / 'runtime_oom.json'}`（OOM / 空输出，未计入主结果）",
        f"- `{out_dir / 'runtime_summary.json'}`",
        "",
    ]


def _oom_section_lines(oom_records: List[Dict[str, Any]], *, attempted: int) -> List[str]:
    n = len(oom_records)
    rate = n / max(attempted, 1)
    lines = [
        "## 3. OOM / 空输出（已从主结果分离）",
        "",
        f"- n=**{n}** / 尝试 {attempted}（{rate:.2%}）",
        "- 这些样本 `generation.text` 为空或 `backend` 含 `:error`，不进入延时、显存均值、生成样例。",
        "",
    ]
    if not oom_records:
        lines += ["- 本组无 OOM 空输出。", ""]
        return lines
    lines += [
        "| # | sample_id | conversation_id | gold | backend | peak_vram_mb |",
        "|---|-----------|-----------------|------|---------|--------------|",
    ]
    for i, r in enumerate(oom_records, 1):
        g = r.get("generation") or {}
        peak = g.get("peak_vram_mb")
        peak_s = "—" if peak is None else f"{float(peak):.1f}"
        sid = str(r.get("sample_id") or r.get("id") or "")
        conv = str(r.get("conversation_id") or "")
        gold = str(r.get("gold_function") or "")
        backend = str(g.get("backend") or "")
        lines.append(f"| {i} | `{sid}` | `{conv}` | `{gold}` | `{backend}` | {peak_s} |")
    lines.append("")
    return lines


def _write_report_ab(
    pipeline_name: str,
    cfg,
    out_dir: Path,
    summary: Dict,
    records: List,
    oom_records: Optional[List] = None,
) -> List[str]:
    mode = summary.get("mode")
    n = len(records)
    attempted = int(summary.get("num_attempted") or (n + len(oom_records or [])))
    matched = sum(1 for r in records if r.get("match_predictor"))
    e2e = [float((r.get("latency_ms") or {}).get("e2e") or 0) for r in records]
    pred = [float((r.get("latency_ms") or {}).get("predict") or 0) for r in records]
    qwait = [
        float((r.get("latency_ms") or {}).get("queue_wait") or 0)
        for r in records
        if (r.get("latency_ms") or {}).get("queue_wait")
    ]
    hidden = [
        float((r.get("latency_ms") or {}).get("hidden_predict") or 0)
        for r in records
        if (r.get("latency_ms") or {}).get("hidden_predict")
    ]
    pref = [float((r.get("latency_ms") or {}).get("llm_prefill") or 0) for r in records]
    llm_tot = [float((r.get("latency_ms") or {}).get("llm_total") or 0) for r in records]
    tok = [float((r.get("prompt_stats") or {}).get("chosen_prompt_tokens") or 0) for r in records]
    full_tok = [float((r.get("prompt_stats") or {}).get("full_prompt_tokens") or 0) for r in records]
    e2e_strat = [
        float(
            (r.get("latency_ms") or {}).get("e2e_strategy")
            or (r.get("latency_ms") or {}).get("e2e")
            or 0
        )
        for r in records
    ]
    modes = summary.get("modes") or {}

    lines = [
        f"# Pipeline {pipeline_name} Runtime Report"
        + (f" (framework={summary.get('framework')})" if pipeline_name.upper() == "C" else ""),
        "",
        "## 设定",
        "",
        f"- 数据流：**单阶段 modular KV**（每组独立进程）",
        f"- Predictor：**{summary.get('predictor_mode')}** on `{summary.get('predictor_device')}`",
        f"- 大模型：`{summary.get('model')}`",
        f"- Stage 策略：**{mode}**",
        f"- prefetch_depth=**{summary.get('prefetch_depth', 0)}** "
        f"predictor_device=`{summary.get('predictor_device')}`",
        f"- 4bit={summary.get('load_in_4bit')} max_ctx={summary.get('max_context_tokens')} "
        f"max_new={summary.get('max_new_tokens')} max_cached_schemas={summary.get('max_cached_schemas')}",
        f"- use_modular_cache=**{summary.get('use_modular_cache', False)}** "
        f"schema={summary.get('modular_schema_style', 'json_only')} "
        f"cache=`{summary.get('prompt_cache_root')}`",
    ]
    if mode == "adaptive":
        ad = summary.get("adaptive") or {}
        lines += [
            f"- uncertainty: `{ad.get('unc_metric')}` τ=**{ad.get('unc_tau')}** "
            f"（margin&lt;τ → full，否则 pruned）",
        ]
    lines += ["", "## 1. Predictor", ""]
    acc = summary.get("function_prediction_accuracy")
    matched_s = summary.get("matched")
    if acc is None:
        acc_line = "- **top1 accuracy: —**（本组不跑 predictor）"
    else:
        acc_line = (
            f"- **top1 accuracy: {float(acc):.4f}** "
            f"({int(matched_s or 0)}/{attempted})（相对全部尝试，含 OOM 样本的预测）"
        )
    lines += [
        f"- samples: **{n}** 成功生成 / **{attempted}** 尝试",
        acc_line,
        f"- avg chosen prompt tokens: **{_avg(tok):.1f}** (full baseline avg: {_avg(full_tok):.1f})",
        "",
    ]
    if mode == "adaptive":
        ad = summary.get("adaptive") or {}
        n_pruned = int(ad.get("n_pruned") or 0)
        n_fb = int(ad.get("n_full_fallback") or 0)
        ok_p = sum(
            1 for r in records if r.get("chosen_mode") == "pruned" and r.get("match_predictor")
        )
        ok_f = sum(
            1 for r in records if r.get("chosen_mode") == "full" and r.get("match_predictor")
        )
        lines += [
            "### 1.1 门控",
            f"- n_pruned: **{n_pruned}** ({n_pruned/max(n,1):.2%})",
            f"- n_fallback_full: **{n_fb}** ({n_fb/max(n,1):.2%})",
            f"- 裁剪子集 predictor acc: **{(ok_p/n_pruned if n_pruned else 0):.4f}**",
            f"- 回退子集 predictor acc: **{(ok_f/n_fb if n_fb else 0):.4f}**",
        ]
        if ad.get("sticky_full"):
            lines.append(
                f"- sticky full: **on**，本轮 latched {int(ad.get('n_sticky_latched') or 0)} 条"
            )
        lines.append("")

    lines += [
        "## 2. 在线延时（主指标）",
        "",
        f"- 墙钟吞吐: **{summary.get('throughput_req_per_sec'):.3f} req/s** "
        f"（elapsed={summary.get('elapsed_sec'):.1f}s, gpus={summary.get('num_gpus')}）",
        f"- **策略臂 e2e mean: {_avg(e2e_strat):.1f} ms** "
        f"(p50 {_pct(e2e_strat,0.5):.1f} / p90 {_pct(e2e_strat,0.9):.1f} / p99 {_pct(e2e_strat,0.99):.1f})",
        f"- 样本墙钟 e2e mean: **{_avg(e2e):.1f} ms**",
        f"- predict mean: **{_avg(pred):.2f} ms**",
    ]
    if hidden and _avg(hidden) > 0:
        lines.append(f"- hidden_predict mean: **{_avg(hidden):.2f} ms**（与 LLM 重叠部分）")
    if qwait and _avg(qwait) > 0:
        lines.append(f"- queue_wait mean: **{_avg(qwait):.2f} ms**（等预取完成）")
    ttft = [float((r.get("latency_ms") or {}).get("ttft") or 0) for r in records]
    if any(ttft):
        lines.append(
            f"- **TTFT mean: {_avg(ttft):.1f} ms** "
            f"（wait + cache overhead + prefill；不含 decode）"
        )
    lines += [
        f"- 策略臂 llm prefill mean: **{_avg(pref):.1f} ms**",
        f"- 策略臂 llm total mean: **{_avg(llm_tot):.1f} ms**",
        "",
    ]
    if modes:
        lines += ["### 2.1 分臂 Prefill（本 pipeline 独立推理）", ""]
        for arm in ("adaptive", "pruned", "full"):
            m = modes.get(arm)
            if not m:
                continue
            lines.append(
                f"- **{arm}**: n={m.get('n')} prefill={float(m.get('avg_prefill_ms') or 0):.1f} ms "
                f"total={float(m.get('avg_total_ms') or 0):.1f} ms"
            )
        if "full" in modes and ("adaptive" in modes or "pruned" in modes):
            strat = modes.get("adaptive") or modes.get("pruned") or {}
            fu = modes.get("full") or {}
            sp = float(strat.get("avg_prefill_ms") or 0)
            fp = float(fu.get("avg_prefill_ms") or 0)
            if fp > 0:
                lines.append(f"- 策略 vs full prefill 相对节省: **{(1.0 - sp / fp):.4f}**")
        lines.append("")

    reuse = summary.get("cache_reuse") or {}
    if reuse and summary.get("use_modular_cache"):
        lines += _cache_reuse_lines(reuse, heading="### 2.2 Modular KV 复用")

    vram = summary.get("vram") or {}
    if vram:
        lines += [
            "### 2.3 显存（请求增量峰值）",
            "",
            f"- **增量峰值显存 mean: {float(vram.get('peak_vram_mb_avg') or 0):.2f} MB** "
            f"(p50 {float(vram.get('peak_vram_mb_p50') or 0):.2f} / "
            f"p95 {float(vram.get('peak_vram_mb_p95') or 0):.2f} / "
            f"max {float(vram.get('peak_vram_mb_max') or 0):.2f})",
            f"- 绝对峰值 mean / max: "
            f"{float(vram.get('absolute_peak_vram_mb_avg') or 0):.1f} / "
            f"{float(vram.get('absolute_peak_vram_mb_max') or 0):.1f} MB",
            f"- 请求开始基线占用 mean: {float(vram.get('baseline_allocated_mb_avg') or 0):.1f} MB",
            f"- 空生成（OOM/error）已分离，见 §3：**{int(vram.get('n_empty_or_error') or 0)}** "
            f"({float(vram.get('empty_rate') or 0):.2%})，不计入增量峰值均值",
            f"- max_cached_schemas=**{summary.get('max_cached_schemas')}**",
            "",
        ]

    lines += ["### 生成样例（截断）", ""]
    for r in records[:3]:
        g = r.get("generation") or {}
        text = (g.get("text") or "").replace("\n", " ")[:180]
        lines.append(
            f"- id=`{r.get('id')}` chosen={r.get('chosen_mode')} "
            f"gold=`{r.get('gold_function')}` pred=`{(r.get('prediction') or {}).get('function_call')}` "
            f"e2e_strat={float((r.get('latency_ms') or {}).get('e2e_strategy') or 0):.0f}ms"
        )
        lines.append(f"  - {text}…")
        lines.append("")

    lines += [
        "> 本组是独立进程的单一策略臂。八组横向对比见 `pipeline/output/final_compare.md`。",
        "",
    ]
    lines += _oom_section_lines(list(oom_records or []), attempted=attempted)
    lines += _report_footer(cfg, out_dir)
    return lines


def write_report(pipeline_name: str, cfg_module: str = "config") -> None:
    cfg = importlib.import_module(cfg_module)
    out_dir = Path(cfg.RUNTIME_OUTPUTS_JSON).parent
    summary_path = out_dir / "runtime_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = json.loads(Path(cfg.RUNTIME_OUTPUTS_JSON).read_text(encoding="utf-8"))
    oom_path = out_dir / "runtime_oom.json"
    oom_records: List = []
    if oom_path.is_file():
        loaded = json.loads(oom_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            oom_records = loaded
    lines = _write_report_ab(pipeline_name, cfg, out_dir, summary, records, oom_records)
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    print(f"[runtime] saved {report}", flush=True)
    try:
        from common.final_compare import write_pipeline_report

        write_pipeline_report(pipeline_name)
    except Exception as exc:
        print(f"[runtime] pipeline report skip: {exc}", flush=True)


def main(pipeline_name: str = "unknown") -> None:
    cfg = importlib.import_module("config")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=cfg.DATASET_PATH)
    parser.add_argument("--output", type=str, default=str(cfg.RUNTIME_OUTPUTS_JSON))
    parser.add_argument(
        "--mode",
        type=str,
        default=getattr(cfg, "RUNTIME_MODE", "pruned"),
        choices=["pruned", "full", "adaptive"],
    )
    parser.add_argument("--limit", type=int, default=cfg.LIMIT)
    parser.add_argument("--model", type=str, default=cfg.LLM_MODEL)
    parser.add_argument("--num-gpus", type=int, default=getattr(cfg, "NUM_GPUS", 1))
    parser.add_argument("--gpu-ids", type=str, default=getattr(cfg, "GPU_IDS", "0"))
    parser.add_argument("--max-new-tokens", type=int, default=cfg.LLM_MAX_NEW_TOKENS)
    parser.add_argument("--max-context-tokens", type=int, default=cfg.LLM_MAX_CONTEXT_TOKENS)
    parser.add_argument("--temperature", type=float, default=cfg.LLM_TEMPERATURE)
    parser.add_argument(
        "--load-in-4bit",
        type=int,
        default=1 if getattr(cfg, "LLM_LOAD_IN_4BIT", True) else 0,
    )
    parser.add_argument(
        "--load-in-8bit",
        type=int,
        default=1 if getattr(cfg, "LLM_LOAD_IN_8BIT", False) else 0,
    )
    parser.add_argument(
        "--predictor-backend",
        type=str,
        default=getattr(cfg, "PREDICTOR_BACKEND", "qwen3_emb"),
    )
    parser.add_argument(
        "--unc-metric",
        type=str,
        default=getattr(cfg, "UNCERTAINTY_METRIC", "margin_top12"),
    )
    parser.add_argument(
        "--unc-tau",
        type=float,
        default=float(getattr(cfg, "UNCERTAINTY_TAU", 1.5)),
    )
    parser.add_argument(
        "--predictor-device",
        type=str,
        default=str(getattr(cfg, "PREDICTOR_DEVICE", "cpu")),
        help="cpu|cuda；默认 cpu，腾 GPU 给 LLM 并与推理重叠",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=int(getattr(cfg, "PREFETCH_DEPTH", 4)),
        help="预测预取队列深度；0=串行 predict→llm",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=int(getattr(cfg, "LLM_BATCH_SIZE", 1)),
        help="LLM micro-batch 大小",
    )
    parser.add_argument(
        "--prompt-cache-root",
        type=str,
        default=str(getattr(cfg, "PROMPT_CACHE_ROOT", PIPELINE_ROOT.parent / "prompt-cache")),
    )
    parser.add_argument(
        "--llm-model-class",
        type=str,
        default=str(getattr(cfg, "LLM_MODEL_CLASS", "Qwen")),
        help="Qwen | Llama2 | CodeLlama",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default=str(getattr(cfg, "PROMPT_CACHE_FRAMEWORK", "dev")),
        help="prompt-cache 框架标签：ori | dev",
    )
    parser.add_argument(
        "--use-modular-cache",
        type=int,
        default=1 if getattr(cfg, "USE_MODULAR_CACHE", True) else 0,
        help="1=XML schema + modular KV（最终实验只支持 1）",
    )
    parser.add_argument(
        "--modular-schema-style",
        type=str,
        default=str(getattr(cfg, "MODULAR_SCHEMA_STYLE", "json_only")),
        help="json_only（C/Llama 短指令）| qwen_tools（A/B 原 system+工具 module）",
    )
    parser.add_argument(
        "--sticky-full",
        type=int,
        default=1 if getattr(cfg, "STICKY_FULL", True) else 0,
        help="1=adaptive 对话内一旦 full 则后续轮次保持 full",
    )
    args = parser.parse_args()
    if not bool(args.use_modular_cache):
        raise SystemExit("最终实验只支持 modular KV，请传 --use-modular-cache 1")

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    num_gpus = max(1, int(args.num_gpus))
    gpu_ids = _parse_gpu_ids(args.gpu_ids, num_gpus)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out.parent / f".runtime_shards_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    prompt_cache_root = str(args.prompt_cache_root)

    cfg.RUNTIME_OUTPUTS_JSON = out
    if hasattr(cfg, "REPORT_MD"):
        cfg.REPORT_MD = out.parent / "report.md"

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    t0 = time.perf_counter()
    print(
        f"[runtime][{pipeline_name}] samples={len(data)} mode={args.mode} "
        f"num_gpus={num_gpus} gpu_ids={gpu_ids} model={args.model} "
        f"unc_metric={args.unc_metric} unc_tau={args.unc_tau} "
        f"predictor_device={args.predictor_device} prefetch={args.prefetch_depth} "
        f"llm_batch={args.llm_batch_size} modular={bool(args.use_modular_cache)} "
        f"schema={args.modular_schema_style} "
        f"sticky_full={bool(args.sticky_full)} "
        f"framework={args.framework} "
        f"cache={args.prompt_cache_root} llm_class={args.llm_model_class}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    procs = []
    shard_paths = []
    shards, shard_gidx = _shard_by_conversation(data, num_gpus)
    for rank in range(num_gpus):
        shard = shards[rank]
        gidx = shard_gidx[rank]
        shard_path = tmp_dir / f"rank{rank}.json"
        shard_paths.append(shard_path)
        p = ctx.Process(
            target=_worker,
            kwargs=dict(
                rank=rank,
                gpu_id=gpu_ids[rank],
                samples=shard,
                global_indices=gidx,
                model=args.model,
                mode=args.mode,
                load_in_8bit=bool(args.load_in_8bit),
                load_in_4bit=bool(args.load_in_4bit),
                max_new_tokens=int(args.max_new_tokens),
                max_context_tokens=int(args.max_context_tokens),
                temperature=float(args.temperature),
                prompt_cache_root=prompt_cache_root,
                predictor_backend=args.predictor_backend,
                unc_metric=args.unc_metric,
                unc_tau=float(args.unc_tau),
                predictor_device=str(args.predictor_device),
                prefetch_depth=int(args.prefetch_depth),
                llm_batch_size=int(args.llm_batch_size),
                llm_model_class=str(args.llm_model_class),
                framework_label=str(args.framework),
                use_modular_cache=True,
                out_path=str(shard_path),
                max_cached_schemas=int(getattr(cfg, "MAX_CACHED_SCHEMAS", 8)),
                modular_schema_style=str(args.modular_schema_style),
                sticky_full=bool(args.sticky_full),
            ),
        )
        p.start()
        procs.append(p)

    failed = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True
            print(f"[runtime] worker pid={p.pid} exitcode={p.exitcode}", flush=True)
    if failed:
        raise RuntimeError("runtime 多卡 worker 失败")

    merged: List[Optional[Dict[str, Any]]] = [None] * len(data)
    for sp in shard_paths:
        part = json.loads(sp.read_text(encoding="utf-8"))
        for item in part:
            gidx = item.pop("_global_index")
            item.pop("_rank", None)
            merged[gidx] = item
    if any(x is None for x in merged):
        raise RuntimeError("runtime merge 不完整")

    outputs: List[Dict[str, Any]] = merged  # type: ignore
    for o in outputs:
        o["framework"] = str(args.framework)
        o["prompt_cache_root"] = prompt_cache_root
        o["llm_batch_size"] = int(args.llm_batch_size)
        o["llm_model_class"] = str(args.llm_model_class)
        o["use_modular_cache"] = bool(args.use_modular_cache)
    all_outputs = outputs
    n_attempted = len(all_outputs)
    ok, oom, _oom_path = _write_ok_and_oom(out, all_outputs)
    outputs = ok

    elapsed = time.perf_counter() - t0
    e2e = [float((o.get("latency_ms") or {}).get("e2e") or 0) for o in outputs]
    e2e_strat = [float((o.get("latency_ms") or {}).get("e2e_strategy") or 0) for o in outputs]
    pref = [float((o.get("latency_ms") or {}).get("llm_prefill") or 0) for o in outputs]
    pred = [float((o.get("latency_ms") or {}).get("predict") or 0) for o in outputs]
    hidden = [float((o.get("latency_ms") or {}).get("hidden_predict") or 0) for o in outputs]
    qwait = [float((o.get("latency_ms") or {}).get("queue_wait") or 0) for o in outputs]
    ttft = [float((o.get("latency_ms") or {}).get("ttft") or 0) for o in outputs]
    matched = sum(1 for o in all_outputs if o.get("match_predictor"))

    summary: Dict[str, Any] = {
        "pipeline": pipeline_name,
        "runtime": "single_phase",
        "num_attempted": n_attempted,
        "num_samples": len(outputs),
        "n_oom": len(oom),
        "elapsed_sec": elapsed,
        "throughput_req_per_sec": n_attempted / max(elapsed, 1e-9),
        "num_gpus": num_gpus,
        "gpu_ids": gpu_ids,
        "mode": args.mode,
        "model": args.model,
        "predictor_mode": args.predictor_backend,
        "predictor_device": str(args.predictor_device),
        "prefetch_depth": int(args.prefetch_depth),
        "llm_batch_size": int(args.llm_batch_size),
        "llm_model_class": str(args.llm_model_class),
        "framework": str(args.framework),
        "prompt_cache_root": prompt_cache_root,
        "use_modular_cache": bool(args.use_modular_cache),
        "modular_schema_style": str(args.modular_schema_style),
        "load_in_4bit": bool(args.load_in_4bit),
        "load_in_8bit": bool(args.load_in_8bit),
        "max_context_tokens": args.max_context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_cached_schemas": int(getattr(cfg, "MAX_CACHED_SCHEMAS", 8)),
        "function_prediction_accuracy": matched / max(n_attempted, 1),
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
            "ttft_mean": _avg(ttft),
            "prefill_mean": _avg(pref),
        },
        "modes": {},
    }
    peak_ok: List[float] = []
    abs_peak_ok: List[float] = []
    baseline_ok: List[float] = []
    n_empty = len(oom)
    for o in outputs:
        g = o.get("generation") or {}
        pv = g.get("peak_vram_mb")
        if pv is not None:
            peak_ok.append(float(pv))
        av = g.get("absolute_peak_vram_mb")
        if av is not None:
            abs_peak_ok.append(float(av))
        bv = g.get("baseline_allocated_mb")
        if bv is not None:
            baseline_ok.append(float(bv))
    summary["vram"] = {
        "peak_vram_mb_avg": _avg(peak_ok),
        "peak_vram_mb_p50": _pct(peak_ok, 0.5),
        "peak_vram_mb_p95": _pct(peak_ok, 0.95),
        "peak_vram_mb_max": max(peak_ok) if peak_ok else 0.0,
        "absolute_peak_vram_mb_avg": _avg(abs_peak_ok),
        "absolute_peak_vram_mb_max": max(abs_peak_ok) if abs_peak_ok else 0.0,
        "baseline_allocated_mb_avg": _avg(baseline_ok),
        "n_measured": len(peak_ok),
        "n_empty_or_error": n_empty,
        "empty_rate": n_empty / max(n_attempted, 1),
        "oom_path": str(out.parent / "runtime_oom.json"),
        "note": (
            "增量峰值显存 = max_memory_allocated - 请求开始时 memory_allocated；"
            "不含模型权重常驻。OOM 空输出在 runtime_oom.json，不进入 mean。"
        ),
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
    n_gen = n_schema = n_staged = 0
    overhead_all: List[float] = []
    overhead_staged: List[float] = []
    overhead_fresh: List[float] = []
    source_counts: Dict[str, int] = {}
    for o in outputs:
        gens = o.get("generations") or {}
        keys: List[str] = []
        if "pruned" in gens:
            keys.append("pruned")
        elif "adaptive" in gens:
            keys.append("adaptive")
        if "full" in gens:
            keys.append("full")
        for k in keys:
            g = gens.get(k)
            if not isinstance(g, dict):
                continue
            n_gen += 1
            n_schema += int(bool(g.get("schema_reused")))
            n_staged += int(bool(g.get("staged_reused")))
            src = str(g.get("cache_source") or "")
            if src:
                source_counts[src] = source_counts.get(src, 0) + 1
            oh = float(g.get("cache_overhead_ms") or 0.0)
            overhead_all.append(oh)
            if g.get("staged_reused"):
                overhead_staged.append(oh)
            else:
                overhead_fresh.append(oh)
    if n_gen:
        summary["cache_reuse"] = {
            "n_generations": n_gen,
            "schema_reused": n_schema,
            "schema_reuse_rate": n_schema / n_gen,
            "staged_reused": n_staged,
            "staged_reuse_rate": n_staged / n_gen,
            "cache_overhead_mean_ms": _avg(overhead_all),
            "cache_overhead_staged_mean_ms": _avg(overhead_staged),
            "cache_overhead_fresh_mean_ms": _avg(overhead_fresh),
            "cache_source": source_counts,
            "note": (
                "schema=复用工具 L1 KV；staged=同对话前缀增量；"
                "cache_source=hit/prefix/sidecar/rebuild（dev sidecar=全工具前缀一次 memcpy）"
            ),
        }
    if args.mode == "adaptive":
        n_full = sum(1 for o in all_outputs if o.get("chosen_mode") == "full")
        n_pruned = sum(1 for o in all_outputs if o.get("chosen_mode") == "pruned")
        ok_pruned = sum(
            1
            for o in all_outputs
            if o.get("chosen_mode") == "pruned" and o.get("match_predictor")
        )
        ok_fallback = sum(
            1 for o in all_outputs if o.get("chosen_mode") == "full" and o.get("match_predictor")
        )
        summary["adaptive"] = {
            "unc_metric": args.unc_metric,
            "unc_tau": args.unc_tau,
            "n_full_fallback": n_full,
            "n_pruned": n_pruned,
            "fallback_rate": n_full / max(n_attempted, 1),
            "pruned_predictor_acc": ok_pruned / n_pruned if n_pruned else 0.0,
            "pruned_predictor_matched": ok_pruned,
            "fallback_predictor_acc": ok_fallback / n_full if n_full else 0.0,
            "fallback_predictor_matched": ok_fallback,
            "n_sticky_latched": sum(
                1
                for o in all_outputs
                if ((o.get("uncertainty_gate") or {}).get("sticky_latched"))
            ),
            "sticky_full": bool(args.sticky_full),
            "note": "本进程只跑 adaptive；full 基线请另起 MODE=full",
        }

    pb = (args.predictor_backend or "").strip().lower()
    if pb in ("none", "no", "off") or args.mode == "full":
        summary["function_prediction_accuracy"] = None
        summary["matched"] = None
    summary_path = out.parent / "runtime_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[runtime] saved {out}", flush=True)

    for sp in shard_paths:
        try:
            sp.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main("unknown")
