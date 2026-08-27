"""共享 prompt-cache / HF 大模型后端（支持 modular KV、8bit/4bit 与 batch generate）。"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


_MB = 1024.0 * 1024.0


@dataclass
class LLMResult:
    text: str
    prefill_ms: float
    total_ms: float
    prompt_tokens: int
    new_tokens: int
    backend: str = "prompt-cache"
    schema_reused: bool = False
    staged_reused: bool = False
    cache_overhead_ms: float = 0.0
    cache_source: str = ""
    baseline_allocated_mb: Optional[float] = None
    absolute_peak_vram_mb: Optional[float] = None
    peak_vram_mb: Optional[float] = None


def _vram_mark_begin() -> Optional[int]:
    """请求开始：记下当前占用并重置 CUDA peak，对齐论文增量峰值显存。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        baseline = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        return baseline
    except Exception:
        return None


def _vram_mark_end(result: LLMResult, baseline: Optional[int]) -> LLMResult:
    if baseline is None:
        return result
    try:
        import torch

        if not torch.cuda.is_available():
            return result
        torch.cuda.synchronize()
        abs_peak = int(torch.cuda.max_memory_allocated())
        result.baseline_allocated_mb = baseline / _MB
        result.absolute_peak_vram_mb = abs_peak / _MB
        result.peak_vram_mb = max(abs_peak - baseline, 0) / _MB
    except Exception:
        pass
    return result


def _legacy_kv_cache(past_key_values):
    """Transformers 5.x DynamicCache → prompt-cache 期望的 tuple[(k,v), ...]。"""
    if past_key_values is None:
        return None
    if isinstance(past_key_values, (list, tuple)):
        if len(past_key_values) == 0:
            return past_key_values
        first = past_key_values[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            return past_key_values
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        return tuple((layer.keys, layer.values) for layer in layers)
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def _pin_promptcache_root(prompt_cache_root: str | Path) -> str:
    """把指定 worktree 放到 sys.path 最前，并清掉已加载的 promptcache。"""
    root = str(Path(prompt_cache_root).resolve())
    drop_names = {"prompt-cache", "prompt-cache-ori", "prompt-cache-dev"}
    cleaned: List[str] = []
    for p in sys.path:
        if not p:
            cleaned.append(p)
            continue
        try:
            name = Path(p).resolve().name
        except Exception:
            cleaned.append(p)
            continue
        if name in drop_names and str(Path(p).resolve()) != root:
            continue
        cleaned.append(p)
    sys.path[:] = cleaned
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    doomed = [k for k in list(sys.modules) if k == "promptcache" or k.startswith("promptcache.")]
    for k in doomed:
        del sys.modules[k]
    return root


def _patch_lm_dynamic_cache_compat(lm) -> None:
    """cache_engine / GenerationEngine 需要 legacy KV；Transformers 5 默认返回 DynamicCache。"""
    try:
        from transformers.cache_utils import Cache, DynamicCache
    except ImportError:
        return

    lm_cls = type(lm)
    if getattr(lm_cls, "_pipeline_cache_compat_patched", False):
        return

    def _is_legacy(past_key_values) -> bool:
        if not isinstance(past_key_values, (list, tuple)):
            return False
        if len(past_key_values) == 0:
            return True
        first = past_key_values[0]
        return isinstance(first, (list, tuple)) and len(first) == 2

    orig_call = lm_cls.__call__

    def __call__(self, **kwargs):
        past_key_values = kwargs.get("past_key_values")
        kv_dtype = self.get_kv_dtype() if hasattr(self, "get_kv_dtype") else None
        if past_key_values is not None and kv_dtype is not None and _is_legacy(past_key_values):
            past_key_values = tuple(
                (k.to(dtype=kv_dtype), v.to(dtype=kv_dtype)) for k, v in past_key_values
            )
            kwargs["past_key_values"] = past_key_values
        if past_key_values is not None and DynamicCache is not None:
            is_cache_obj = Cache is not None and isinstance(past_key_values, Cache)
            if not is_cache_obj and _is_legacy(past_key_values):
                if hasattr(DynamicCache, "from_legacy_cache"):
                    kwargs["past_key_values"] = DynamicCache.from_legacy_cache(tuple(past_key_values))
                else:
                    new_cache = DynamicCache()
                    for i, (k, v) in enumerate(past_key_values):
                        new_cache.update(k, v, layer_idx=i)
                    kwargs["past_key_values"] = new_cache

        if "position_ids" in kwargs and kwargs.get("input_ids") is not None:
            input_len = kwargs["input_ids"].shape[1]
            if kwargs["position_ids"].shape[1] != input_len:
                kwargs["position_ids"] = kwargs["position_ids"][:, -input_len:]
            if kwargs.get("past_key_values") is not None:
                kwargs["cache_position"] = kwargs["position_ids"][0]

        out = orig_call(self, **kwargs)
        if hasattr(out, "past_key_values") and out.past_key_values is not None:
            out.past_key_values = _legacy_kv_cache(out.past_key_values)
        return out

    lm_cls.__call__ = __call__  # type: ignore[method-assign]
    lm_cls._pipeline_cache_compat_patched = True


def _is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg and ("cuda" in msg or "gpu" in msg)


def _is_cuda_device_assert(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "device-side assert" in msg or "cudaerrorassert" in msg


class PromptCacheLLM:
    def __init__(
        self,
        model_path: str,
        *,
        prompt_cache_root: str | Path,
        gpu_id: str = "0",
        max_new_tokens: int = 64,
        max_context_tokens: int = 8192,
        temperature: float = 0.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        batch_size: int = 1,
        llm_model_class: str = "Qwen",
        set_visible_devices: bool = True,
        use_modular_cache: bool = False,
        framework_label: str = "",
        max_cached_schemas: int = 8,
        modular_schema_style: str = "",
    ):
        self.model_path = model_path
        self.gpu_id = str(gpu_id)
        self.max_new_tokens = int(max_new_tokens)
        self.max_context_tokens = int(max_context_tokens)
        self.temperature = float(temperature)
        self.use_modular_cache = bool(use_modular_cache)
        self.framework_label = str(framework_label or "")
        self.batch_size = max(1, int(batch_size))
        self.llm_model_class = llm_model_class
        self._bound_schema_key: Optional[tuple] = None
        self._active_schema_name: Optional[str] = None
        self._last_conv_id: Optional[str] = None
        self._modular_calls = 0
        self.max_cached_schemas = max(1, int(max_cached_schemas))
        self.modular_schema_style = str(modular_schema_style or "").strip()

        if set_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

        root = Path(_pin_promptcache_root(prompt_cache_root))

        import promptcache  # noqa: F401
        from promptcache import CacheEngine, CompactSpaces, GenerationEngine, GenerationParameters, Prompt
        print(f"[llm] promptcache={getattr(promptcache, '__file__', '?')}", flush=True)

        llm_cls_name = (llm_model_class or "Qwen").strip()
        if llm_cls_name.lower() in ("llama", "llama2", "codellama"):
            from promptcache.model import Llama2 as _LM

            # CodeLlama-34B 与 Qwen3-32B 同级：4bit 才能塞进 32GB；8bit 会 OOM
            default_quant_4bit = True
            default_quant_8bit = False
        else:
            from promptcache.model import Qwen as _LM

            default_quant_4bit = True
            default_quant_8bit = False

        if load_in_4bit is False and load_in_8bit is False:
            load_in_4bit = default_quant_4bit
            load_in_8bit = default_quant_8bit

        quant_kwargs = {}
        dtype_kwargs = {}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
                import torch

                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_in_8bit = False
            except Exception as e:
                print(f"[llm] 4bit 不可用，回退: {e}", flush=True)
                load_in_4bit = False
        if load_in_8bit and not load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401

                quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as e:
                print(f"[llm] 8bit 不可用，改用 bf16/fp16: {e}", flush=True)
                load_in_8bit = False
        if not load_in_4bit and not load_in_8bit:
            import torch

            dtype_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

        print(
            f"[llm] loading {model_path} class={llm_cls_name} 4bit={load_in_4bit} 8bit={load_in_8bit} "
            f"batch_size={self.batch_size} modular={self.use_modular_cache} "
            f"cache_root={root} max_ctx={self.max_context_tokens} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )
        # 每进程只看见 1 张卡。device_map=auto 偶发把层拆到“不存在的卡”再打到 CPU。
        load_kwargs = {"device_map": {"": 0}, **quant_kwargs, **dtype_kwargs}
        self.lm = _LM(model_path, **load_kwargs)
        if self.use_modular_cache:
            _patch_lm_dynamic_cache_compat(self.lm)
        self.gen_engine = GenerationEngine(self.lm)
        self.GenerationParameters = GenerationParameters
        self.Prompt = Prompt
        tok = self.lm.hf_tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        hf_model = getattr(self.lm, "hf_model", None)
        if hf_model is not None and getattr(hf_model, "generation_config", None) is not None:
            hf_model.generation_config.max_length = None
        self.max_input_tokens = self._resolve_max_input_tokens(hf_model)
        if hf_model is not None:
            model_max = int(getattr(hf_model.config, "max_position_embeddings", 0) or 0)
            if model_max and self.max_input_tokens < self.max_context_tokens:
                print(
                    f"[llm] max_input_tokens={self.max_input_tokens} "
                    f"(model max_position={model_max}, reserve new={self.max_new_tokens})",
                    flush=True,
                )

        self.cache_engine = None
        self._preproc = None
        self.schema_max_tokens = max(1, self.max_input_tokens)
        if self.use_modular_cache:
            self.cache_engine = CacheEngine(int(max_context_tokens), self.lm)
            # 必须含 formatter：把 <user> 等标签换成 Llama chat 文本，否则 Prompt 解析失败
            self._preproc = [CompactSpaces(), self.lm.get_formatter()]
            self.schema_max_tokens = max(1, int(max_context_tokens) - int(max_new_tokens))

        print("[llm] ready", flush=True)

    def _modular_backend_tag(self) -> str:
        fw = self.framework_label or "unknown"
        return f"prompt-cache-modular:{fw}"

    def _resolve_max_input_tokens(self, hf_model) -> int:
        cap = self.max_context_tokens
        if hf_model is not None:
            model_max = int(getattr(hf_model.config, "max_position_embeddings", 0) or 0)
            if model_max > 0:
                cap = min(cap, model_max)
        return max(1, int(cap) - self.max_new_tokens)

    def reset_staged(self) -> None:
        """丢掉 GPU prompt cache 拼接结果；保留 schema L1（CPU pinned KV）。"""
        if self.cache_engine is None:
            return
        from common.prompt_cache_xml import reset_prompt_cache_session

        reset_prompt_cache_session(self.cache_engine)
        self._last_conv_id = None
        # 保留 schema L1 GPU 副本和 sidecar；只丢掉 working staged

    def _iter_l1_token_caches(self):
        engine = self.cache_engine
        if engine is None:
            return
        pc = getattr(engine, "prompt_cache", None)
        for m in getattr(pc, "staged", None) or []:
            yield m
        schemas = getattr(engine, "schemas", None) or {}
        for sc in schemas.values():
            for tsc in (getattr(sc, "cache_l1", None) or {}).values():
                yield tsc

    def _release_l1_gpu_copies(self) -> None:
        """L1 留在 CPU pin_memory；GPU 只保留 PromptCache 那一块拼接缓冲。"""
        seen = set()
        for m in self._iter_l1_token_caches():
            mid = id(m)
            if mid in seen:
                continue
            seen.add(mid)
            free = getattr(m, "free", None)
            if callable(free):
                free()

    def _reclaim_cuda_fragments(self, *, force: bool = False, label: str = "") -> None:
        """丢掉未引用的 GPU L1 副本；empty_cache 只在 OOM / 阶段边界用，热路径不要调。"""
        self._release_l1_gpu_copies()
        pc = getattr(self.cache_engine, "prompt_cache", None)
        drop = getattr(pc, "drop_sidecar", None)
        if callable(drop):
            drop()
        if not force:
            return
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                if label:
                    print(f"[llm] reclaim_cuda {label}", flush=True)
        except Exception:
            pass

    def _infer_schema_style(self) -> str:
        if self.modular_schema_style:
            return self.modular_schema_style
        cls = (self.llm_model_class or "").strip().lower()
        if cls in ("llama", "llama2", "codellama"):
            return "json_only"
        return "qwen_tools"

    def _sidecar_ready(self, schema_name: str) -> bool:
        pc = getattr(self.cache_engine, "prompt_cache", None) if self.cache_engine else None
        if pc is None:
            return False
        return (
            getattr(pc, "_sidecar", None) is not None
            and str(getattr(pc, "_sidecar_schema", "") or "") == schema_name
        )

    def _pruned_sidecars_ready(self, schema_name: str, tools: List[Dict[str, Any]]) -> bool:
        pc = getattr(self.cache_engine, "prompt_cache", None) if self.cache_engine else None
        if pc is None:
            return False
        pool = getattr(pc, "_pruned_pool", None) or {}
        n_tools = len([t for t in tools if t.get("name")])
        if n_tools <= 0:
            return True
        have = sum(1 for k in pool if k[0] == schema_name)
        return have >= n_tools

    def _warm_full_sidecar(self, system_prefix: str, tools: List[Dict[str, Any]]) -> None:
        """schema 就绪后预拼全工具前缀并 snapshot sidecar，避免 adaptive pruned→full 换臂 rebuild。"""
        if not self.use_modular_cache or self.cache_engine is None:
            return
        from common.prompt_cache_xml import (
            build_modular_prompt_text,
            build_qwen_modular_prompt_text,
            schema_cache_name,
        )

        schema_name = schema_cache_name(system_prefix, tools)
        if self._sidecar_ready(schema_name):
            return
        full_names = [str(t.get("name") or "") for t in tools if t.get("name")]
        style = self._infer_schema_style().lower()
        if style in ("qwen", "qwen_tools", "find_model_qwen"):
            prompt = build_qwen_modular_prompt_text(full_names, "", schema_name=schema_name)
        else:
            prompt = build_modular_prompt_text(full_names, [("user", "")], schema_name=schema_name)
        self._process_prompt(
            prompt,
            use_modular_cache=True,
            reset_staged=True,
            is_full_arm=True,
        )
        from common.prompt_cache_xml import reset_prompt_cache_session

        reset_prompt_cache_session(self.cache_engine)
        pc = self.cache_engine.prompt_cache
        src = str(getattr(pc, "last_update_source", "") or "")
        print(
            f"[llm] warm_full_sidecar schema={schema_name} source={src} "
            f"sidecar={'yes' if getattr(pc, '_sidecar', None) else 'no'}",
            flush=True,
        )

    def _warm_pruned_sidecars(self, system_prefix: str, tools: List[Dict[str, Any]]) -> None:
        """每个单工具 pruned 布局预 snapshot，避免 adaptive pruned 步 ~10ms rebuild。"""
        if not self.use_modular_cache or self.cache_engine is None:
            return
        from common.prompt_cache_xml import (
            build_modular_prompt_text,
            build_qwen_modular_prompt_text,
            reset_prompt_cache_session,
            schema_cache_name,
            unique_module_names,
        )

        schema_name = schema_cache_name(system_prefix, tools)
        if self._pruned_sidecars_ready(schema_name, tools):
            return
        tool_names = unique_module_names(
            [str(t.get("name") or "") for t in tools if t.get("name")]
        )
        style = self._infer_schema_style().lower()
        warmed = 0
        for name in tool_names:
            if style in ("qwen", "qwen_tools", "find_model_qwen"):
                prompt = build_qwen_modular_prompt_text([name], "", schema_name=schema_name)
            else:
                prompt = build_modular_prompt_text([name], [("user", "")], schema_name=schema_name)
            self._process_prompt(
                prompt,
                use_modular_cache=True,
                reset_staged=True,
                is_full_arm=False,
            )
            reset_prompt_cache_session(self.cache_engine)
            warmed += 1
        pc = self.cache_engine.prompt_cache
        pool = getattr(pc, "_pruned_pool", None) or {}
        have = sum(1 for k in pool if k[0] == schema_name)
        print(
            f"[llm] warm_pruned_sidecars schema={schema_name} warmed={warmed} pool={have}",
            flush=True,
        )

    def _process_prompt(
        self,
        prompt_text: str,
        *,
        use_modular_cache: bool,
        reset_staged: bool = False,
        is_full_arm: bool = False,
    ):
        assert self.cache_engine is not None and self._preproc is not None
        if reset_staged:
            from common.prompt_cache_xml import reset_prompt_cache_session

            reset_prompt_cache_session(self.cache_engine)
        out = self.cache_engine.process(
            self.Prompt(prompt_text, self._preproc),
            no_cache=not use_modular_cache,
            return_full_position_ids=getattr(self.lm, "use_full_position_ids", False),
            is_full_arm=is_full_arm,
        )
        return out

    def bind_schema(
        self,
        system_prefix: str,
        tools: List[Dict[str, Any]],
        *,
        warm_pruned: bool = True,
    ) -> bool:
        from common.prompt_cache_xml import bind_toolbench_schema, schema_cache_name

        # schema prefill 需要连续显存；只丢掉 GPU L1 副本，不要 empty_cache
        self._release_l1_gpu_copies()
        built = bind_toolbench_schema(
            self.cache_engine,
            self.lm,
            system_prefix,
            tools,
            schema_max_tokens=self.schema_max_tokens,
            preproc=self._preproc,
            schema_batch_size=self.batch_size,
            max_cached_schemas=self.max_cached_schemas,
        )
        self._active_schema_name = schema_cache_name(system_prefix, tools)
        if built:
            self._warm_full_sidecar(system_prefix, tools)
            if warm_pruned:
                self._warm_pruned_sidecars(system_prefix, tools)
        return bool(built)

    def bind_schema_if_needed(
        self,
        system_prefix: str,
        tools: List[Dict[str, Any]],
        *,
        warm_pruned: bool = True,
    ) -> bool:
        """命中 CacheEngine.schemas 则复用 L1；换对话也不再拆掉已缓存的工具 KV。"""
        from common.prompt_cache_xml import schema_cache_name, touch_schema_lru

        name = schema_cache_name(system_prefix, tools)
        schemas = getattr(self.cache_engine, "schemas", None) or {}
        if name in schemas:
            self._active_schema_name = name
            touch_schema_lru(self.cache_engine, name)
            if not self._sidecar_ready(name):
                self._warm_full_sidecar(system_prefix, tools)
            if warm_pruned and not self._pruned_sidecars_ready(name, tools):
                self._warm_pruned_sidecars(system_prefix, tools)
            return False
        return self.bind_schema(system_prefix, tools, warm_pruned=warm_pruned)

    def count_prompt_tokens(self, prompt_text: str) -> int:
        """prepare 阶段统计 token（无需先 bind schema）。"""
        if self.use_modular_cache:
            text = prompt_text
            for p in self._preproc:
                text = p(text)
            return len(self.lm.hf_tokenizer.encode(text, add_special_tokens=False))
        tok = self.lm.hf_tokenizer
        return len(tok.encode(prompt_text, add_special_tokens=True))

    def generate(self, prompt: str) -> LLMResult:
        return self.generate_batch([prompt])[0]

    def generate_for_prep(self, prep: Dict[str, Any], arm: str) -> LLMResult:
        """modular 路径：schema 用 full 工具集；pruned/full 只改 prompt 选中的 module。"""
        meta = prep["_pc_schema"]
        tools = meta["full_tools"]
        prompt = prep["_prompts"][arm]
        conv_id = str(prep.get("conversation_id") or "")
        is_full_arm = str(arm).strip().lower() == "full"
        last_exc: Optional[BaseException] = None
        for attempt in range(2):
            mem0 = _vram_mark_begin()
            try:
                prev_schema = self._active_schema_name
                rebound = self.bind_schema_if_needed(
                    meta["system_prefix"],
                    tools,
                    warm_pruned=not is_full_arm,
                )
                schema_changed = self._active_schema_name != prev_schema
                pc = self.cache_engine.prompt_cache if self.cache_engine is not None else None
                had_staged = bool(pc and getattr(pc, "staged", None))
                reset_staged = rebound or schema_changed or (conv_id != self._last_conv_id)
                schema_reused = not rebound
                staged_reused = (not reset_staged) and had_staged
                self._last_conv_id = conv_id
                result = self._generate_modular(
                    prompt,
                    reset_staged=reset_staged,
                    schema_reused=schema_reused,
                    staged_reused=staged_reused,
                    is_full_arm=is_full_arm,
                )
                self._modular_calls += 1
                return _vram_mark_end(result, mem0)
            except Exception as exc:
                last_exc = exc
                if _is_cuda_device_assert(exc):
                    # device-side assert 会毒化当前进程的 CUDA context，吞掉再跑 predictor 只会二次崩
                    print(
                        f"[llm] generate_for_prep arm={arm} CUDA assert (fatal): {exc}",
                        flush=True,
                    )
                    raise
                if attempt == 0 and _is_cuda_oom(exc):
                    print(
                        f"[llm] generate_for_prep arm={arm} OOM, reclaim and retry once: {exc}",
                        flush=True,
                    )
                    self._reclaim_cuda_fragments(force=True, label="after_oom")
                    continue
                print(
                    f"[llm] generate_for_prep arm={arm} failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                return _vram_mark_end(
                    LLMResult(
                        text="",
                        prefill_ms=0.0,
                        total_ms=0.0,
                        prompt_tokens=0,
                        new_tokens=0,
                        backend=f"{self._modular_backend_tag()}:error",
                    ),
                    mem0,
                )
        print(
            f"[llm] generate_for_prep arm={arm} failed: {type(last_exc).__name__}: {last_exc}",
            flush=True,
        )
        return LLMResult(
            text="",
            prefill_ms=0.0,
            total_ms=0.0,
            prompt_tokens=0,
            new_tokens=0,
            backend=f"{self._modular_backend_tag()}:error",
        )

    def generate_batch(self, prompts: List[str]) -> List[LLMResult]:
        if not prompts:
            return []
        if self.use_modular_cache:
            raise RuntimeError("modular 路径请使用 generate_for_prep(prep, arm)")
        if self.batch_size <= 1 or len(prompts) == 1:
            return [self._generate_one(p) for p in prompts]
        return self._generate_hf_batch(prompts)

    def _generation_params(self):
        # 忽略 EOS / stop string，一直 decode 到 max_new_tokens
        greedy = self.temperature <= 1e-5
        return self.GenerationParameters(
            temperature=0.0 if greedy else self.temperature,
            top_p=1.0 if greedy else 0.95,
            max_new_tokens=self.max_new_tokens,
            stop_token_ids=[],
            stop_str=[],
            echo=False,
        )

    def _generate_modular(
        self,
        prompt_text: str,
        *,
        reset_staged: bool = False,
        schema_reused: bool = False,
        staged_reused: bool = False,
        is_full_arm: bool = False,
    ) -> LLMResult:
        token_ids, position_ids, cache_overhead_ms, schema_cache = self._process_prompt(
            prompt_text,
            use_modular_cache=True,
            reset_staged=reset_staged,
            is_full_arm=is_full_arm,
        )
        pc = self.cache_engine.prompt_cache
        cache_source = str(getattr(pc, "last_update_source", "") or "")
        # modular KV 拼接顶满 ctx 说明发生 clip，回退 no_cache 保证生成正确
        if schema_cache is not None and pc.length >= pc.max_ctx_length:
            token_ids, position_ids, cache_overhead_ms, schema_cache = self._process_prompt(
                prompt_text, use_modular_cache=False, reset_staged=True
            )
            staged_reused = False
            cache_source = "fallback"

        if len(token_ids) > self.max_input_tokens:
            token_ids = token_ids[-self.max_input_tokens :]
            position_ids = position_ids[-self.max_input_tokens :]

        params = self._generation_params()
        t0 = time.perf_counter()
        text = ""
        prefill_ms = 0.0
        backend = self._modular_backend_tag()
        if schema_cache is None:
            backend = f"{backend}:no_kv_fallback"
        elif cache_source == "sidecar":
            backend = f"{backend}:sidecar_hit"
        elif staged_reused:
            backend = f"{backend}:staged_hit"
        elif schema_reused:
            backend = f"{backend}:schema_hit"
        for i, out in enumerate(
            self.gen_engine.generate(
                token_ids,
                position_ids,
                params,
                cache=schema_cache,
                use_full_position_ids=getattr(self.lm, "use_full_position_ids", False),
            )
        ):
            if i == 0:
                prefill_ms = float(getattr(out, "response_time", 0.0) or 0.0)
            text = out.new_text
        total_ms = (time.perf_counter() - t0) * 1000.0
        tok = self.lm.hf_tokenizer
        cls = str(self.llm_model_class or "").lower()
        if cls in ("llama", "llama2", "codellama") and text and not text.lstrip().startswith("{"):
            text = '{"function_call":' + text
        new_tokens = len(tok.encode(text, add_special_tokens=False)) if text else 0
        return LLMResult(
            text=text,
            prefill_ms=prefill_ms,
            total_ms=total_ms,
            prompt_tokens=len(token_ids),
            new_tokens=new_tokens,
            backend=backend,
            schema_reused=schema_reused,
            staged_reused=staged_reused and schema_cache is not None,
            cache_overhead_ms=float(cache_overhead_ms or 0.0),
            cache_source=cache_source,
        )

    def _generate_one(self, prompt: str) -> LLMResult:
        tok = self.lm.hf_tokenizer
        ids: List[int] = tok.encode(prompt, add_special_tokens=True)
        if len(ids) > self.max_input_tokens:
            ids = ids[-self.max_input_tokens :]
        position_ids = list(range(len(ids)))
        params = self._generation_params()
        t0 = time.perf_counter()
        text = ""
        prefill_ms = 0.0
        for i, out in enumerate(
            self.gen_engine.generate(
                ids,
                position_ids,
                params,
                cache=None,
                use_full_position_ids=getattr(self.lm, "use_full_position_ids", False),
            )
        ):
            if i == 0:
                prefill_ms = float(getattr(out, "response_time", 0.0) or 0.0)
            text = out.new_text
        total_ms = (time.perf_counter() - t0) * 1000.0
        new_tokens = len(tok.encode(text, add_special_tokens=False)) if text else 0
        return LLMResult(
            text=text,
            prefill_ms=prefill_ms,
            total_ms=total_ms,
            prompt_tokens=len(ids),
            new_tokens=new_tokens,
            backend="prompt-cache",
        )

    def _generate_hf_batch(self, prompts: List[str]) -> List[LLMResult]:
        import torch

        tok = self.lm.hf_tokenizer
        model = self.lm.hf_model
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        prompt_lens = attention_mask.sum(dim=1).tolist()

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 1e-5,
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": [],
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.95

        import copy

        gen_config = copy.deepcopy(model.generation_config)
        gen_config.update(**gen_kwargs)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
        total_ms = (time.perf_counter() - t0) * 1000.0
        per = total_ms / max(len(prompts), 1)

        results: List[LLMResult] = []
        for i, plen in enumerate(prompt_lens):
            gen = out_ids[i, int(plen) :]
            text = tok.decode(gen, skip_special_tokens=True)
            results.append(
                LLMResult(
                    text=text,
                    prefill_ms=per * 0.7,
                    total_ms=per,
                    prompt_tokens=int(plen),
                    new_tokens=int(gen.numel()),
                    backend=f"hf-batch:{self.batch_size}",
                )
            )
        return results
