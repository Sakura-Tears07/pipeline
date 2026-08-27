"""ToolBench / find_model 的 Prompt Cache XML 构造（Pipeline C modular 路径）。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

def escape_xml(text: str) -> str:
    # 延迟 import，避免在 PromptCacheLLM 钉住 worktree 之前绑到错误的 promptcache
    from promptcache.prompt import escape_xml as _escape_xml

    return _escape_xml(text)

API_MARKER = "Specifically, you have access to the following APIs:"

# 对齐 find_model/src/sft_prompt.py llama 前缀：JSON function_call，避免 ReAct 长指令
FIND_MODEL_LLAMA_SCHEMA_SYSTEM = """You are a helpful assistant that uses tools via function calls.

Your task:
Given a multi-turn tool-using conversation (API docs may appear in the history),
predict which function should be called next.

You must output a JSON object ONLY, in the following format:

{"function_call": "<function_name>"}

Do not explain anything.
Do not output extra text."""


def sanitize_module_name(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not name:
        return "fn_unknown"
    if re.match(r"^[0-9]", name) or not re.match(r"^[A-Za-z_]", name):
        name = f"fn_{name}"
    return name


def unique_module_name_map(names: List[str]) -> Dict[str, str]:
    """原始工具名 → 唯一 XML module 名。同名只保留第一次；sanitize 冲突加 _2/_3。"""
    used: set[str] = set()
    mapping: Dict[str, str] = {}
    for raw in names:
        raw = str(raw or "").strip()
        if not raw or raw in mapping:
            continue
        base = sanitize_module_name(raw)
        cand = base
        n = 2
        while cand in used:
            cand = f"{base}_{n}"
            n += 1
        used.add(cand)
        mapping[raw] = cand
    return mapping


def unique_module_names(names: List[str]) -> List[str]:
    mapping = unique_module_name_map(names)
    out: List[str] = []
    seen: set[str] = set()
    for raw in names:
        raw = str(raw or "").strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        out.append(mapping[raw])
    return out


def _tool_module_json(tool: Dict[str, Any]) -> str:
    name = tool["name"]
    payload = {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_toolbench_schema_xml(
    system_prefix: str, tools: List[Dict[str, Any]], *, schema_name: str = "function-call"
) -> str:
    mapping = unique_module_name_map([str(t.get("name") or "") for t in tools])
    modules = []
    seen: set[str] = set()
    for tool in tools:
        raw_name = str(tool.get("name") or "").strip()
        if not raw_name or raw_name in seen:
            continue
        seen.add(raw_name)
        mod_name = mapping[raw_name]
        body = escape_xml(_tool_module_json(tool))
        modules.append(f'<module name="{mod_name}">\n{body}\n</module>')
    body = "\n".join(modules)
    sys_text = escape_xml(system_prefix or "")
    return f"""<schema name="{schema_name}">
    <system>
{sys_text}
(tools)
{body}
(/tools)
    </system>
</schema>"""


def _format_user_xml(text: str) -> str:
    text = escape_xml((text or "").strip())
    return f"        <user>\n            {text}\n        </user>"


def _format_assistant_xml(text: str) -> str:
    text = escape_xml((text or "").strip())
    return f"        <assistant>\n            {text}\n        </assistant>"


def _stringify_modular_turn_value(value: Any, role: str) -> str:
    if isinstance(value, dict):
        if role == "assistant" and "function_call" in value:
            fc = value["function_call"]
            if isinstance(fc, dict) and fc.get("name"):
                return str(fc["name"])
        return json.dumps(value, ensure_ascii=False)
    text = str(value or "").strip()
    if role == "assistant":
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "function_call" in parsed:
                fc = parsed["function_call"]
                if isinstance(fc, dict) and fc.get("name"):
                    return str(fc["name"])
        except Exception:
            pass
    return text


def extract_modular_turns(sample: Dict[str, Any]) -> List[Tuple[str, str]]:
    """find_model 多轮对话 → modular XML  turn 列表（去掉 system 与 target assistant）。"""
    conversations = list(sample.get("conversations") or [])
    if conversations and conversations[-1].get("from") == "assistant":
        conversations = conversations[:-1]
    turns: List[Tuple[str, str]] = []
    for turn in conversations:
        role = str(turn.get("from") or "").strip()
        if role == "system" or not role:
            continue
        text = _stringify_modular_turn_value(turn.get("value", ""), role)
        if not text:
            continue
        if role == "function":
            raw = turn.get("value", "")
            if isinstance(raw, dict) and not (raw.get("response") or raw.get("error")):
                continue
            role = "assistant"
        turns.append((role, text))
    return turns


def build_modular_prompt_text(
    tool_names: List[str],
    turns: List[Tuple[str, str]],
    *,
    schema_name: str = "function-call",
) -> str:
    mod_names = unique_module_names(tool_names)
    tool_tags = "\n".join(f"        <{name}/>" for name in mod_names)
    body_parts: List[str] = []
    for role, text in turns:
        if role == "user":
            body_parts.append(_format_user_xml(text))
        elif role == "assistant":
            body_parts.append(_format_assistant_xml(text))
        else:
            body_parts.append(_format_user_xml(f"{role}: {text}"))
    body = "\n".join(body_parts)
    # 生成前缀：引导 Llama 直接续写 JSON（对齐 find_model fair 协议）
    body += '\n        <assistant>\n            {"function_call":'
    return f"""
        <prompt schema='{schema_name}'>
{tool_tags}
{body}
        </prompt>
        """


def modular_schema_system_prefix(_system_value: str = "") -> str:
    """Pipeline C + find_model：schema 用 JSON-only 指令，不用样本内 ReAct system。"""
    return FIND_MODEL_LLAMA_SCHEMA_SYSTEM


def qwen_schema_system_prefix(system_value: str) -> str:
    """A/B modular：保留 find_model 原 system 指令 + API 标记，工具本身走 <module>。"""
    if API_MARKER in (system_value or ""):
        prefix = system_value.split(API_MARKER, 1)[0].rstrip()
    else:
        prefix = (system_value or "").rstrip()
    return f"{prefix}\n\n{API_MARKER}".strip()


def build_qwen_modular_prompt_text(
    tool_names: List[str],
    conv_user: str,
    *,
    schema_name: str = "function-call",
) -> str:
    """对齐 A/B 的 ChatML：system/tools 在 schema，对话仍是单条 user。"""
    mod_names = unique_module_names(tool_names)
    tool_tags = "\n".join(f"        <{name}/>" for name in mod_names)
    user_xml = _format_user_xml(conv_user)
    return f"""
        <prompt schema='{schema_name}'>
{tool_tags}
{user_xml}
        </prompt>
        """


def _unique_module_attrs(xml: str) -> str:
    """formatter / recover parser 之后再去重一次 <module name>。"""
    seen: set[str] = set()

    def _repl(m: re.Match[str]) -> str:
        name = m.group(1)
        base = name
        n = 2
        while name in seen:
            name = f"{base}_{n}"
            n += 1
        seen.add(name)
        return f'<module name="{name}">'

    return re.sub(r'<module name="([^"]+)">', _repl, xml)


def conversation_key(sample: Dict[str, Any]) -> str:
    """`{qid}:{step}:{fn}` → qid；用于同对话内 staged 复用与按对话切分 GPU。"""
    sid = str(sample.get("sample_id") or sample.get("id") or "").strip()
    if ":" in sid:
        return sid.split(":", 1)[0]
    return sid or "anon"


def conversation_step(sample: Dict[str, Any]) -> int:
    sid = str(sample.get("sample_id") or "").strip()
    parts = sid.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


def schema_fingerprint(system_prefix: str, tools: List[Dict[str, Any]]) -> Tuple[str, Tuple[str, ...]]:
    names = tuple(sorted({str(t.get("name") or "") for t in tools if t.get("name")}))
    return (str(system_prefix or ""), names)


def schema_cache_name(system_prefix: str, tools: List[Dict[str, Any]]) -> str:
    """CacheEngine 可同时挂多份 schema；名字必须是合法 XML Name。"""
    prefix, names = schema_fingerprint(system_prefix, tools)
    raw = prefix + "\0" + "\0".join(names)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"fc_{digest}"


def reset_prompt_cache_session(cache_engine) -> None:
    """清空 GPU 上已拼接的 PromptCache.staged。

    仅在 schema 重建、对话切换、Phase2 独立基线开始时调用。
    同一对话连续 step 应保留 staged，让 `PromptCache.update()` 做前缀增量 copy。
    sidecar（全工具前缀 GPU 副本）跨对话保留，换 schema 时由 CacheEngine 丢掉。
    """
    pc = getattr(cache_engine, "prompt_cache", None)
    if pc is None:
        return
    pc.staged = []
    pc.length = 0


def _schema_lru(cache_engine) -> OrderedDict:
    lru = getattr(cache_engine, "_pipeline_schema_lru", None)
    if lru is None:
        lru = OrderedDict()
        cache_engine._pipeline_schema_lru = lru
    return lru


def touch_schema_lru(cache_engine, name: str) -> None:
    if cache_engine is None or not name:
        return
    lru = _schema_lru(cache_engine)
    if name in lru:
        lru.move_to_end(name)
    else:
        lru[name] = True


def drop_schema_l1(cache_engine, name: str) -> None:
    """丢掉一份 CPU L1。不要走 CacheEngine.remove_schema：它会 gc.collect + empty_cache。"""
    release = getattr(cache_engine, "release_schema_gpu", None)
    if callable(release):
        release(name)
    schemas = getattr(cache_engine, "schemas", None)
    if not schemas or name not in schemas:
        return
    sc = schemas.pop(name)
    lru = getattr(cache_engine, "_pipeline_schema_lru", None)
    if lru is not None:
        lru.pop(name, None)
    for tsc in (getattr(sc, "cache_l1", None) or {}).values():
        free = getattr(tsc, "free", None)
        if callable(free):
            free()
    del sc


def bind_toolbench_schema(
    cache_engine,
    lm,
    system_prefix: str,
    tools: List[Dict[str, Any]],
    *,
    schema_max_tokens: int,
    preproc=None,
    schema_batch_size: int = 1,
    max_cached_schemas: int = 8,
) -> bool:
    """按工具清单缓存 schema L1。已存在则直接复用，不再 remove_all / 重 prefill。

    返回 True 表示新构建，False 表示命中缓存。
    """
    from promptcache import CompactSpaces
    from promptcache.cache_engine import Schema

    name = schema_cache_name(system_prefix, tools)
    schemas = getattr(cache_engine, "schemas", None)
    if schemas is not None and name in schemas:
        touch_schema_lru(cache_engine, name)
        return False

    if schemas is not None and max_cached_schemas > 0:
        lru = _schema_lru(cache_engine)
        while len(schemas) >= max_cached_schemas:
            victim = next(iter(lru), None) or next(iter(schemas), None)
            if victim is None or victim == name:
                break
            drop_schema_l1(cache_engine, victim)
            n_evict = int(getattr(cache_engine, "_pipeline_schema_evicts", 0)) + 1
            cache_engine._pipeline_schema_evicts = n_evict
            if n_evict <= 3 or n_evict % 50 == 0:
                print(
                    f"[schema] evict {victim} pool={len(schemas)} n_evict={n_evict}",
                    flush=True,
                )

    print(
        f"[schema] bind {name} tools={len(tools)} max_tokens={int(schema_max_tokens)}",
        flush=True,
    )
    xml = build_toolbench_schema_xml(system_prefix, tools, schema_name=name)
    procs = list(preproc) if preproc else [CompactSpaces()]
    for p in procs:
        xml = p(xml)
    xml = _unique_module_attrs(xml)
    cache_engine.add_schema(
        Schema(xml, lm, max_tokens=int(schema_max_tokens)),
        max_tokens=int(schema_max_tokens),
        batch_size=max(1, int(schema_batch_size)),
    )
    touch_schema_lru(cache_engine, name)
    return True
