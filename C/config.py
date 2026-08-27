"""Pipeline C：CodeLlama-34B + modular KV。ori/dev × adaptive/full。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUNTIME_MODE = "adaptive"
UNCERTAINTY_METRIC = "margin_top12"
UNCERTAINTY_TAU = 1.4
STICKY_FULL = True
PREDICTOR_BACKEND = "qwen3_emb"
PREDICTOR_DEVICE = "cpu"
PREFETCH_DEPTH = 4

LLM_MODEL = "/data/model/CodeLlama-34b-Instruct-hf"
LLM_MODEL_CLASS = "Llama2"
USE_MODULAR_CACHE = True
MODULAR_SCHEMA_STYLE = "json_only"
LLM_BATCH_SIZE = 1
LLM_LOAD_IN_4BIT = True
LLM_LOAD_IN_8BIT = False
LLM_MAX_NEW_TOKENS = 64
LLM_MAX_CONTEXT_TOKENS = 4096
# greedy：temperature>0 会走 multinomial，logits 一旦 inf/nan 会毒化整卡
LLM_TEMPERATURE = 0.0
MAX_CACHED_SCHEMAS = 8

_FRAMEWORK = os.environ.get("FRAMEWORK", "dev").strip().lower()
_CACHE_ROOTS = {
    "ori": Path("/home/lihaoran/workspace/prompt-cache-ori"),
    "dev": Path("/home/lihaoran/workspace/prompt-cache-dev"),
}
PROMPT_CACHE_FRAMEWORK = _FRAMEWORK if _FRAMEWORK in _CACHE_ROOTS else "dev"
PROMPT_CACHE_ROOT = _CACHE_ROOTS[PROMPT_CACHE_FRAMEWORK]

DATASET_PATH = "/home/lihaoran/workspace/pipeline/data/find_model_sft811_test.json"
NUM_GPUS = 4
GPU_IDS = "0,1,2,3"
LIMIT = 0

_MODE = os.environ.get("MODE", RUNTIME_MODE).strip().lower() or RUNTIME_MODE
_OUT_SUB = OUTPUT_DIR / PROMPT_CACHE_FRAMEWORK / _MODE
_OUT_SUB.mkdir(parents=True, exist_ok=True)
RUNTIME_OUTPUTS_JSON = _OUT_SUB / "runtime_outputs.json"
REPORT_MD = _OUT_SUB / "report.md"
