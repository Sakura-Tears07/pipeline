"""Pipeline A：Qwen3-32B + modular KV。独立跑 pruned / full。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTOR_BACKEND = "qwen3_emb"

LLM_MODEL = "/data/model/Qwen3-32B"
PROMPT_CACHE_ROOT = Path("/home/lihaoran/workspace/prompt-cache-dev")

RUNTIME_MODE = "pruned"
PREDICTOR_DEVICE = "cpu"
PREFETCH_DEPTH = 4

LLM_LOAD_IN_4BIT = True
LLM_LOAD_IN_8BIT = False
LLM_MAX_NEW_TOKENS = 64
LLM_MAX_CONTEXT_TOKENS = 4096
LLM_TEMPERATURE = 0.1
LLM_BATCH_SIZE = 1
LLM_MODEL_CLASS = "Qwen"
USE_MODULAR_CACHE = True
MODULAR_SCHEMA_STYLE = "qwen_tools"
MAX_CACHED_SCHEMAS = 8

DATASET_PATH = "/home/lihaoran/workspace/pipeline/data/find_model_sft811_test.json"
NUM_GPUS = 4
GPU_IDS = "0,1,2,3"
LIMIT = 0

RUNTIME_OUTPUTS_JSON = OUTPUT_DIR / "pruned" / "runtime_outputs.json"
REPORT_MD = OUTPUT_DIR / "pruned" / "report.md"
