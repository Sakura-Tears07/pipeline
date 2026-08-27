#!/usr/bin/env bash
# Pipeline B：modular KV，独立进程 MODE=adaptive | full
# 输出：B/output/${MODE}/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${ROOT}/.." && pwd)"
cd "${ROOT}"

source /home/lihaoran/miniconda3/etc/profile.d/conda.sh

LIMIT="${LIMIT:-0}"
MODE="${MODE:-adaptive}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
UNC_TAU="${UNC_TAU:-1.4}"
UNC_METRIC="${UNC_METRIC:-margin_top12}"
PREDICTOR_DEVICE="${PREDICTOR_DEVICE:-cpu}"
PREDICTOR_BACKEND="${PREDICTOR_BACKEND:-qwen3_emb}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-4}"
LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-64}"
CLEAR_CACHE="${CLEAR_CACHE:-1}"
STICKY_FULL="${STICKY_FULL:-1}"
RUN_TAG="${RUN_TAG:-}"
PROMPT_CACHE_ROOT="${PROMPT_CACHE_ROOT:-/home/lihaoran/workspace/prompt-cache-dev}"
LLM_MODEL_CLASS="${LLM_MODEL_CLASS:-Qwen}"
MODULAR_SCHEMA_STYLE="${MODULAR_SCHEMA_STYLE:-qwen_tools}"

if [[ "${MODE}" != "adaptive" && "${MODE}" != "full" ]]; then
  echo "Pipeline B MODE must be adaptive or full (got ${MODE})" >&2
  exit 1
fi

if [[ -n "${RUN_TAG}" ]]; then
  OUT_SUB="${MODE}/${RUN_TAG}"
else
  OUT_SUB="${MODE}"
fi
OUTPUT="${ROOT}/output/${OUT_SUB}/runtime_outputs.json"
mkdir -p "${ROOT}/output/${OUT_SUB}"

unset CUDA_VISIBLE_DEVICES || true

echo "=== Pipeline B modular MODE=${MODE} ==="
echo "LIMIT=${LIMIT} UNC_TAU=${UNC_TAU} backend=${PREDICTOR_BACKEND} pred=${PREDICTOR_DEVICE} prefetch=${PREFETCH_DEPTH} max_new=${LLM_MAX_NEW_TOKENS} sticky=${STICKY_FULL} cache=${PROMPT_CACHE_ROOT} out=${OUTPUT}"

if [[ "${CLEAR_CACHE}" == "1" ]]; then
  bash "${PIPELINE_ROOT}/common/clear_runtime_cache.sh" "B_${MODE}_before"
fi

conda activate prompt_cache
export PYTHONPATH="/home/lihaoran/workspace/find_model:${PROMPT_CACHE_ROOT}:${PIPELINE_ROOT}:${ROOT}:${PYTHONPATH:-}"

python run_online.py \
  --dataset "${DATASET_PATH:-${PIPELINE_ROOT}/data/find_model_sft811_test.json}" \
  --output "${OUTPUT}" \
  --limit "${LIMIT}" \
  --mode "${MODE}" \
  --num-gpus "${NUM_GPUS}" \
  --gpu-ids "${GPU_IDS}" \
  --unc-metric "${UNC_METRIC}" \
  --unc-tau "${UNC_TAU}" \
  --predictor-backend "${PREDICTOR_BACKEND}" \
  --predictor-device "${PREDICTOR_DEVICE}" \
  --prefetch-depth "${PREFETCH_DEPTH}" \
  --max-new-tokens "${LLM_MAX_NEW_TOKENS}" \
  --prompt-cache-root "${PROMPT_CACHE_ROOT}" \
  --llm-model-class "${LLM_MODEL_CLASS}" \
  --use-modular-cache 1 \
  --modular-schema-style "${MODULAR_SCHEMA_STYLE}" \
  --sticky-full "${STICKY_FULL}" \
  ${LLM_EXTRA_ARGS:-}

if [[ "${CLEAR_CACHE}" == "1" ]]; then
  bash "${PIPELINE_ROOT}/common/clear_runtime_cache.sh" "B_${MODE}_after"
fi

echo "DONE. ${OUTPUT}"
