#!/usr/bin/env bash
# Pipeline C：CodeLlama-34B modular KV，FRAMEWORK=ori|dev × MODE=adaptive|full
# 输出：C/output/${FRAMEWORK}/${MODE}/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "${ROOT}/.." && pwd)"
cd "${ROOT}"

source /home/lihaoran/miniconda3/etc/profile.d/conda.sh

LIMIT="${LIMIT:-0}"
FRAMEWORK="${FRAMEWORK:-dev}"
MODE="${MODE:-adaptive}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
UNC_TAU="${UNC_TAU:-1.4}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-1}"
CLEAR_CACHE="${CLEAR_CACHE:-1}"
PREDICTOR_DEVICE="${PREDICTOR_DEVICE:-cpu}"
PREDICTOR_BACKEND="${PREDICTOR_BACKEND:-qwen3_emb}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-4}"
LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-64}"
LLM_MAX_CONTEXT_TOKENS="${LLM_MAX_CONTEXT_TOKENS:-8192}"
STICKY_FULL="${STICKY_FULL:-1}"
RUN_TAG="${RUN_TAG:-}"

if [[ "${FRAMEWORK}" != "ori" && "${FRAMEWORK}" != "dev" ]]; then
  echo "FRAMEWORK must be ori or dev (got ${FRAMEWORK})" >&2
  exit 1
fi
if [[ "${MODE}" != "adaptive" && "${MODE}" != "full" ]]; then
  echo "Pipeline C MODE must be adaptive or full (got ${MODE})" >&2
  exit 1
fi

if [[ -n "${RUN_TAG}" ]]; then
  OUT_DIR="${ROOT}/output/${FRAMEWORK}/${MODE}/${RUN_TAG}"
else
  OUT_DIR="${ROOT}/output/${FRAMEWORK}/${MODE}"
fi
mkdir -p "${OUT_DIR}"

export FRAMEWORK
export MODE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES || true

echo "=== Pipeline C ${PREDICTOR_BACKEND} framework=${FRAMEWORK} MODE=${MODE} ==="
echo "LIMIT=${LIMIT} LLM_BATCH=${LLM_BATCH_SIZE} UNC_TAU=${UNC_TAU} TEMP=${LLM_TEMPERATURE:-0} pred=${PREDICTOR_DEVICE} prefetch=${PREFETCH_DEPTH} max_new=${LLM_MAX_NEW_TOKENS} max_ctx=${LLM_MAX_CONTEXT_TOKENS} sticky=${STICKY_FULL} out=${OUT_DIR}"

if [[ "${CLEAR_CACHE}" == "1" ]]; then
  bash "${PIPELINE_ROOT}/common/clear_runtime_cache.sh" "C_${FRAMEWORK}_${MODE}_before"
fi

conda activate prompt_cache
_FILTERED=""
IFS=':' read -ra _PP_ARR <<< "${PYTHONPATH:-}"
for _p in "${_PP_ARR[@]}"; do
  _base="$(basename "${_p%/}")"
  if [[ "${_base}" == "prompt-cache" || "${_base}" == "prompt-cache-ori" || "${_base}" == "prompt-cache-dev" ]]; then
    continue
  fi
  if [[ -n "${_p}" ]]; then
    _FILTERED="${_FILTERED:+${_FILTERED}:}${_p}"
  fi
done
export PYTHONPATH="/home/lihaoran/workspace/find_model:/home/lihaoran/workspace/prompt-cache-${FRAMEWORK}:${PIPELINE_ROOT}:${ROOT}${_FILTERED:+:${_FILTERED}}"

python run_online.py \
  --dataset "${DATASET_PATH:-${PIPELINE_ROOT}/data/find_model_sft811_test.json}" \
  --output "${OUT_DIR}/runtime_outputs.json" \
  --limit "${LIMIT}" \
  --mode "${MODE}" \
  --num-gpus "${NUM_GPUS}" \
  --gpu-ids "${GPU_IDS}" \
  --model "${LLM_MODEL:-/data/model/CodeLlama-34b-Instruct-hf}" \
  --predictor-backend "${PREDICTOR_BACKEND}" \
  --predictor-device "${PREDICTOR_DEVICE}" \
  --unc-metric margin_top12 \
  --unc-tau "${UNC_TAU}" \
  --prefetch-depth "${PREFETCH_DEPTH}" \
  --llm-batch-size "${LLM_BATCH_SIZE}" \
  --use-modular-cache 1 \
  --llm-model-class Llama2 \
  --framework "${FRAMEWORK}" \
  --prompt-cache-root "/home/lihaoran/workspace/prompt-cache-${FRAMEWORK}" \
  --load-in-4bit 1 \
  --load-in-8bit 0 \
  --sticky-full "${STICKY_FULL}" \
  --temperature "${LLM_TEMPERATURE:-0}" \
  ${LLM_EXTRA_ARGS:-} \
  --max-context-tokens "${LLM_MAX_CONTEXT_TOKENS}" \
  --max-new-tokens "${LLM_MAX_NEW_TOKENS}"

if [[ "${CLEAR_CACHE}" == "1" ]]; then
  bash "${PIPELINE_ROOT}/common/clear_runtime_cache.sh" "C_${FRAMEWORK}_${MODE}_after"
fi

echo "DONE. ${OUT_DIR}/"
