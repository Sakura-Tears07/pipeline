#!/usr/bin/env bash
# 最终 8 组：A pruned/full、B adaptive/full、C ori/dev × adaptive/full
# 全部 modular KV、CPU 预取。每组独立进程，组间清缓存。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="${ROOT}/_eval_logs/run_final.log"
mkdir -p "${ROOT}/_eval_logs"
exec > >(tee -a "${LOG}") 2>&1

LIMIT="${LIMIT:-0}"
export LIMIT NUM_GPUS="${NUM_GPUS:-4}" GPU_IDS="${GPU_IDS:-0,1,2,3}" UNC_TAU="${UNC_TAU:-1.6}"
RUN_A="${RUN_A:-1}"
RUN_B="${RUN_B:-1}"
RUN_C="${RUN_C:-1}"

echo "========== $(date -Iseconds) FINAL 8-group START LIMIT=${LIMIT} A=${RUN_A} B=${RUN_B} C=${RUN_C} =========="

if [[ "${RUN_A}" == "1" ]]; then
  echo "========== $(date -Iseconds) A MODE=pruned =========="
  MODE=pruned bash "${ROOT}/A/start.sh"
  echo "========== $(date -Iseconds) A MODE=full =========="
  MODE=full bash "${ROOT}/A/start.sh"
fi

if [[ "${RUN_B}" == "1" ]]; then
  echo "========== $(date -Iseconds) B MODE=adaptive =========="
  MODE=adaptive bash "${ROOT}/B/start.sh"
  echo "========== $(date -Iseconds) B MODE=full =========="
  MODE=full bash "${ROOT}/B/start.sh"
fi

if [[ "${RUN_C}" == "1" ]]; then
  for fw in ori dev; do
    for mode in adaptive full; do
      echo "========== $(date -Iseconds) C FRAMEWORK=${fw} MODE=${mode} =========="
      FRAMEWORK="${fw}" MODE="${mode}" bash "${ROOT}/C/start.sh"
    done
  done
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python3 -c "from common.final_compare import write_final_compare; write_final_compare()"

echo "========== $(date -Iseconds) FINAL DONE =========="
echo "compare: ${ROOT}/output/final_compare.md"
echo "A report: ${ROOT}/A/output/report.md"
echo "B report: ${ROOT}/B/output/report.md"
echo "C report: ${ROOT}/C/output/report.md"
