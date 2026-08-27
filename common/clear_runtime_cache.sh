#!/usr/bin/env bash
# 清 runtime 缓存（CUDA + 残留 shard）；需在 prompt_cache 环境下调用
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${1:-}"

if [[ -n "${LABEL}" ]]; then
  echo "========== $(date -Iseconds) clear_cache (${LABEL}) =========="
else
  echo "========== $(date -Iseconds) clear_cache =========="
fi

source /home/lihaoran/miniconda3/etc/profile.d/conda.sh
conda activate prompt_cache
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python "${ROOT}/common/clear_runtime_cache.py" --pipeline-root "${ROOT}"

# 给驱动一点时间回收显存
sleep 2

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
fi
