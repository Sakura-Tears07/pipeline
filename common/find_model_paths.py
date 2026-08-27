"""find_model 路径：A/B/C 统一用微调后的 Qwen3-Embedding-0.6B。"""

from __future__ import annotations

from pathlib import Path

# H1 λ=0.10 best（P2 → hardest-neg）部署副本；训练中间产物已清理
QWEN3_EMB_CKPT = Path("/data/model/Qwen3-Embedding-0.6B-finetuned")

MAX_CANDIDATES = 16
