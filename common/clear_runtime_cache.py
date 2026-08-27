#!/usr/bin/env python3
"""清 runtime 相关缓存：CUDA 显存 + 残留 shard 目录。"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
from pathlib import Path


def clear_cuda() -> None:
    try:
        import torch
    except ImportError:
        print("[clear_cache] torch 不可用，跳过 CUDA", flush=True)
        return

    gc.collect()
    if not torch.cuda.is_available():
        print("[clear_cache] 无 CUDA 设备", flush=True)
        return

    n = torch.cuda.device_count()
    for i in range(n):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    print(f"[clear_cache] CUDA empty_cache 完成 (devices={n})", flush=True)


def clear_shards(pipeline_root: Path) -> None:
    removed = 0
    for sub in ("A", "B", "C"):
        out = pipeline_root / sub / "output"
        if not out.is_dir():
            continue
        for p in out.rglob(".runtime_shards_*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                print(f"[clear_cache] 删除 shard: {p}", flush=True)
                removed += 1
    if removed == 0:
        print("[clear_cache] 无残留 .runtime_shards_*", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--skip-shards", action="store_true")
    args = parser.parse_args()

    root = Path(args.pipeline_root)
    if not args.skip_shards:
        clear_shards(root)
    if not args.skip_cuda:
        clear_cuda()
    gc.collect()
    print("[clear_cache] done", flush=True)


if __name__ == "__main__":
    main()
