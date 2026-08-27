# Pipeline（最终 8 组）

全部 **modular KV** + Qwen3-Embedding-0.6B-finetuned @ CPU 预取（`PREFETCH_DEPTH=4`）。每组独立进程，组间清缓存。

| # | Pipeline | 组 | LLM | 输出 |
|---|----------|----|-----|------|
| 1 | A | modular **pruned** | Qwen3-32B 4bit | `A/output/pruned/` |
| 2 | A | modular **full** | Qwen3-32B 4bit | `A/output/full/` |
| 3 | B | modular **adaptive** τ=1.6 | Qwen3-32B 4bit | `B/output/adaptive/` |
| 4 | B | modular **full** | Qwen3-32B 4bit | `B/output/full/` |
| 5 | C | **ori** adaptive | CodeLlama-34B 4bit | `C/output/ori/adaptive/` |
| 6 | C | **ori** full | CodeLlama-34B 4bit | `C/output/ori/full/` |
| 7 | C | **dev** adaptive | CodeLlama-34B 4bit | `C/output/dev/adaptive/` |
| 8 | C | **dev** full | CodeLlama-34B 4bit | `C/output/dev/full/` |

A/B 比 **策略**（始终 pruned vs 不确定门控）。C 比 **框架**（prompt-cache-ori vs prompt-cache-dev）。策略臂和 full 基线不在同一进程里跑，避免 modular schema L1 被复用。

主指标：**TTFT / prefill / cache overhead**。参数见 `EXPERIMENT_PARAMS.md`。

```bash
# 一次跑齐 8 组
LIMIT=0 bash run_final.sh

# 只跑一部分
RUN_C=0 LIMIT=0 bash run_final.sh          # 只 A+B
RUN_A=0 RUN_B=0 LIMIT=0 bash run_final.sh  # 只 C 四组

# 单组
MODE=pruned bash A/start.sh
MODE=full bash A/start.sh
MODE=adaptive bash B/start.sh
MODE=full bash B/start.sh
FRAMEWORK=ori MODE=adaptive bash C/start.sh
FRAMEWORK=ori MODE=full bash C/start.sh
FRAMEWORK=dev MODE=adaptive bash C/start.sh
FRAMEWORK=dev MODE=full bash C/start.sh
```

对比：`output/final_compare.md`。
