# Runtime A / B / C 实验参数

## 最终 8 组（全部 modular）

| 项 | A / B | C |
|----|-------|---|
| 数据集 | `data/find_model_sft811_test.json` | 同左 |
| Predictor | Qwen3-Embedding-0.6B-finetuned @ **cpu** | 同左 |
| prefetch | **4** | **4** |
| max_new_tokens | **64** | **64** |
| max_context | 8192 | 8192 |
| KV | modular，`qwen_tools` | modular，`json_only` |
| cache 树 | `prompt-cache-dev` | `prompt-cache-ori` / `prompt-cache-dev` |

`MODE=full` 不加载 predictor。Adaptive 默认 **sticky full**。

| # | 组 | 启动 |
|---|----|------|
| 1–2 | A pruned / full | `MODE=pruned\|full bash A/start.sh` |
| 3–4 | B adaptive / full | `MODE=adaptive\|full bash B/start.sh` |
| 5–8 | C ori/dev × adaptive/full | `FRAMEWORK=ori\|dev MODE=adaptive\|full bash C/start.sh` |

一次跑齐：`LIMIT=0 bash run_final.sh`。主看 **TTFT / prefill**，不要用 decode / e2e 当下结论。
