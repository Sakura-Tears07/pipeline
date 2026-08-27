# Pipeline B

Qwen3-32B + modular KV。独立两组：**adaptive** vs **full**。

```
MODE=adaptive: Embedding → margin_top12（τ=1.6）
  ├─ margin ≥ τ → pruned
  └─ margin < τ → full
  对话内 sticky full
MODE=full: 不加载 Embedding，全体 full prompt
```

```bash
MODE=adaptive LIMIT=0 bash start.sh
MODE=full LIMIT=0 bash start.sh
```

输出：`output/adaptive/`、`output/full/`。
