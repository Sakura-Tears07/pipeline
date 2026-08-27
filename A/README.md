# Pipeline A

Qwen3-32B + modular KV。独立两组：**pruned** vs **full**。

```
MODE=pruned: Embedding → 全体 pruned prompt → Qwen3-32B
MODE=full:   不加载 Embedding，全体 full prompt
```

```bash
MODE=pruned LIMIT=0 bash start.sh
MODE=full LIMIT=0 bash start.sh
```

输出：`output/pruned/`、`output/full/`。
