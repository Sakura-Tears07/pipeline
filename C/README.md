# Pipeline C

CodeLlama-34B + modular KV。四组：**ori/dev × adaptive/full**。

- 框架：`prompt-cache-ori` vs `prompt-cache-dev`
- schema：`json_only`；schema L1 常驻，同对话 staged 前缀复用
- CodeLlama **不接** Qwen session append

```bash
FRAMEWORK=ori MODE=adaptive bash start.sh
FRAMEWORK=ori MODE=full bash start.sh
FRAMEWORK=dev MODE=adaptive bash start.sh
FRAMEWORK=dev MODE=full bash start.sh
```

输出：`output/{ori,dev}/{adaptive,full}/`。八组总表见 `../output/final_compare.md`。
