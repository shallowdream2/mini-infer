---
paths:
  - "benchmarks/**/*.py"
---

# Benchmark 规则

这个文件约束 benchmark 设计、执行和结果表达的基本口径。

- benchmark 必须说明环境、模型、batch size、prompt 长度、output 长度和并发数
- 至少关注 `throughput`、`TTFT`、`TPOT`、`peak memory`
- 如果某个指标在当前脚本中不可得、只近似可得或口径与其他脚本不同，必须明确标注 `N/A`、`近似` 或差异说明，不能伪装成精确值
- benchmark 结果默认与 `HuggingFace Transformers` baseline 对照
- 没有真实 Ubuntu + CUDA 数据时，不得输出性能优于 baseline 的结论
- benchmark 结论应同步整理到 `本地资料/实验记录/`
