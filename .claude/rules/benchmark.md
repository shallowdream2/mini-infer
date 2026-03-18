---
paths:
  - "benchmarks/**/*.py"
---

# Benchmark 规则

这个文件约束 benchmark 设计、执行和结果表达的基本口径。

- benchmark 必须说明环境、模型、batch size、prompt 长度、output 长度和并发数
- 至少关注 `throughput`、`TTFT`、`TPOT`、`peak memory`
- benchmark 结果默认与 `HuggingFace Transformers` baseline 对照
- 没有真实 Ubuntu + CUDA 数据时，不得输出性能优于 baseline 的结论
- benchmark 结论应同步整理到 `本地资料/实验记录/`
