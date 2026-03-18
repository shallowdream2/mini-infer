---
name: infer-benchmark
description: 为 mini-infer 设计或执行 benchmark，并输出统一口径的结果说明。
argument-hint: [benchmark 任务]
disable-model-invocation: true
---

# infer-benchmark

这个文件用于 benchmark 设计、执行和结果整理。

把附加文本视为 benchmark 任务，并遵守这些要求：

- 先说明对象：HF baseline 或 mini-infer 当前实现
- 说明 workload：模型、batch size、prompt 长度、output 长度、并发数
- 至少关注 `throughput`、`TTFT`、`TPOT`、`peak memory`
- 输出环境、命令、数据口径、结果、结论和局限性
- 没有真实数据时，不输出性能结论

需要指标定义时，优先参考同目录下的 `METRICS_SPEC.md`。
