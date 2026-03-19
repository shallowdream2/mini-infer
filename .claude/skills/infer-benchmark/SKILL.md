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

## 性能优化阶段的附加要求

如果本阶段包含性能优化（如 gather 向量化、kernel 替换、调度改进等），benchmark 必须额外包含：

1. **优化前后的时间分布对比**（不只是 throughput 数字）：用 `benchmarks/profile_decode.py` 或等效工具，确认优化命中了目标瓶颈
2. **量化优化效果**：被优化的操作占总 CUDA 时间的比例（优化前 vs 优化后），防止优化了非瓶颈导致收益为零
3. **说明剩余差距的根因**：不仅报告"还差 X%"，还要说明差距主要来自哪里

**理由**：Phase 3 向量化 gather 后达到 88.4% HF，但"剩余 12% 在哪里"直到 Phase 5 才量化。每个性能优化阶段应当即时回答这个问题，而不是留给后续 phase 补测。
