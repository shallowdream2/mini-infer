# Benchmark Metrics Spec

这个文件用于定义 mini-infer 项目中 benchmark 的核心指标口径，避免前后报告混乱。

## 指标定义

- `throughput`：单位时间内完成生成的 token 数，默认单位为 tokens/s
- `TTFT`：time to first token，从请求进入系统到首个输出 token 返回的耗时
- `TPOT`：time per output token，首 token 之后每个输出 token 的平均耗时
- `peak memory`：单次实验期间 GPU 显存峰值

## 报告要求

- 必须说明 workload 和环境
- 必须说明统计窗口和样本数量
- 必须区分真实测量结果与估算值
- 对比不同实现时，保持模型、输入规模和采样参数一致