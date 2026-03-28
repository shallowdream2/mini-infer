# Roadmap

mini-infer 作为学习型推理引擎，当前实现了 21 个完整阶段（Phase 1–21）。
本文件记录已明确的后续扩展方向，以及与生产级框架之间的已知 gap。

## 当前已完成的主线能力

完整 21 阶段说明见 [docs/phases.md](phases.md)。核心成果速览：

| 层次 | 代表技术 | 关键数据 |
|------|----------|---------|
| Runtime | Continuous Batching、Paged KV Cache、Chunked Prefill | batch=8 达到 HF baseline 100% |
| 性能优化 | CUDA Graph、Flash Decoding、Prefix Caching | decode 延迟 −28.9%，SM 利用率 9%→103% |
| 算法 | Speculative Decoding | acceptance rate 55.85% |
| 分布式 | Tensor Parallelism、MoE EP（Grouped Execution） | EP grouped / dense = 2.500× |
| 前沿架构 | MLA（DeepSeek-V2/V3）、PD 解耦 | latent cache −56.25% |
| 量化 | W8A8 per-channel int8 + mixed fallback | 权重显存 −32.4% |

## 下一步技术扩展方向

这些方向尚未实现，按优先级排序：

### 量化扩展
- **FP8 推理**：利用 H100 FP8 Tensor Core，相比 W8A8 更高精度
- **Triton INT8 GEMM**：替代 `torch._int_mm`，消除小 M decode 路径的 fallback 开销
- **AWQ / GPTQ 加载**：复用已有量化框架的权重格式

### Attention 内核
- **PagedAttention v2**：更细粒度的 block 划分，减少碎片化
- **Token-level Prefix Cache**：当前是 block-level，更细粒度可进一步提升命中率
- **Multi-block Flash Decoding**：当前 split-K 实现仅验证 seq=4096，更长序列待测

### 分布式
- **TP + EP 混合并行**：TP 管 attention，EP 管 MoE expert，组合使用
- **跨机 PD 解耦**：当前 Phase 15 是同机双进程原型，跨机需 RDMA 或以太网 KV 传输
- **Pipeline Parallelism 闭环**：当前 PPEngine 只用于吞吐测量，未接入 continuous batching

### 服务化
- **Multi-LoRA serving**：在同一引擎下同时服务多个 LoRA adapter
- **Request priority API**：当前调度器支持 priority，但 HTTP API 未透传
- **SLO-aware 调度**：基于 TTFT / TBT SLO 动态调整调度策略
- **Prefix-aware 请求路由**：多副本下把相同前缀的请求路由到同一副本

### 工程
- **YAML 配置文件**：当前所有参数通过 `EngineConfig` 代码传入
- **Benchmark 结果自动归档**：运行后自动写入 `本地资料/实验记录/`
- **性能可视化**：吞吐演进图、ITL 分布图、EP 通信量对比图

## 与生产级框架的已知 Gap

| 维度 | mini-infer 当前 | vLLM / TRT-LLM 生产框架 |
|------|----------------|------------------------|
| 稳定性 | 实验原型，无长期运行测试 | SLO 保障，内存泄漏监控 |
| 模型覆盖 | 仅 Qwen2.5 / DeepSeek-V2（synthetic） | 数十种架构，自动适配 |
| 量化精度 | W8A8 greedy match 71.8% | 标定工具链，PTQ/QAT |
| 多 LoRA | 未实现 | 支持 |
| 调度精细度 | 基础 priority + preemption | 完整 SLO、KV 共享感知 |
| 部署 | 单机原型 | K8s、多机 RDMA |

这些 gap 是**有意识的设计选择**：mini-infer 的目标是把核心机制讲清楚，而不是复现完整的生产系统。

## 贡献指南

如有兴趣扩展某个方向，建议：
1. 在 [Issues](https://github.com/psmarter/mini-infer/issues) 提前说明动机和方案
2. 每个新功能对应一个 benchmark 脚本和最小测试
3. 实现前先确认与已有 Phase 数值的兼容性（greedy token match）
