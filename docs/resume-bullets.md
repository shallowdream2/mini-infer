# 简历 Bullets：大模型推理系统学习与性能分析

> 按使用场景分组，根据投递岗位（推理系统 / MLSys / 大模型平台）挑选 4-6 条。
> 加粗数字是可以对外引用的实测数据。

---

## A 组：系统实现能力（核心，必选 2-3 条）

- 从零实现大模型推理引擎 mini-infer，覆盖 OpenAI-compatible HTTP serving、连续批处理（Continuous Batching）、PagedAttention KV Cache 管理及请求调度，batch=8 场景下吞吐达到 HuggingFace baseline 的 **100%（406 tok/s）**

- 实现 PagedAttention 物理 Block Pool + BlockTable 映射机制，将 KV cache 管理从 per-request 连续分配改为 block-level 统一调度；在此基础上实现 block-level SHA-256 链式 hash 前缀缓存（Prefix Caching），相同系统 prompt 场景 TTFT **降低 22%**

- 实现 Chunked Prefill 调度器（WAITING / PREFILLING / RUNNING 三状态机），将长 prompt（4096 tokens）拆成 chunk_size=256 的多步 prefill，消除 decode batch 阻塞；实测 Inter-Token Latency spike **降低 57–67%**

- 实现 AsyncEngine 异步推理引擎：后台线程持续 step loop，多并发 HTTP 请求通过 asyncio.Queue 合并进同一 decode batch；并发 1→8 吞吐提升 **3.9×（55.7 → 219.1 tok/s）**

---

## B 组：性能优化（适合强调内核/底层的岗位，选 1-2 条）

- 实现 CUDA Graph 静态捕获：对每种 batch size 预捕获 decode_batch 计算图，消除 Python dispatch overhead，bs=1 decode 延迟 **降低 28.9%**

- 实现 Triton split-K Flash Decoding kernel，通过跨 SM 并行归约长序列 Attention，seq=4096 时延迟 **较标准 Triton kernel 提升 3.31×**，SM 利用率从 9% 提升至 103%

- 实现 W8A8 per-channel int8 量化（QuantLinear + mixed fallback），权重显存 **降低 32.4%（3.4GB → 2.3GB）**，greedy token match 71.8%

---

## C 组：架构理解（适合偏系统架构 / 平台方向，选 1-2 条）

- 实现 Speculative Decoding（0.5B draft + 7B target），modified rejection sampling，K=4 时 acceptance rate **55.85%**，验证了 draft model 质量对实际加速比的影响

- 实现 DeepSeek-V2 MLA（Multi-head Latent Attention）三种方案（Naive / LatentCache / Absorbed），Latent cache 方案 KV cache 体积较 GQA **减少 56.25%**，验证了低秩 KV 压缩的显存收益

- 实现 Prefill/Decode 解耦原型（PD Disaggregation）：同机双进程，KV 序列化 socket 传输，量化 TTFT 三段组成（prefill 12.3ms / KV transfer 14.7ms / first decode 519ms），理解 P/D 分离的延迟 trade-off

---

## D 组：分布式（适合强调多卡/分布式的岗位，选 1 条）

- 实现 Tensor Parallelism（NCCL all-reduce，Megatron-LM 风格列/行并行），TP=2 greedy 输出与单卡完全一致；实现 MoE Expert Parallelism grouped execution，2-GPU 吞吐 **2.5× vs 单卡 dense**（padded 1.95× → packed 2.32× → grouped 2.50× 三阶段演进）

---

## 推荐选取方式

| 岗位方向 | 推荐条数 | 优先选择 |
|---------|---------|---------|
| 推理系统 / Serving 框架 | 4-5 条 | A1、A2、A3、A4、B1 |
| 大模型平台 / MLSys | 4 条 | A1、A3、C1、C2 |
| 底层内核 / CUDA 优化 | 3-4 条 | A1、B1、B2、B3 |
| 分布式系统 | 3-4 条 | A1、A4、D1 |

---

## 写法说明

- 数字来自 `docs/benchmarks.md` 的实测数据，可直接引用
- "从零实现"强调理解深度，而非"复现 vLLM"
- 每条遵循 **动词 + 做了什么 + 数据支撑** 结构
- 面试时每条背后都有代码位置（文件+函数名）可以追问
