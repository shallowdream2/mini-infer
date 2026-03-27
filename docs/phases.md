# Phases — 详细阶段说明

mini-infer 按照**现代 LLM 推理系统的演化路径**逐步构建，共 21 个阶段。
每个阶段有独立的 benchmark 脚本和验收指标，阶段结束后项目是完整可运行的状态。

---

## Runtime 基础（Phase 1–7）

### Phase 1 — 单卡最小推理链路
**目标**：建立最小可运行的推理链路，理解 HF Transformers 的 decode 流程。

- 加载 Qwen2.5-7B-Instruct，串行 decode
- 建立 `LLMEngine` 和 `Request` 基础数据结构
- 与 HF baseline 对比：吞吐 56 tok/s（HF 的 13.8%）

### Phase 2 — Paged KV Cache + Continuous Batching
**目标**：实现真正的 Paged KV Cache 和 Continuous Batching。

- `BlockTable` + `FreeBlockPool`：block 粒度 KV 管理
- `Scheduler`：waiting / running 队列，动态准入
- Prefill / Decode 分离：单步 prefill + batch decode
- 吞吐：201 tok/s（HF 的 49.5%）

### Phase 3 — 向量化 KV Gather + DynamicCache
**目标**：消除逐请求 KV gather 的 Python loop 开销。

- `gather_batch_kv`：advanced indexing，向量化多请求 KV 聚合
- DynamicCache：HF 标准 cache 格式，direct HF model forward
- 吞吐：361 tok/s（HF 的 88.4%）

### Phase 4 — 双卡扩展
**目标**：验证 Replica 数据并行和 HF Pipeline Parallel 两种多卡路径。

- `ReplicaEngine`：双卡数据并行，请求 round-robin 分发
- `PPEngine`：HF device_map 自动 PP，吞吐测量用

### Phase 5 — Profiling + 项目收尾
**目标**：用 PyTorch Profiler 定位性能瓶颈，完善文档。

- `profile_decode.py`：decode_batch 内部 profiling
- 标记 Python dispatch overhead 和 KV transfer 是下一阶段的主要瓶颈

### Phase 6 — True PagedAttention
**目标**：接入 `flash_attn_with_kvcache` 的 `block_table` 路径，完成真实的 Paged Attention。

- `attention.py`：`PagedDecodeContext` + `patch_model_for_paged_decode`
- 将 HF attention 层替换为 block_table 路径
- **吞吐：406 tok/s（HF 的 100%）** ← 主线闭环

### Phase 6.5 — Triton Decode Attention Kernel
**目标**：从零实现 Triton decode attention kernel，理解 GPU kernel 设计。

- online softmax（Dao 2022）+ GQA 扩展
- 与 flash_attn 基准对比，分析 SM 利用率和 roofline
- `triton_attn.py`：独立 kernel，不接入主推理链路

### Phase 7 — Preemption + Priority Scheduling
**目标**：处理 KV cache 不足时的抢占，支持请求优先级。

- Preemption：高优先级请求抢占低优先级，KV blocks swap to CPU
- `swapped` 队列：swap_in / swap_out 管理
- Priority：`SamplingParams.priority`

---

## 性能优化（Phase 8–12.5）

### Phase 8 — OpenAI Chat Completions 兼容 HTTP API
**目标**：提供真实可用的 HTTP serving 接口。

- `AsyncEngine`：后台线程 step loop + `asyncio.Queue`
- `FastAPI` server：`GET /v1/models` + `POST /v1/chat/completions`
- SSE streaming + non-streaming
- `openai_schema.py`：Pydantic 请求/响应模型

### Phase 9 — Chunked Prefill
**目标**：防止长 prefill 饿死 decode 请求，降低 ITL spike。

- 调度器改造：`prefilling` 队列 + chunk 状态机
- 每步最多处理 `chunk_prefill_size` 个 prefill tokens
- ITL spike：−57%（chunk=256）/ −67%（chunk=128）

### Phase 10 — Prefix Caching
**目标**：对共享前缀的请求复用 KV blocks，降低 TTFT。

- Block-level SHA-256 链式 hash
- LRU eviction + ref_count 管理
- 1-block 共享前缀：TTFT −22%

### Phase 11 — Speculative Decoding
**目标**：用小模型 draft 加速大模型 decode。

- `SpecEngine`：draft（0.5B） + target（7B）双模型
- Modified rejection sampling
- acceptance_rate：55.85%（K=4）

### Phase 12 — CUDA Graph
**目标**：消除 decode_batch 的 Python dispatch 开销。

- decode_batch 静态捕获（per batch size 一张图）
- graph pool + replay
- 1.5B bs=1 decode 延迟 −28.9%

### Phase 12.5 — Flash Decoding / Split-K Attention
**目标**：针对长序列 decode 的 SM 利用率问题，实现 split-K attention。

- Triton split-K kernel：`triton_flash_decode.py`
- seq=4096：3.31× vs 标准 Triton kernel，SM 利用率 9%→103%
- 独立实验性 kernel，不接入主推理链路

---

## 扩展能力（Phase 13–21）

### Phase 13 — Tensor Parallelism
**目标**：实现真 TP，而不是 HF device_map PP。

- NCCL all-reduce，Megatron-LM 风格权重切分
- Column/Row Parallel Linear + all-reduce hook
- TP=2，greedy 输出与单卡完全一致

### Phase 14 — MLA（Multi-head Latent Attention）
**目标**：理解并验证 DeepSeek-V2/V3 的 MLA 架构。

- 三种实现：Naive / LatentCache / Absorbed
- Latent cache：KV cache 体积 −56.25% vs GQA
- 矩阵吸收优化（将 W_UK/W_UV 预先吸收到投影矩阵）

### Phase 15 — Prefill/Decode 解耦（PD Disaggregation）
**目标**：将 Prefill 和 Decode 拆分到不同进程/节点。

- 同机双进程原型
- KV 序列化传输（socket）
- TTFT 三段分解：prefill 12.3ms / transfer ≈14.7ms / decode 519ms

### Phase 16 — W8A8 量化
**目标**：第一版权重 + 激活量化原型。

- `QuantLinear`：W8A8 per-channel int8
- attention 层跳过（mixed fallback）
- 权重显存 −32.4%（3392→2292 MB），greedy match 71.8%

### Phase 17 — MoE + Expert Parallelism
**目标**：实现 synthetic MoE 和 2-GPU EP，验证通信口径。

- `TopKRouter` + `MoELayer` + `EPMoELayer`
- NCCL all-to-all dispatch/gather
- EP 2 卡 / dense 1 卡：1.891×

### Phase 18 — True Expert Sharding
**目标**：验证 per-rank expert shard，统计参数量与 shard ratio。

- 每个 rank 只持有部分 expert 权重
- shard_ratio = 0.5002（接近理想值 0.5）

### Phase 19 — Non-Padded EP Dispatch
**目标**：消除 EP 通信中的 padding 冗余，达到 ideal payload。

- Packed dispatch：ep_packed_bytes = ep_ideal_bytes（exact match）
- EP packed / dense：2.323×
- EP packed / EP padded：1.217×

### Phase 20 — EP Control Plane 收敛
**目标**：量化并收敛 EP 的控制面开销。

- Packed control-plane 显式量化
- control_plane_share ≈ 1.94%
- EP packed / dense：2.310×

### Phase 21 — Grouped Expert Execution
**目标**：对本 rank local expert 进行分组批量执行，减少 overhead。

- Grouped local expert execution
- EP grouped / dense：**2.500×**
- EP grouped / EP packed：1.070×
- runtime resident ratio：0.8334

---

## 阶段验收原则

- 每个阶段结束时项目是**完整可运行**的，不依赖后续阶段
- 每个阶段有**独立 benchmark 脚本**和**可复现的指标**
- 每个阶段的 prototype 边界和口径说明已在实验记录中归档
