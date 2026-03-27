# Architecture

mini-infer 的整体架构围绕 **LLMEngine** 展开，逐步扩展出分布式、量化、MoE EP 等能力。

## 请求生命周期

```
HTTP 请求 / Python API
        │
        ▼
  AsyncEngine              后台线程 step loop + asyncio.Queue
        │
        ▼
  LLMEngine                continuous batching 主循环
    ├── Scheduler           waiting / running / swapped / prefilling 队列
    │                       支持 Preemption（GPU→CPU swap）、Priority、Chunked Prefill
    ├── KVCacheManager      BlockTable + FreeBlockPool
    │                       Phase 10：block-level SHA-256 hash + LRU Prefix Cache
    └── ModelRunner
            ├── prefill(requests)
            └── decode_batch(requests)
                    ├── PagedAttention      flash_attn_with_kvcache + block_table
                    ├── CUDA Graph          decode step 静态捕获，消除 Python dispatch
                    └── QuantLinear         W8A8 per-channel int8 / mixed fallback
```

## 模块说明

### Runtime Core

| 模块 | 职责 |
|------|------|
| `engine.py` | `LLMEngine` 主循环，管理 Continuous Batching 完整状态机；`step()` 每轮驱动一次 prefill + decode |
| `scheduler.py` | `Scheduler`，维护 waiting / running / swapped / prefilling 四队列；Phase 7 加入 preemption swap；Phase 9 加入 chunked prefill 状态机 |
| `async_engine.py` | `AsyncEngine`，后台线程 step loop + `asyncio.Queue`；Phase 8 HTTP serving 的异步前端 |
| `config.py` | `EngineConfig` dataclass，集中管理 block_size / max_blocks / num_hidden_layers / num_kv_heads / head_dim 等所有引擎参数 |
| `request.py` | `Request` / `RequestState` / `SamplingParams`，定义请求数据结构和完整生命周期状态 |

### KV Cache

| 模块 | 职责 |
|------|------|
| `kv_cache.py` | `KVCacheManager`，实现 BlockTable + FreeBlockPool；Phase 10 集成 Prefix Cache：block-level SHA-256 链式 hash + LRU eviction + ref_count |

### Attention & Kernels

| 模块 | 职责 |
|------|------|
| `attention.py` | `PagedDecodeContext` + `patch_model_for_paged_decode`，将 HF 模型的 attention 替换为 `flash_attn_with_kvcache` block_table 路径 |
| `triton_attn.py` | Triton decode attention kernel（Phase 6.5），online softmax + GQA，对应 Phase 6.5 benchmark |
| `triton_flash_decode.py` | Flash Decoding split-K kernel（Phase 12.5），长序列并行；seq=4096 时 3.31× vs 标准 Triton，SM 利用率 9%→103% |

### Model Execution

| 模块 | 职责 |
|------|------|
| `model_runner.py` | `ModelRunner`，管理 prefill / decode_batch 执行路径；Phase 12 集成 CUDA Graph 静态捕获（decode step replay） |
| `quantization.py` | `QuantLinear`（W8A8 per-channel int8），`quantize_model`，mixed fallback 策略；1.5B 权重显存 −32.4% |
| `mla_attention.py` | MLA 三种实现：`MLAAttentionNaive` / `MLAAttentionLatentCache` / `MLAAttentionAbsorbed`；DeepSeek-V2/V3 架构，latent cache 体积 −56.25% vs GQA |

### Distributed

| 模块 | 职责 |
|------|------|
| `tp_engine.py` | `TPEngine`，真 TP（NCCL all-reduce，Phase 13）；`mp.spawn` + 文件锁 rendezvous |
| `tp_model_runner.py` | `TensorParallelModelRunner`，Megatron-LM 风格 column/row parallel + all-reduce hook |
| `ep_engine.py` | `EPEngine`，2-GPU Expert Parallelism（NCCL all-to-all，Phase 17–21）；从 padded→packed→grouped 三阶段优化 |
| `replica_engine.py` | `ReplicaEngine`，数据并行副本（Phase 4） |
| `pp_engine.py` | `PPEngine`，HF Pipeline Parallel（Phase 4，吞吐测量用） |

### Specialized Engines

| 模块 | 职责 |
|------|------|
| `spec_engine.py` | `SpecEngine`，draft + target 双模型 Speculative Decoding（Phase 11）；modified rejection sampling，acceptance_rate 55.85% |
| `pd_engine.py` | `PDEngine`，Disaggregated Prefill/Decode（Phase 15）；同机双进程原型，KV 序列化传输 |
| `pd_worker.py` | `PrefillWorker` / `DecodeWorker` + KV 传输辅助；TTFT 三段分解：prefill 12.3ms / transfer ≈14.7ms / decode 519ms |

### MoE

| 模块 | 职责 |
|------|------|
| `moe_layer.py` | `TopKRouter` / `MoELayer` / `EPMoELayer`；Phase 17 基础 EP，Phase 18 per-rank expert shard，Phase 19 non-padded packed dispatch，Phase 21 grouped local execution |
| `moe_model.py` | `SyntheticMoEConfig` / `SyntheticMoEModel`，benchmark 专用 synthetic MoE layer |

### Serving

| 模块 | 职责 |
|------|------|
| `server.py` | FastAPI HTTP server；`GET /v1/models`，`POST /v1/chat/completions`（streaming + non-streaming） |
| `openai_schema.py` | `ChatCompletionRequest` / `ChatCompletionResponse` Pydantic 模型 |
| `clients/chat_client.py` | 交互式 CLI 聊天客户端，支持 dry-run 自动启动临时服务 |

## 关键设计选择

### BlockTable vs 连续 KV Buffer

Phase 1–2 使用连续 KV buffer，Phase 6 切换到 `flash_attn_with_kvcache` 的 `block_table` 路径。block table 允许：
- 不同长度请求的 KV 块共享物理内存
- Prefix Cache 的 block-level 复用
- Speculative Decoding 的高效 KV 管理

### Continuous Batching 状态机

`Scheduler.step()` 每轮返回一批 `running` 请求，其中部分处于 prefill 阶段，其余处于 decode 阶段。`ModelRunner` 根据请求状态分别调用 `prefill()` 和 `decode_batch()`，实现请求粒度的动态批处理。

### CUDA Graph 的适用范围

CUDA Graph 仅对 decode_batch 有效（输入 shape 固定），不适用于 prefill（序列长度动态变化）。Phase 12 的实现对每种 batch size 分别捕获静态图，在 decode 阶段直接 replay，1.5B bs=1 延迟 −28.9%。

### MoE EP 通信演进

Phase 17 → 21 逐步优化：
- Phase 17：dense all-to-all（padded，有冗余通信）
- Phase 19：packed dispatch（ep_packed_bytes = ep_ideal_bytes，消除 padding）
- Phase 21：grouped local execution（2.500× vs dense）

## 后续可扩展方向

- 更细粒度的 prefix cache eviction（支持 token-level 而非 block-level）
- FP8 量化 / Triton INT8 GEMM
- TP + EP 混合并行
- SLO-aware 请求调度
- 完整 serving runtime（multi-lora、request priority API）
