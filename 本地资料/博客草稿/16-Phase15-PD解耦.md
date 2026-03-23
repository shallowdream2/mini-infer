# 从零实现 LLM 推理系统（15）：PD 解耦原型——把 Prefill 和 Decode 拆进两个进程

> 系列最终篇。前 14 个阶段把推理系统从单卡串行跑到了分布式、前沿架构，这一篇做架构级的解耦实验：把 Prefill 和 Decode 放进两个独立进程，用 KV 传输连接，测量 TTFT 的三段分解。

---

## 背景：统一部署的根本矛盾

Prefill 和 Decode 是两种性质截然不同的计算：

- **Prefill**：计算密集（compute-bound），对所有 prompt token 做一次完整 attention，GPU 算力利用率高
- **Decode**：内存密集（memory-bound），每步只生成一个 token，瓶颈在 KV cache 读写带宽

统一部署时，这两种工作负载互相干扰：

1. 一批长 prompt 到来 → prefill 占用整张 GPU → 已在 decode 的请求被阻塞 → TTFT 抖动
2. 大量请求在 decode → GPU 被 KV 读写占满 → 新请求的 prefill 排队等待

Phase 9 用 Chunked Prefill 把长 prefill 分块投送，ITL spike 降低 57-67%，但两者仍在同一进程抢同一块 GPU，矛盾只是缓解，没有消除。

**PD 解耦的思路**（Mooncake/Splitwise 架构）：把两个角色分配到不同节点（或进程），Prefill 节点做完 prefill 后，把生成的 KV cache 通过网络传给 Decode 节点，Decode 节点拿到 KV 后接着做 decode loop。

本文记录在 mini-infer 里实现一个**同机双进程原型**的过程。

---

## 问题定义

实现 PD 解耦需要解决三个子问题：

1. **KV 序列化**：prefill 产出的 `past_key_values`（HF DynamicCache，每层 `[batch, heads, seq, dim]`）如何提取成可传输的格式？
2. **进程间传输**：用什么机制把 KV 数据从 Prefill 进程传给 Decode 进程？
3. **KV 重建**：Decode 进程收到数据后如何重建成可以直接做 forward 的 `DynamicCache`？

同时还有一个工程约束：mini-infer 的现有 `LLMEngine` 和 `ModelRunner` 能不能直接复用？

---

## 方案设计

### 整体架构

```
PDEngine（主进程）
    ├── PrefillWorker（子进程）
    │       tokenize → HF forward → extract_kv → KVPayload → kv_queue
    ├── DecodeWorker（子进程）
    │       kv_queue → rebuild_cache → decode loop → DecodeResult → result_queue
    └── generate() 收集结果，按原始顺序返回
```

三个队列：`prefill_req_queue`（主进程 → Prefill）、`kv_queue`（Prefill → Decode）、`result_queue`（Decode → 主进程）。

### KV 序列化格式选择

DynamicCache 的存储格式是 `[batch, heads, seq, dim]`，跨进程传输时需要变成可 pickle 的格式。设计约束：

- 只传 KV tensor 数据，不传 `KVCacheManager` 内部状态（block_table / free_blocks 等都是单进程状态）
- Decode 侧重新走 HF DynamicCache，不复用 mini-infer 的 Paged Attention 路径

最终格式：`list[tuple[Tensor, Tensor]]`，每元素是一层的 `(k, v)`，shape `[seq_len, num_kv_heads, head_dim]`（去掉 batch 维度，batch=1，调整轴顺序便于按 seq 切片）。

```python
# extract_kv_from_past：DynamicCache → list[(k, v)]
k = past_key_values.key_cache[i][0, :, :seq_len, :].permute(1, 0, 2).cpu()
# [batch, heads, seq, dim] → [heads, seq, dim] → [seq, heads, dim]
```

反向（重建 DynamicCache）：

```python
# _rebuild_dynamic_cache：list[(k, v)] → DynamicCache
k_gpu = k.to(device).permute(1, 0, 2).unsqueeze(0)
# [seq, heads, dim] → [heads, seq, dim] → [1, heads, seq, dim]
cache.update(k_gpu, v_gpu, layer_idx)
```

### 传输机制：Queue + pickle

同机场景首选 `multiprocessing.shared_memory`（零拷贝），但 `pickle+Queue` 实现更简单且足够验证架构正确性。对于 Qwen2.5-1.5B seq=128，KV 大小约 3.5 MB，pickle 开销可接受。

---

## 实现中的两个坑

### 坑 1：CUDA + fork = 死锁

Linux 的 `multiprocessing` 默认使用 `fork` 启动子进程。CUDA 在 fork 后无法正常工作——子进程继承了父进程的 CUDA context，但 CUDA driver 不支持 fork 后的多进程使用，轻则 CUDA 初始化报错，重则子进程直接挂死。

症状：子进程启动后，`PDEngine.generate()` 等待 120 秒后超时，`result_queue` 没有任何数据，子进程 exit code 为 -9（被 killed）或无响应。

修复：强制使用 `spawn` context。

```python
# pd_engine.py
_ctx = _mp.get_context("spawn")  # CUDA 不兼容 fork
```

`spawn` 会重新启动一个干净的 Python 进程，CUDA 在子进程内重新初始化，无冲突。代价是启动时间比 fork 慢（需要重新 import 所有模块）。

### 坑 2：ModelRunner 不能传 kv_cache=None

mini-infer 的 `ModelRunner.__init__` 在 GPU 模式下会调用 `patch_model_for_paged_decode(model, kv_cache)`，把每个 attention 层永久 patch 成 Paged Attention 路径。这个 patch 需要 `kv_cache` 不为 None（用于在 decode 时直接从 block tensor 寻址）。

PD 解耦的 worker 不需要也不应该用 Paged Attention——它只需要做普通的 HF forward，把 KV 存在 DynamicCache 里。如果传 `kv_cache=None`，`patch_model_for_paged_decode` 在第一次 decode forward 时就会 crash。

修复：在 worker 内绕过 `ModelRunner`，直接加载 HF 模型。

```python
def _load_model_and_tokenizer(config: EngineConfig):
    """在 worker 进程内直接加载模型（不经过 ModelRunner，避免 paged decode patch）。"""
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=dtype, device_map=config.device
    )
    model.eval()
    return model, tokenizer, eos_token_id
```

---

## 实验结果

**环境**：Ubuntu 24.04，RTX 4090，Qwen2.5-1.5B-Instruct，fp16，batch=1

### 正确性验证（Section 2）

| | LLMEngine（统一进程） | PDEngine（PD 解耦） |
|--|--|--|
| 输出（greedy） | `The capital of France is Paris...` | `The capital of France is Paris...` |
| 一致性 | — | **✓ token 级完全相同** |

### TTFT 分解（Section 3）

**Workload**：3 个 prompt，max_new_tokens=64，greedy，1 次预热

| 指标 | Unified LLMEngine | PDEngine |
|------|------------------|----------|
| 平均端到端时间 (ms) | 459.2 | 546.0 |
| prefill 时间 (ms) | N/A | 12.3 |
| transfer 时间·近似 (ms) | N/A | 14.7 |
| decode 时间 (ms) | N/A | 519.0 |
| 相对开销 | 1.00× | **1.19×** |

*transfer_time = total − prefill − decode，是近似估算，包含 Queue IPC + pickle 开销。*

几个值得关注的数字：

- **prefill 只占 12.3ms**：1.5B 模型 + 短 prompt，prefill 本身非常快。真实场景下长 prompt 的 prefill 会更明显（≥ 100ms），PD 解耦的收益才会体现。
- **decode 占 519ms**：64 个 token ÷ 519ms ≈ 8ms/token。这是单 token batch=1 的 decode 延迟，decode 侧的主要耗时。
- **transfer≈14.7ms**：这是 pickle+Queue 的 IPC 开销，不是真实的网络传输。Mooncake 用 RDMA，同机 GPU-Direct 可降至 < 1ms。

### KV 传输大小（Section 1，理论值）

| 模型 | seq=128 | seq=1024 |
|------|---------|----------|
| Qwen2.5-1.5B（GQA） | 3.50 MB | 28.00 MB |
| Qwen2.5-7B（GQA） | 7.00 MB | 56.00 MB |
| DeepSeek-V2-Lite（MLA） | 3.80 MB | 30.38 MB |

MLA 只存一路压缩向量（latent + rope），即使 27 层也只有 GQA 的约 54%。

---

## 与生产 PD 解耦系统的差距

本原型验证了架构正确性，但与生产系统（Mooncake、vLLM 的 PD 解耦版本）有几个关键差距：

| 维度 | mini-infer 原型 | 生产系统 |
|------|----------------|--------|
| 传输机制 | pickle + multiprocessing.Queue | RDMA / GPU-Direct / 共享内存 |
| 传输延迟 | ~14.7ms（近似） | < 1ms |
| 并发 pipeline | 串行（prefill i 完才发送 i+1）| prefill i+1 与 decode i 并行 |
| 显存占用 | 两进程各加载一份模型（2×） | 不同节点各加载一份（正常） |
| KV 格式 | DynamicCache（HF 格式） | Paged block tensor（直接映射 GPU 地址）|
| 跨节点 | 同机 | 真正多节点 |

最核心的差距是 **并发 pipeline**：生产 PD 解耦的优势不是单请求延迟更短，而是 throughput 更高——当 Decode Worker 在处理请求 i 的第 k 个 token 时，Prefill Worker 已经在处理请求 i+1 的 prompt。两者完全并行，互不阻塞。本原型串行发送请求，完全没有体现这一优势。

---

## 回顾：Phase 9 vs Phase 15

两个阶段都在解决 prefill 对 decode 的干扰问题，路径不同：

| | Phase 9 Chunked Prefill | Phase 15 PD 解耦 |
|--|--|--|
| 思路 | 把 prefill 拆成小 chunk，穿插在 decode 步骤之间 | 把 prefill 和 decode 放进不同进程 |
| 实现复杂度 | 调度器改造（约 100 行） | 跨进程协议 + KV 序列化（约 400 行） |
| 适用场景 | 单机，长 prompt 不饿死短 decode | 多机，算力/内存专用化 |
| ITL 改善 | −57%~−67%（实测） | N/A（本原型不测并发场景）|
| 引入复杂度 | 中（调度器状态机） | 高（进程间协议、KV 传输、spawn 约束）|

Chunked Prefill 是在"共享资源"框架内的调度优化，PD 解耦是"彻底分离资源"的架构变革。两者不互斥，生产系统可以同时用（Prefill 节点内部也可以做 Chunked Prefill）。

---

## 总结

1. **正确性达成**：两进程跨边界生成，greedy 输出与统一进程完全一致，KV 序列化/反序列化数学路径正确。
2. **TTFT 分解达成**：首次在 mini-infer 中实现 prefill / transfer / decode 三段独立计时（12.3ms / 14.7ms / 519ms）。
3. **传输延迟未达目标**：< 10ms 的目标在 pickle+Queue 下无法实现（实测≈14.7ms），需要共享内存才能达标。
4. **并发 pipeline 未实现**：PD 解耦的核心吞吐优势依赖 prefill/decode 并行，本原型验证的是架构正确性，不是吞吐提升。

至此，mini-infer 系列的 15 个阶段全部完成——从单卡串行到分布式 PD 解耦，覆盖了现代 LLM 推理系统的主要技术方向。

---

## 自查结果（QUALITY_CHECKLIST.md）

- [x] 文章主题对应真实项目阶段或真实技术问题（Phase 15 PD 解耦，有真实实现和 benchmark）
- [x] 结构覆盖：背景、问题定义、方案设计、实现细节、实验结果、坑点反思、总结
- [x] 有明确背景和问题定义（统一部署的 compute/memory 矛盾，三个子问题拆分）
- [x] 有设计取舍（KV 格式选择、传输机制选择 pickle vs 共享内存、绕过 ModelRunner 的理由）
- [x] 有代码、实验或运行结果支撑（数字来自 2026-03-23-phase15-benchmark.md；代码片段来自源文件）
- [x] 有失败点、坑点或反思（fork+CUDA 死锁、ModelRunner patch 崩溃；传输目标 14.7ms 未达 < 10ms；并发 pipeline 未实现）
- [x] 语言克制、专业、可发表（无夸大性能宣称，局限性明确标注）
