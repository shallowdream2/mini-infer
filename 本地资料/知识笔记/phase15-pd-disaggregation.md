# Phase 15 知识笔记：PD 解耦（Disaggregated Prefill/Decode）

## 核心动机

Prefill（compute-bound）和 Decode（memory-bound）性质根本不同，放在同一进程里互相干扰：

- Prefill 阻塞 Decode → TTFT 抖动
- Decode 占用 GPU → Prefill 吞吐下降

PD 解耦把二者放进独立进程/节点，用 KV 传输连接。

## 数据流

```
PDEngine（主进程）
  ├── PrefillWorker：tokenize → HF forward → extract_kv → KVPayload → kv_queue
  ├── DecodeWorker：kv_queue → rebuild_cache → decode loop → result_queue
  └── generate()：收集结果，按原始顺序返回
```

三个队列：`prefill_req_queue`（主 → Prefill）、`kv_queue`（Prefill → Decode）、`result_queue`（Decode → 主）。

## KV 序列化格式

传输格式：`list[tuple[Tensor, Tensor]]`，每层一个 `(k, v)`，shape `[seq_len, num_kv_heads, head_dim]`

```python
# extract：DynamicCache → 传输格式
# DynamicCache shape: [1, heads, seq, dim]
k = cache.key_cache[i][0, :, :seq_len, :].permute(1, 0, 2).cpu()
# [heads, seq, dim] → [seq, heads, dim]

# rebuild：传输格式 → DynamicCache
k_gpu = k.to(device).permute(1, 0, 2).unsqueeze(0)
# [seq, heads, dim] → [heads, seq, dim] → [1, heads, seq, dim]
cache.update(k_gpu, v_gpu, layer_idx)
```

去掉 batch 维度（batch=1），CPU tensor，避免 CUDA IPC 复杂性。

## KV 大小（理论值）

| 模型 | seq=128 | seq=1024 |
|------|---------|----------|
| Qwen2.5-1.5B（GQA，2KV heads） | 3.50 MB | 28.00 MB |
| Qwen2.5-7B（GQA，4KV heads） | 7.00 MB | 56.00 MB |
| DeepSeek-V2-Lite（MLA，单向量） | 3.80 MB | 30.38 MB |

MLA 只传一路压缩向量（latent + rope），约 GQA 的 54%。

## 两个必须知道的坑

### CUDA + fork = 死锁

Linux 默认 `multiprocessing.start_method=fork`，CUDA 初始化后 fork 导致子进程挂死（exit code -9）。

**修复**：`_ctx = _mp.get_context("spawn")`

spawn 会重启干净的 Python 进程，CUDA 在子进程内重新初始化。代价：启动更慢。

### ModelRunner 不可直接用

`ModelRunner.__init__` 强制调用 `patch_model_for_paged_decode(model, kv_cache)`，把所有 attention 层永久 patch 成 Paged Attention 路径，kv_cache=None 时 crash。

PD worker 只需要普通 HF forward（DynamicCache），**必须绕过 ModelRunner**，直接用 AutoModelForCausalLM 加载模型。

## TTFT 三段分解（实测）

| 指标 | Unified LLMEngine | PDEngine |
|------|------------------|----------|
| 平均端到端 (ms) | 459.2 | 546.0 |
| prefill (ms) | N/A | 12.3 |
| transfer≈ (ms) | N/A | 14.7 |
| decode (ms) | N/A | 519.0 |

transfer_time = total − prefill − decode（近似），14.7ms 主要来自 pickle+Queue IPC。
生产系统（RDMA/共享内存）可降至 < 1ms。

## 与生产系统的差距

| 维度 | 本原型 | 生产（Mooncake）|
|------|--------|----------------|
| 传输机制 | pickle + Queue | RDMA / GPU-Direct |
| 并发 pipeline | 串行 | prefill i+1 与 decode i 并行 |
| 显存 | 2× 模型（每进程一份）| 不同节点正常 |
| KV 格式 | DynamicCache | Paged block tensor |

核心差距是**并发 pipeline**：本原型串行发送，完全没有体现 PD 解耦的吞吐优势。
