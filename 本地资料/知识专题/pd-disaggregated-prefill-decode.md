# PD 解耦（Disaggregated Prefill/Decode）

## 一、问题定义

LLM 推理有两个计算阶段，性质根本不同：

- **Prefill**：输入所有 prompt token，做完整 attention，计算密集（compute-bound），GPU 算力利用率高
- **Decode**：每步生成一个 token，KV cache 随序列增长，内存密集（memory-bound），瓶颈在 KV 读写带宽

统一部署时这两种负载共享同一 GPU，互相干扰：
- 一批长 prompt 到来 → prefill 独占 GPU → 已在 decode 的请求被阻塞 → TTFT 抖动
- Decode 大量请求 → GPU 被 KV 读写占满 → 新请求 prefill 等待

**PD 解耦的目标**：把两种角色分配到独立计算资源，各自在最适合的模式下运行，用 KV 传输连接两端。

---

## 二、核心原理

### 解耦后的数据流

```
客户端 → Router
    → Prefill 节点（处理 prompt，生成 first token + KV cache）
    → KV Transfer（把 KV blocks 发送给 Decode 节点）
    → Decode 节点（接收 KV，从 first token 开始 decode loop）
    → 流式返回 token 给客户端
```

### KV 传输内容

Prefill 节点完成 forward 后，每层都有 `(k, v)` 张量，shape `[batch, num_kv_heads, seq_len, head_dim]`。需要将其序列化后跨进程/跨节点传输。

**传输大小估算（fp16）**：

```
KV_size = num_layers × seq_len × num_kv_heads × head_dim × 2 (K+V) × 2 (fp16 bytes)
```

Qwen2.5-7B, seq=1024: 28 × 1024 × 4 × 128 × 2 × 2 = 58,720,256 B ≈ 56 MB

MLA（DeepSeek-V2-Lite）只传一路压缩向量（latent），约为 GQA 的 54%。

### 传输方案对比

| 方案 | 延迟 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| pickle + Queue | ~15ms（56 MB） | 低 | 同机原型验证 |
| multiprocessing.shared_memory | ~2ms（无序列化） | 中 | 同机生产原型 |
| GPU-Direct / RDMA | < 1ms | 高 | 跨节点生产系统 |
| NVLink（同机多卡） | < 0.5ms | 中 | 同机多卡 |

### 并发 pipeline（生产核心优势）

生产 PD 解耦的主要优势不是**单请求延迟**更短，而是**吞吐更高**：

```
时间轴：
Prefill Worker:  [req1 prefill] [req2 prefill] [req3 prefill] ...
KV Transfer:               [req1 kv] [req2 kv] [req3 kv] ...
Decode Worker:                  [req1 decode...][req2 decode...]...
```

Decode Worker 在处理 req1 的第 k 个 token 时，Prefill Worker 已经在处理 req2 的 prompt。两者完全并行，互不阻塞。

---

## 三、工程实现方式

### 同机双进程原型（mini-infer Phase 15）

```python
# 进程启动：必须用 spawn（不能用 fork，CUDA 不支持）
_ctx = _mp.get_context("spawn")

# 三个队列
prefill_req_queue = _ctx.Queue()  # 主进程 → PrefillWorker
kv_queue = _ctx.Queue()            # PrefillWorker → DecodeWorker
result_queue = _ctx.Queue()        # DecodeWorker → 主进程
```

**PrefillWorker 主循环**：
```python
model, tokenizer, eos_id = _load_model_and_tokenizer(config)
while not stop.is_set():
    req = req_queue.get(timeout=1.0)
    # tokenize → forward（DynamicCache）→ extract KV → send
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    kv_layers = extract_kv_from_past(out.past_key_values, seq_len)
    payload = KVPayload(request_id=req.request_id, kv_layers=kv_layers,
                        first_token_id=first_token_id, ...)
    kv_queue.put(payload)
```

**KV 序列化**：
```python
# DynamicCache: key_cache[i].shape = [1, num_kv_heads, seq, head_dim]
k = cache.key_cache[i][0, :, :seq_len, :].permute(1, 0, 2).cpu()
# → [num_kv_heads, seq, head_dim] → [seq, num_kv_heads, head_dim]
```

**KV 重建**：
```python
# [seq, num_kv_heads, head_dim] → [1, num_kv_heads, seq, head_dim]
k_gpu = k.to(device).permute(1, 0, 2).unsqueeze(0)
cache.update(k_gpu, v_gpu, layer_idx)
```

### 绕过 ModelRunner 的原因

mini-infer 的 `ModelRunner.__init__` 调用 `patch_model_for_paged_decode(model, kv_cache)`，将模型的每个 attention 层永久 patch 成 Paged Attention 路径（直接从 block tensor 寻址）。PD worker 需要的是普通 HF forward（DynamicCache），不能经过这个 patch，因此必须直接加载 HF 模型：

```python
def _load_model_and_tokenizer(config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=dtype, device_map=config.device
    )
    model.eval()
    return model, tokenizer, eos_token_id
```

---

## 四、设计取舍

### KV 格式：`[seq, heads, dim]` vs `[heads, seq, dim]`

选择 `[seq, heads, dim]`（去掉 batch 维度后调整轴顺序）的原因：
- 按 seq 切片更方便（KV 前缀共享的扩展场景）
- 与 batch 独立，batch=1 时去掉 batch 维度无歧义

### 传输机制：pickle+Queue vs 共享内存

选择 pickle+Queue 的原因：
- 实现简单（约 20 行），够验证架构正确性
- 共享内存需要管理生命周期（SharedMemory 的 close/unlink），在多进程错误场景下容易泄漏

代价：~14.7ms 传输延迟（vs 共享内存的 ~2ms）。

### 模型加载：2× 显存

同机原型中两个 worker 各加载一份模型（Qwen2.5-1.5B ~3GB × 2 = ~6GB）。生产系统里 Prefill 和 Decode 在不同节点，不存在这个问题。

---

## 五、常见误区

**误区 1：PD 解耦一定能降低单请求延迟**

PD 解耦的核心优势是**吞吐**，不是单请求延迟。实际上，单请求端到端延迟会因为 KV 传输开销而**增加**（mini-infer 原型 1.19×）。延迟降低依赖并发 pipeline——当 decode 处理 req1 时，prefill 已经在处理 req2，系统整体吞吐更高。

**误区 2：KV 传输要传所有层的完整 K/V**

MLA 模型只需传一路压缩向量（latent），体积约 GQA 的 54%。传输量与缓存量一致，体积不会更大。

**误区 3：Linux 多进程默认是安全的**

Linux 的 `multiprocessing` 默认 `start_method=fork`。fork 之后子进程继承父进程的 CUDA context，但 CUDA driver 不支持 fork 后的多进程使用，结果是子进程死锁或 crash。必须显式指定 `spawn`。

**误区 4：Paged Attention 可以直接在 PD 解耦中复用**

Paged Attention 的 block_table 是单进程的局部状态（free_blocks、block 分配方案都是本地的）。跨进程传输时只传 KV 数据本身（tensor 值），不传 block_table，Decode 侧需要重建自己的 block 管理。

---

## 六、和 mini-infer 的关系

Phase 15 实现了同机双进程 PD 解耦原型：

- `mini_infer/kv_transfer.py`：KV 序列化/反序列化接口
- `mini_infer/pd_worker.py`：Prefill/Decode worker 进程主循环
- `mini_infer/pd_engine.py`：协调层，管理三队列和两子进程

与现有代码的关系：
- 不使用 `LLMEngine`（统一进程引擎）
- 不使用 `ModelRunner`（其 Paged Attention patch 与本场景不兼容）
- 不使用 `KVCacheManager`（Block 管理是单进程状态）
- 用 HF 原生 DynamicCache 存储 KV，适合原型验证

---

## 七、进一步阅读

- **Mooncake**（月之暗面）：生产级 PD 解耦，RDMA KV 传输，KV cache 池化调度
- **Splitwise**（微软）：提出 PD 解耦架构，分析 Prefill/Decode 的计算特征差异
- **vLLM disaggregated prefilling**：vLLM 社区的 PD 解耦实现方案
- Phase 9（Chunked Prefill）：同机缓解 prefill/decode 干扰的替代方案，复杂度更低
