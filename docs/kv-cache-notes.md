# KV Cache 与 PagedAttention 机制笔记

> 结合 mini-infer `cache/kv_cache.py` 和 `kernels/attention.py` 源码整理。

---

## 1. 为什么需要 KV Cache？

Transformer decode 每步生成 1 个 token，但 Attention 需要看到**所有历史 token 的 K、V**。
不缓存 → 每步从头重算所有历史，时间复杂度 O(n²)。
缓存后 → 每步只算新 token 的 Q、K、V，历史 K、V 直接读缓存，时间复杂度 O(n)。

代价：显存。一个 7B 模型（32 层，32 KV heads，head_dim=128，fp16）：
```
每 token 的 KV：2 × 32层 × 32头 × 128维 × 2字节 = 524,288 字节 ≈ 0.5 MB
1024 tokens：≈ 512 MB
并发 8 请求 × 2048 tokens：≈ 8 GB
```
显存管理成为 serving 系统的核心瓶颈。

---

## 2. 朴素 KV Cache 的问题

HF Transformers 用 `DynamicCache`（list of tensors）：每个请求独立一块连续 GPU 内存。

问题：
- **内存碎片**：不同请求长度不同，连续分配导致碎片
- **无法共享**：两个请求有相同前缀（如 system prompt），却存了两份
- **batch decode 困难**：长度不齐的多个请求无法高效合并

---

## 3. PagedAttention：核心思想

受操作系统虚拟内存分页启发：
- 将 KV cache 切成**等大小的物理块**（block_size=256 tokens）
- 每个请求维护**逻辑块 → 物理块**的映射表（BlockTable）
- 物理块由全局 FreeBlockPool 统一管理

```
物理 Block Pool（GPU 上一整块 tensor）：
  k_cache[layer] shape: [num_gpu_blocks, block_size, num_kv_heads, head_dim]
  v_cache[layer] 同上

请求 A 的 BlockTable：[42, 17, 93]   ← 3个逻辑块，映射到物理块 42、17、93
请求 B 的 BlockTable：[7, 55]        ← 2个逻辑块，映射到物理块 7、55

flash_attn_with_kvcache(
    q=...,
    k_cache=k_cache,   ← 整个 pool
    v_cache=v_cache,
    block_table=[[42,17,93], [7,55,0]],  ← 每请求的映射
    cache_seqlens=[768, 512],             ← 每请求实际序列长度
)
```

关键代码：`cache/kv_cache.py:KVCacheManager.build_block_tables()`

---

## 4. mini-infer 的 BlockTable 生命周期

### 分配（Prefill 前）
```python
# engine.py: step() 准入阶段
kv_cache.init_request(state)
# → 按 ceil(prompt_len / block_size) 分配物理块
# → _block_tables[request_id] = [phys_block_0, phys_block_1, ...]
# → 每块 _ref_count[block] = 1
```

### 扩展（Decode 每步）
```python
# engine.py: step() decode 阶段
kv_cache.ensure_next_slot(state)
# → 检查当前最后一块是否已满（seq_len % block_size == 0）
# → 已满则分配新块，追加到 block_table
```

### 释放（请求完成）
```python
kv_cache.free_request(state)
# → 遍历 block_table，每块 ref_count -= 1
# → ref_count == 0 → 归还 _free_blocks（shared block 不会立即释放）
```

---

## 5. 显存占用分析

```
总显存 = 模型权重 + KV Block Pool + Activations

KV Block Pool 大小：
  num_gpu_blocks × block_size × num_layers × 2(K+V) × num_kv_heads × head_dim × dtype_bytes

示例（Qwen2.5-7B，num_gpu_blocks=200，block_size=256）：
  200 × 256 × 32 × 2 × 4 × 128 × 2 = 2,684,354,560 bytes ≈ 2.5 GB

最大并发请求数上界：
  max_concurrent ≈ num_gpu_blocks / (avg_seq_len / block_size)
  avg_seq_len=512：max_concurrent ≈ 200 / 2 = 100
  avg_seq_len=2048：max_concurrent ≈ 200 / 8 = 25
```

---

## 6. Prefix Caching（Phase 10）

场景：100 个请求共享同一个 system prompt（如 "你是一个助手..."），
朴素方案每次 prefill 都重算这部分 KV。

### 方案：Block-level SHA-256 链式 Hash
```
token_ids = [t0, t1, ..., t255,   t256, ..., t511,   t512, ..., t767, ...]
            └── block 0 ──────┘  └── block 1 ──────┘  └── block 2 ──

hash(block 0) = SHA256(struct.pack(">256i", *block_0_tokens))
hash(block 1) = SHA256(prev_hash + struct.pack(">256i", *block_1_tokens))
                         ↑ 链式：包含历史上下文，防止哈希碰撞
```

命中逻辑：
```python
# 新请求到来，先查前缀
cached_len, cached_blocks = kv_cache.find_prefix_cache(token_ids)
if cached_len > 0:
    kv_cache.init_request_with_prefix(state, cached_len, cached_blocks)
    # → block_table 前缀部分复用已缓存的物理块
    # → _ref_count[cached_block] += 1（避免被 LRU 淘汰）
    # → 只 prefill token_ids[cached_len:] 部分（suffix）
```

### 引用计数保证正确性
```
ref_count = 1：只有请求持有（或只有 cache 持有）
ref_count = 2：cache 持有 + 某请求正在使用（不可淘汰）

淘汰条件：ref_count == 1 且 LRU 队尾
  → evict_lru_prefix_block()
  → ref_count -= 1 → 0 → 归还 _free_blocks
```

---

## 7. Continuous Batching 下的 KV Cache

传统静态 batching：一批请求同时开始、同时结束，不同长度填 padding → 显存浪费。

Continuous Batching：请求随到随入，完成立即释放：
```
时间轴：
  t=0  请求A(len=100) 进入，分配 1 个 block
  t=0  请求B(len=500) 进入，分配 2 个 block
  t=5  请求A 完成，1 个 block 立即归还 free_pool
  t=5  请求C(len=200) 进入，使用归还的 block
  t=10 请求B 完成，blocks 归还
```

物理块被不同请求在时间维度上复用，不浪费显存。

---

## 8. Chunked Prefill 对 KV Cache 的影响

长 prompt（如 4096 tokens）一次性 prefill 会：
1. **ITL spike**：prefill 期间 decode batch 被阻塞，已有请求的 Inter-Token Latency 暴增
2. **显存峰值**：大 activation tensor

Chunked Prefill（chunk_size=256）：
- 每步只处理 256 tokens，多步完成一个请求的 prefill
- 中间 DynamicCache 存在 `engine._prefilling_caches[rid]`（GPU 上）
- 每步结束时 decode batch 正常执行，ITL spike **−57~67%**

---

## 9. 关键数据速查

| 指标 | 数值 | 来源 |
|------|------|------|
| batch=8 吞吐（True PagedAttention） | 406 tok/s = HF 的 100% | benchmark_flash.py |
| Chunked Prefill ITL spike 降低 | −57%（chunk=256）/ −67%（chunk=128） | benchmark_chunked_prefill.py |
| Prefix Cache TTFT 节省 | −22%（1 block 命中） | benchmark_prefix_cache.py |
| CUDA Graph decode 延迟降低 | −28.9%（bs=1） | benchmark_cuda_graph.py |
| W8A8 量化权重显存节省 | −32.4%（3392→2292 MB） | benchmark_quant.py |
| MLA latent cache vs GQA | −56.25% | benchmark_mla.py |
