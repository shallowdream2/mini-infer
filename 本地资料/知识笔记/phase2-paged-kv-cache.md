# Phase 2 知识笔记：Paged KV Cache、Batch Decode、Continuous Batching

这个文件记录 Phase 2 涉及的核心概念，供快速回顾。

---

## Paged KV Cache

**核心思路**：把 KV cache 切成固定大小的 block，预先分配 GPU tensor 池，用 BlockTable 管理每个请求占哪些物理块。

**存储格式**：
```
k_cache[layer][block_id, slot_id, num_kv_heads, head_dim]
```
- `block_id`：物理块编号（0 到 num_gpu_blocks-1）
- `slot_id`：块内位置（0 到 block_size-1）
- 每个 token 的 KV 占一个 slot

**BlockTable**：`dict[request_id, list[int]]`，值是物理块号的有序列表。第 `pos` 个 token 对应的物理位置：
```python
block_idx = pos // block_size
slot_idx  = pos % block_size
phys_blk  = block_table[request_id][block_idx]
```

**FreeBlockPool**：`deque(range(num_gpu_blocks))`，分配 popleft()，归还 extend()，O(1)。

**与 HF past_key_values 的区别**：
- HF：每个请求 KV 大小无上限，随序列增长动态增加
- Paged：显存有上限（num_gpu_blocks 固定），请求结束即归还

---

## Batch Decode

**问题**：Phase 1 串行 for 循环，每条请求单独 forward，GPU 利用率低。

**方案**：把 N 个请求的最后一个 token 拼成 `[N, 1]` 的 input_ids，KV 聚合到 `[N, kv_heads, max_seq_len, head_dim]`，一次 batch forward。

**左填充对齐**：不同请求序列长度不同，短的在左边补 0。
- 原因：RoPE 位置编码通过 attention_mask 的 cumsum 推导，左填充保证真实 token 的 position_ids 正确；右填充会导致新 token 的位置 ID 错误。
- attention_mask：左侧填充区域为 0，右侧真实 token + 新 token 位置为 1

**KV 聚合（gather_batch_kv）**：遍历 block table，把每个请求的 KV 从 block pool 复制到 dense tensor。每个 decode step 都要复制一次，这是当前 mini-infer 吞吐低于 HF 的主因。

---

## Continuous Batching

**vs 静态 batching**：
- 静态：等一批请求全完成才拉新请求
- Continuous：每个 decode step 后检查能否加入新请求，先完成的立即腾位

**准入条件**：`num_running < max_batch_size` 且 `free_blocks >= blocks_needed`。

**blocks_needed 估算**：`ceil((prompt_len + max_new_tokens) / block_size)`，保守估计，避免运行中 OOM。

**OOM 安全**：
- 请求所需块数超过空闲块且 `num_running==0`：直接 raise RuntimeError，不死循环
- 异常时 try/except 归还所有 running 请求的 KV 块，防止泄漏

---

## Qwen2.5-7B GQA 参数

| 参数 | 值 |
|------|-----|
| num_hidden_layers | 28 |
| num_attention_heads | 28 |
| num_key_value_heads | 4 |
| head_dim | 128 |
| GQA 比例 | 7:1（28 Q heads 共享 4 KV heads） |

**记住**：num_key_value_heads=4，不是 8。从 `config.json` 读，别凭印象。

---

## 实测数据摘要（RTX 4090，batch=8，max_new_tokens=128）

| 引擎 | Throughput | TTFT | Peak Mem |
|------|-----------|------|----------|
| HF baseline | 408.9 tok/s | 23.6ms | 15.88 GB |
| mini-infer Phase 2 | 201.0 tok/s | 18.8ms | 16.42 GB |

- TTFT 对齐，说明 prefill 路径正确
- Throughput 差距来自 `gather_batch_kv()` 的 KV 复制开销
- 多 0.5 GB 显存来自预分配的 block pool（512 块 × 16 × 28 层 × 2 × 4 heads × 128 dim × fp16 ≈ 0.48 GB）
