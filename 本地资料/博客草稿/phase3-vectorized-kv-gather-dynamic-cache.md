# 从 49% 到 88%：向量化 KV Gather 让 batch=8 吞吐翻倍

这篇文章是 mini-infer 项目的第三篇技术复盘。Phase 2 跑通了 Paged KV Cache + Batch Decode，但 batch=8 throughput 只有 HF baseline 的 49.1%。Phase 3 的目标是找到根本瓶颈并消除它。实测结果：batch=8 throughput 从 201 tok/s 涨到 361 tok/s，达到 HF baseline 的 88.4%。

---

## 一、Phase 2 的残余问题

Phase 2 的 `gather_batch_kv()` 是这样的：

```python
# Phase 2：嵌套 Python 循环
for l in range(num_layers):
    for b, rid in enumerate(request_ids):
        seq_len = self._seq_lens[rid]
        block_table = self._block_tables[rid]
        for i in range(seq_len):
            blk_idx = i // self.block_size
            slot_idx = i % self.block_size
            phys_blk = block_table[blk_idx]
            k_dense[l, b, i] = self.k_cache[l][phys_blk, slot_idx]
```

三层循环：`num_layers × batch_size × seq_len`。以 Qwen2.5-7B（28 层）、batch=8、seq_len=128 为例，这是 28 × 8 × 128 = 28672 次 Python 迭代，每次都是一次小的 GPU tensor 索引操作。

**每次小索引就是一次独立的 CUDA kernel 调用，或者至少是一次 Python 层面的同步点。** Python 解释器的开销不在计算上，在调度上——每次 `.item()`、每次标量索引，CPU 和 GPU 之间都要协调一次。

这解释了 Phase 2 的数据：batch=1 时没有 gather（单请求无需聚合），throughput 88%；batch=8 时 28672 次 Python 循环，throughput 跌到 49%。瓶颈不在 GPU，在 CPU 的 Python 解释器。

---

## 二、向量化方案

核心思路：把 3 层嵌套循环变成一次 PyTorch advanced indexing。

### 2.1 把 BlockTable 变成 Tensor

```python
# [batch, max_num_blocks]
max_num_blocks = max(len(self._block_tables[rid]) for rid in request_ids)
block_table_tensor = torch.zeros(batch_size, max_num_blocks, dtype=torch.long, device=device)
for b, rid in enumerate(request_ids):
    blocks = self._block_tables[rid]
    block_table_tensor[b, :len(blocks)] = torch.tensor(blocks, dtype=torch.long, device=device)
```

这里有一个 Python 循环，但它只遍历 batch（8 次），而不是 seq_len（128 次）。后续不再需要访问 `self._block_tables` 字典。

### 2.2 计算每个输出位置的物理地址

每个请求长度不同，gather 后需要左填充对齐到 max_seq_len。左填充的含义是：对于 seq_len=3、max_seq_len=6 的请求，output position 0-2 是填充（零），position 3-5 才是真实 token。

```python
seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device).unsqueeze(1)  # [batch, 1]
out_positions = torch.arange(max_seq_len, device=device).unsqueeze(0)              # [1, max_seq_len]

# token_positions[b, i] = i - (max_seq_len - seq_len[b])
# 负值 = 填充区，非负值 = 对应真实 token 的逻辑位置
token_positions = out_positions - (max_seq_len - seq_lens_t)  # [batch, max_seq_len]

valid_mask = token_positions >= 0
token_pos_clamped = token_positions.clamp(min=0)  # 填充区 clamp 到 0，gather 结果后续被 mask 置零
```

然后推算物理块号和块内 slot：

```python
block_indices = token_pos_clamped // self.block_size   # [batch, max_seq_len]
slot_indices  = token_pos_clamped % self.block_size    # [batch, max_seq_len]

# phys_blocks[b, i] = block_table_tensor[b, block_indices[b, i]]
batch_range = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(block_indices)
phys_blocks = block_table_tensor[batch_range, block_indices]  # [batch, max_seq_len]
```

### 2.3 一次 advanced indexing 完成 gather

```python
valid_mask_f = valid_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=cache_dtype)  # [batch, max_seq_len, 1, 1]

for l in range(num_layers):
    # k_cache[l]: [num_gpu_blocks, block_size, num_kv_heads, head_dim]
    # advanced indexing → [batch, max_seq_len, num_kv_heads, head_dim]
    k_tokens = self.k_cache[l][phys_blocks, slot_indices] * valid_mask_f
    v_tokens = self.v_cache[l][phys_blocks, slot_indices] * valid_mask_f
    # permute → [batch, num_kv_heads, max_seq_len, head_dim]
    k_batch.append(k_tokens.permute(0, 2, 1, 3))
    v_batch.append(v_tokens.permute(0, 2, 1, 3))
```

仍然有一个 `for l in range(num_layers)` 循环，但这层只有 28 次（固定的层数），不随 batch 或 seq_len 增长。每层内部是一次 `k_cache[l][phys_blocks, slot_indices]`，这是一次 PyTorch advanced indexing，最终对应**一次 CUDA kernel**。

相比 Phase 2 的 28 × 8 × 128 = 28672 次 Python 循环 → Phase 3 的 28 次 Python 循环，量级差了 1000 倍。

### 2.4 填充区为什么不会出错

`token_pos_clamped` 对填充位 clamp 到 0，这意味着填充位会去 gather block 0 的 slot 0——也就是某个请求的第一个 token 的 KV（或者是未初始化的零值块）。这个值本身是错的，但 `valid_mask_f = 0` 会把它乘零，最终输出为零。所以填充位的 gather 目标是什么并不重要，只要乘以 mask 就能正确置零。

这个细节在测试里验证过：

```python
# req_b seq_len=3, max_seq_len=6：前 3 位应为零
expected_b = torch.tensor(
    [[[0, 0], [0, 0], [0, 0], [20, 30], [21, 31], [22, 32]]], dtype=torch.float32
)
assert torch.allclose(k_batch[0][1], expected_b)
```

---

## 三、DynamicCache：警告出在哪里

Phase 2 还有一个问题：每次运行都会打印：

```
UserWarning: `past_key_values` must be a `DynamicCache` instance, using a tuple is deprecated...
```

自然的判断是在 `decode_batch()` 里——那里把 gathered KV 构造成 tuple 传给模型。于是先修了 decode_batch：

```python
# 改造后的 decode_batch
cache = DynamicCache()
for l in range(num_layers):
    cache.update(k_batch[l], v_batch[l], l)
out = self.model(input_ids=input_ids, past_key_values=cache, ...)
```

跑起来，警告还在。

加上 `-W error::UserWarning` 让警告变成异常并打印 traceback：

```
File ".../model_runner.py", line 121, in prefill
    out = self.model(input_ids=input_ids, use_cache=True)
```

警告出在 `prefill()`，不是 `decode_batch()`。

根因：transformers 4.43.4 里，当 `model()` 收到 `past_key_values=None`（即不传这个参数），内部会走一条旧路径，把 `past_key_values` 构造成 tuple 然后再发出弃用警告——**不是调用方传了 tuple，是模型内部生成了 tuple 并警告自己**。

修复：显式传一个空的 DynamicCache：

```python
out = self.model(input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True)
```

这样模型知道调用方期望 DynamicCache 格式，走新路径，不触发警告。两个字的修复，但不读源码找不到根因。

---

## 四、实测数据

环境：Ubuntu 24.04 + RTX 4090，Qwen2.5-7B-Instruct，float16，transformers 4.43.4，flash-attn 2.3.6。

### 标准 benchmark（max_new_tokens=128）

| batch | Phase 2 | Phase 3 | HF baseline |
|-------|---------|---------|-------------|
| 1 | 49.4 tok/s | 53.7 tok/s | 56.2 tok/s |
| 4 | 135.2 tok/s | 194.2 tok/s | 210.5 tok/s |
| 8 | 201.0 tok/s | 361.3 tok/s | 408.9 tok/s |

Phase 3 / HF：batch=1 为 95.5%，batch=4 为 92.3%，batch=8 为 88.4%。

batch=1 改善有限（+8.7%），因为 batch=1 时没有 gather 压力，Phase 2 已接近最优。批次越大，向量化的收益越显著。

### 混合长度 benchmark

8 条请求：短 prompt（"一句话介绍 KV Cache"）× 4，长 prompt（"请详细解释……"）× 4，统一 max_new_tokens=256。

Throughput：**356.0 tok/s**，Peak Mem：**17.00 GB**。

注：当前 `generate()` 不支持 per-request `max_new_tokens`，短 prompt 不会因为答完提前释放 KV 块，实际以最大值 256 统一运行。混合 benchmark 的意义更多在于验证混合长度场景的稳定性，而不是完整体现 continuous batching 的调度优势。

---

## 五、剩余差距在哪里

Phase 3 与 HF baseline 仍有 ~10% 的差距（batch=8），主要来自两个地方：

**1. gather 本身仍有每步 KV 复制**

向量化消除了 Python 循环，但 gather 操作本身仍需把分散的 block 数据复制到连续 dense tensor，然后传给 HF 模型。HF 的 KV 是连续分配的，不需要这次复制。

彻底消除需要在 attention 计算内直接支持分散的 block 寻址——即 flash_attn 2.5+ 的 `block_tables` 参数（PagedAttention 的真正核心）。当前环境 flash_attn 2.3.6 不具备这个能力。升级到 2.5+ 是 Phase 4 的选项之一。

**2. 预分配 block tensor pool**

mini-infer 预分配 512 个 block（约 1.8 GB），HF 按需分配。这部分不影响 throughput，但使 Peak Mem 比 HF 高约 1 GB。

---

## 六、一些工程细节

**valid_mask 的 dtype**：`valid_mask = token_positions >= 0` 得到 bool tensor。直接 `k_tokens * valid_mask_f` 会触发 PyTorch 隐式类型转换。正确做法是 `valid_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=cache_dtype)` 显式转换为 float16，避免隐式转换带来的精度和性能问题。

**从 `out.past_key_values` 提取新 KV**：DynamicCache forward 后，`out.past_key_values.key_cache[l]` 的 shape 是 `[batch, num_kv_heads, max_seq_len+1, head_dim]`，最后一个位置（`[:, :, -1, :]`）才是本次新 token 的 KV。

**及时释放大张量**：decode_batch 里 gathered KV 是 `[num_layers, batch, num_kv_heads, max_seq_len, head_dim]` 量级的大 tensor。写回 block tensor 后立刻 `del k_batch, v_batch, cache, out, k_new, v_new`，避免在下一步采样时峰值显存叠加。

---

## 七、小结

Phase 3 做了两件事：

- **向量化 gather_batch_kv()**：3 层 Python 嵌套循环 → 28 次 Python + 每层 1 次 CUDA kernel，batch=8 throughput 从 Phase 2 的 49.1% HF 提升到 88.4% HF
- **DynamicCache 迁移**：消除 transformers 弃用警告，根因在 prefill 不在 decode，修复是传空实例 `DynamicCache()`

剩余约 12% 的差距来自 gather 本身的 KV 复制，解决路径是 flash_attn 2.5+ 或自写 Triton kernel。这留给 Phase 4。

从另一个角度看：Phase 2 的瓶颈实际上是 CPU Python 解释器，而不是 GPU 算力。一个看似"GPU 密集"的推理系统，完全可以被 Python 层面的循环拖垮。理解这一点，是做推理优化的基本前提。
