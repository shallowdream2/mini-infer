# Phase 3 知识笔记：向量化 KV Gather 与 DynamicCache

本文件是 mini-infer Phase 3 的核心概念速记，供自己回顾。

---

## 为什么 Python 循环是推理瓶颈

GPU 做矩阵乘很快，但每次 Python 层面的索引操作（哪怕只是 `tensor[i, j]`）都有 Python 解释器调度开销，如果循环次数多（layers × batch × seq_len），这个开销会超过 GPU 实际算力的消耗。

Phase 2 的 gather：28层 × 8batch × 128seq_len = 28672 次 Python 迭代。不是 GPU 太慢，是 CPU 拖后腿。

---

## Advanced Indexing 一次性 Gather

关键：把所有维度的索引提前算好（两个 `[batch, max_seq_len]` 的 tensor），然后 `k_cache[l][phys_blocks, slot_indices]` 一次完成整层的 gather。

物理块号的推导：
```
block_indices = token_pos_clamped // block_size
slot_indices  = token_pos_clamped % block_size
phys_blocks   = block_table_tensor[batch_range, block_indices]
```

---

## 左填充的数学

不同请求 seq_len 不同，gather 后要左填充对齐：
```
token_position[b, i] = i - (max_seq_len - seq_len[b])
```
负值 = 填充区。填充区 clamp 到 0 后 gather 出"随机"值，再乘 `valid_mask_f=0` 置零。

**关键：valid_mask 是 bool，乘 float16 前要 `.to(dtype=cache_dtype)`，否则隐式转换。**

---

## DynamicCache 迁移的坑

问题：改了 `decode_batch()` 后 warning 还在。

根因：transformers 4.43.4 在 `past_key_values=None` 时内部生成 tuple 并发出 warning，是模型内部行为，不是调用方的问题。

修复：`prefill()` 里传 `past_key_values=DynamicCache()`。

诊断方式：`-W error::UserWarning` 把 warning 变成 exception，traceback 直接指向根因位置。

---

## gather 的残余开销

向量化消除了 Python 循环，但 gather 本身仍是把分散 block 数据复制到 dense tensor 再传给 HF 模型。要彻底消除，需要在 attention 计算里直接支持 block 寻址（flash_attn 2.5+ 的 `block_tables`）。

Phase 3 实现的是"用 CUDA kernel 做复制"，而不是"不复制"。这个区别很重要，别在简历或文章里混淆。

---

## DynamicCache API 要点

- `cache = DynamicCache()`
- `cache.update(k, v, layer_idx)` 写入一层 KV
- `out.past_key_values.key_cache[l]` 读取 KV，shape `[batch, num_kv_heads, seq_len, head_dim]`
- forward 后最后一个位置 `[:, :, -1, :]` 是新 token 的 KV
