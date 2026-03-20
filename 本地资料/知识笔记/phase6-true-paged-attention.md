# Phase 6 知识笔记：True PagedAttention

这个文件记录 Phase 6 涉及的核心概念，供自己快速回顾用。

---

## flash_attn_with_kvcache 的 block_table 参数

```python
flash_attn_with_kvcache(
    q,           # (batch, 1, num_q_heads, head_dim)
    k_cache,     # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache,     # 同上
    k=k_new,     # (batch, 1, num_kv_heads, head_dim) — 新 token 的 K（会被写入 k_cache）
    v=v_new,
    cache_seqlens=cache_seqlens,   # (batch,) int32 — 各请求当前 cache 长度
    block_table=block_table,       # (batch, max_blocks) int32 — 物理块映射
    causal=True,
)
```

一次调用同时完成：(1) 把 k_new/v_new 写入 k_cache/v_cache 正确位置，(2) 用 block_table 寻址计算 attention。
副作用：k_cache 和 v_cache 被 in-place 修改。

## block_size 必须是 256 的倍数

flash_attn_with_kvcache 的内核约束：`k_cache.shape[1] % 256 == 0`，否则报 RuntimeError。
实际用 block_size=256（一块可存 256 个 token）。200 块 × 256 = 51200 token 容量。

## PagedDecodeContext：共享状态的正确传递方式

28 层 patched_forward 都需要同样的 block_table、cache_seqlens、max_kv_len。
直接传参会破坏 HF model.forward() 的接口，用一个可变对象（context）在层间共享：

```python
ctx.set(block_table, cache_seqlens, max_kv_len)  # forward 前
model.forward(...)                                  # 28 层各读 ctx
ctx.clear()                                         # forward 后（try/finally 保证）
```

`ctx.block_table is None` 作为 prefill/decode 的路由信号。

## .item() 是隐式 CPU-GPU sync

任何把 GPU tensor 读到 CPU 的操作（`.item()`, `.numpy()`, `print(gpu_tensor)`）都需要等待 GPU 上所有 kernel 完成。

在被 N 层各调用一次的函数里：N × sync_cost。Qwen2.5-7B 28 层 × 18ms/sync = 504ms。

**解决方法**：提前在 CPU 侧算好需要的标量值，通过 context 对象传入。

## RoPE 在 decode 路径的处理

flash_attn_with_kvcache 有内置 `rotary_cos/sin` 参数，但 Qwen2.5 的 RoPE 格式（非 interleaved）与之不兼容。

做法：手动调用 `rotary_emb(v, seq_len=max_kv_len)` + `apply_rotary_pos_emb(q, k, cos, sin, position_ids)`。

`position_ids = cache_seqlens.long().unsqueeze(1)`（每个请求的新 token 位置）。
`max_kv_len = max(cache_seqlens) + 1`（确保 cos/sin 覆盖最大位置，避免越界）。

## 预分配 KV cache 的显存代价

mini-infer 预分配 num_gpu_blocks 个 block，HF DynamicCache 按需分配。

```
200 blocks × 256 tokens × 28 layers × 2(K+V) × 4 kv_heads × 128 head_dim × 2 bytes ≈ 2.93 GB
```

这是 Paged KV Cache 的取舍：以固定显存换取 O(1) 分配、无碎片、支持 preemption。

## 性能结果（batch=8, Qwen2.5-7B, float16, RTX 4090）

| 指标 | Phase 3 | Phase 6 |
|------|---------|---------|
| Throughput | 361.3 tok/s（88.4%）| **406.3 tok/s（100.0%）** |
| gather_batch_kv | 0.31ms/step | 0（消除） |
| write_decode_kv | 0.19ms/step | 0（消除） |
| model_forward | 17.89ms/step | 17.06ms/step |
