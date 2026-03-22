# Phase 12.5 知识笔记：Flash Decoding（Split-K Attention）

日期：2026-03-22

---

## Flash Decoding 是什么

Flash Decoding 是 Flash Attention 的 decode 阶段扩展，核心思路：把 KV 序列在 seq_len 方向切分成多段（split-K），每段由独立的 SM 并行处理，最后归约合并。

Grid 从 `(batch, num_heads)` 扩展为 `(batch, num_heads, num_splits)`，SM 利用率从受限于 batch×heads，变为可以通过 num_splits 调节。

---

## 为什么需要 split-K

标准 decode attention：batch=1、num_q_heads=12 时 grid=12 个 thread block，RTX 4090 有 128 SM，利用率约 9%。seq_len 越长，每个 thread block 处理越多 KV，latency 线性增长，但多余的 116 个 SM 全程空转。

split-K 之后：12 heads × 11 splits = 132 个 thread block，充满 128 SM。每个 split 处理 seq_len/11 ≈ 373 tokens（for seq_len=4096），并行结束后归约。

---

## 两阶段数学

**阶段一 per-split online softmax：**

```
对 KV[split_start : split_end] 运行 online softmax
输出：
  partial_out_s = acc / l_s     （V 加权均值，float16）
  partial_lse_s = m_s + log(l_s) （log-sum-exp，float32）
```

**阶段二归约：**

```python
weight_s = exp(lse_s - max(lse)) / sum(exp(lse_s - max(lse)))
out = sum_s(weight_s * partial_out_s)
```

正确性证明：`partial_out_s * exp(lse_s)` = 对应 split 的无归一化加权 V 之和，再除以全局 exp(lse) 之和就是完整 attention。

---

## 关键 Triton API 注意事项

- `tl.ones` 不存在（Triton 2.1.0），用 `tl.full([1], 1.0, dtype=tl.float32)`
- 标量 pointer 不能 store block tensor：`tl.store(ptr + tl.arange(0, 1), val)` 把两边统一成 `[1]` block
- `range(runtime_start, runtime_end, CONSTEXPR_step)` 在 Triton 2.1.0 中可行（start/end 可运行时，step 要 constexpr）

---

## 性能特征（RTX 4090 实测）

| seq_len | triton_65_ms | flash_decode_ms | 加速 |
|---------|-------------|----------------|------|
| 128 | 0.021 | 0.064 | 0.32× |
| 1024 | 0.058 | 0.064 | 0.91× |
| 2048 | 0.112 | 0.070 | 1.60× |
| 4096 | 0.224 | 0.068 | 3.31× |

盈亏平衡点（1.5B）：约 seq_len=1200。短序列 split-K 开销大于并行收益，不适用。
flash_decode 对长序列的延迟几乎恒定（128→4096 仅 +6%），这是核心收益。

vs flash_attn：始终约慢 5-7×（实现层级差距，flash_attn 有 SMEM tiling，Triton 实现走 L2 cache）。
