# Tensor Parallelism 知识笔记

**来源**：Phase 13 实现 + Megatron-LM 论文

---

## 一句话理解

TP 把同一层的算力切分到多张卡。每张卡处理部分 attention heads 或 FFN neurons，forward 结束时 all-reduce 合并结果。

和 PP 的核心区别：PP 是层间串行（每步只有一张卡工作），TP 是层内并行（每步所有卡同时工作）。

---

## Column Parallel vs Row Parallel

```
线性层 y = x @ W.T 的 TP=2 分解：

Column Parallel（沿 dim=0 切分 W）：
  W = [W0; W1]（W0/W1 各一半行）
  rank 0: y0 = x @ W0.T
  rank 1: y1 = x @ W1.T
  → y = concat(y0, y1, dim=-1)，无通信

Row Parallel（沿 dim=1 切分 W）：
  W = [W0, W1]（W0/W1 各一半列）
  rank 0: y0 = x0 @ W0.T  （x0 是 x 的前半列，来自上一层 column parallel 输出）
  rank 1: y1 = x1 @ W1.T
  → y = y0 + y1，需要 all-reduce(SUM)
```

组合使用：column parallel 之后接 row parallel，只在 row parallel 后通信一次。

---

## Qwen2.5 的切分位置

| 层 | 类型 | 切分方式 |
|----|------|---------|
| q_proj, k_proj, v_proj | Column Parallel | dim=0，每卡一半 heads |
| o_proj | Row Parallel | dim=1，forward 后 all-reduce |
| gate_proj, up_proj | Column Parallel | dim=0，每卡一半 intermediate |
| down_proj | Row Parallel | dim=1，forward 后 all-reduce |

每层共 2 次 all-reduce（self_attn 后 + mlp 后），28 层共 56 次/decode step。

---

## 权重切分后必须更新的元数据

```python
attn.num_heads //= tp_size
attn.num_key_value_heads //= tp_size
# num_key_value_groups 不变（减小的比例相同）
attn.hidden_size = attn.num_heads * attn.head_dim  # ← 必须！
```

最后一行：Qwen2Attention.forward() 有 `attn_output.reshape(bsz, q_len, self.hidden_size)`，切分后 shape 变了，如果不更新 hidden_size 会 shape 不匹配。

---

## 通信模式

- **每次 all-reduce 的数据量**：`batch × seq_len × hidden_size × 2 bytes`（fp16）
  - 1.5B，bs=1，seq=1（decode）：1 × 1 × 1536 × 2 = 3 KB/次，56次 = 168 KB/step
- **通信是同步阻塞的**：forward hook 内 `dist.all_reduce(tensor, op=SUM)` 是阻塞调用，直到所有 rank 完成
- **synced_gpus=True**：generate 时必须加，防止不同 rank 在不同时间遇到 EOS 导致 collective hang

---

## 适用场景

| 场景 | TP 是否有帮助 |
|------|-------------|
| 单卡 OOM（模型 > 24 GB） | ✅ 必须用 |
| 大 batch prefill（compute-bound） | ✅ 有明显加速 |
| 小 batch decode（memory-bound） | ❌ 通信开销抵消甚至超过收益 |
| NVLink 互联 | ✅ 通信开销低，性价比高 |
| PCIe 互联（本环境） | ⚠ 通信瓶颈更明显 |

---

## 实测数据（1.5B，bs=3 decode）

- single：98.0 tok/s
- pp：82.4 tok/s（84%）
- tp=2（NCCL）：76.5 tok/s（78%）— 符合预期，小模型 decode TP 不适用

---

## 注意事项

1. **transformers 4.43.4 的 eager 模式有 attention mask bug**，TP 模式下必须用 `flash_attention_2`
2. **load-then-shard 的 VRAM peak 不准确**：先加载全量再替换分片，peak 含全量权重；shard-during-load 才能真正减半
3. **GQA 切分约束**：num_key_value_heads 必须能被 tp_size 整除，0.5B（2 KV heads）TP=2 刚好满足，但 TP=4 不满足
