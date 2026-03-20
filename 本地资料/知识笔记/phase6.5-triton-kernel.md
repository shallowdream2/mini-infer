# Phase 6.5 知识笔记：Triton Kernel 开发

## Triton 编程模型核心概念

**@triton.jit**：把 Python 函数编译为 GPU kernel。函数里只能用 Triton 提供的操作（`tl.*`），不能用普通 Python。

**tl.constexpr**：编译期常量。Tiling 大小（BLOCK_N、HEAD_DIM）必须是 constexpr，因为编译器需要在编译时确定寄存器分配和 SRAM 用量。

**Grid**：决定启动多少个并发 program。`grid = (batch, num_q_heads)` 意味着为每个 (batch_idx, q_head_idx) 对启动一个独立的 GPU program。

**tl.program_id(axis)**：每个 program 的唯一 ID。用来确定"我是哪个 batch，哪个 head"。

**tl.load / tl.store**：向量化内存访问。传入 pointer 数组（形状 = block size），一次性加载 BLOCK 个元素。比逐元素 CUDA 访问高效。

**tl.arange(0, BLOCK)**：生成 0..BLOCK-1 的整数向量，用于构造 pointer 偏移。

**mask 参数**：`tl.load(ptrs, mask=mask, other=0.0)` 对 mask=False 的位置不实际加载，返回 other 值。处理序列末尾的 padding 必须用 mask。

---

## Online Softmax（Milakov & Gimelshein 2018）

**为什么需要 online softmax**：标准 softmax 需要两遍：先找 max，再算 exp/sum。对长序列，第一遍的所有 scores 放不进 SRAM，只能存 HBM，效率低。

**Online softmax 思路**：维护运行状态 (m, l, acc)，每次处理一个 block 就更新，不需要把所有 scores 保存：

```
初始化：m = -inf，l = 0，acc = 0

对每个 K/V block：
  scores_block = Q · K_block / scale
  m_new = max(m, max(scores_block))

  # 修正之前的 accumulator（因为 max 变了）
  l_new = l * exp(m - m_new) + sum(exp(scores_block - m_new))
  acc = acc * exp(m - m_new) + sum(exp(scores_block - m_new)[:, None] * V_block, axis=0)

  m = m_new，l = l_new

最终：acc / l
```

**关键性质**：每个 block 只需要 O(head_dim) 的 SRAM（存 acc 向量），与 seq_len 无关。

---

## Triton Encoding 不匹配陷阱

Triton 的 MLIR 层区分 blocked tensor 和 scalar：
- `tl.full([1], -inf)`：blocked tensor，有 layout encoding
- `tl.max(tensor, axis=0)`：scalar reduce，无 layout encoding
- `tl.maximum(blocked, scalar)`：encoding 不匹配 → 编译报错

**规则**：要对 1D scores 做 max/sum 并与 `[1]` 状态变量运算，需要先升维：
```python
block_max = tl.max(scores[None, :], axis=1)   # [1, BLOCK_N] → [1]，encoding 一致
```

---

## Decode Attention 是 Memory-Bound 操作

- Query length = 1，seq_len 可以很长
- 主要开销：从 HBM 读取 seq_len × num_kv_heads × head_dim 的 K/V 数据
- 算术强度 AI ≈ 7 FLOPs/Byte（RTX 4090 ridge point = 82 FLOPs/Byte）
- 结论：性能瓶颈是内存带宽，不是算力

**优化方向不是减少 FLOPs，而是**：
- 向量化 load（float4 代替 float16 逐元素）
- Prefetch pipeline（下一个 block 的 load 与当前 block 的计算重叠）
- GQA cache line 复用（多个 Q head 共享同一 K/V 加载）

---

## 与 flash_attn 的差距根因（首版 Triton）

| 差距点 | 首版 Triton | flash_attn |
|--------|-----------|-----------|
| K/V load 粒度 | float16 逐元素（stride 访问） | float4 向量化（128-bit 对齐） |
| K/V pipeline | 串行（load 后再 compute） | 软件 prefetch（重叠） |
| GQA 优化 | 每个 Q head 独立加载 K/V | 多 Q heads 共享 K/V load |
| seq_len=128 差距 | 1.23× | — |
| seq_len=2048 差距 | 9.46× | — |

差距随 seq_len 增大，因为 seq_len 越大，K/V 读取量越多，内存访问效率差距放大。

---

## GQA 映射

Qwen2.5-7B：28 Q heads / 4 KV heads，每 7 个 Q heads 共享 1 个 KV head：
```python
kv_head_idx = q_head_idx * num_kv_heads // num_q_heads
# q_head_idx ∈ [0,28) → kv_head_idx ∈ [0,4)
```

---

## @triton.jit 不能用 python -c 测试

`@triton.jit` 调用 `inspect.getsource()` 读取 kernel 源代码，`python -c "..."` 没有源文件。
→ 测试 Triton kernel 必须写到 `.py` 文件里运行。
