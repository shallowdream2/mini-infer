# 知识专题：Triton Decode Attention Kernel 设计与实现

## 主题

用 Triton 实现 decode-only（query length=1）attention kernel，覆盖 GQA、online softmax、tiling 设计，以及与 flash_attn 的性能差距分析。

---

## 一、问题定义

### Decode Attention 的特殊性

Autoregressive 生成的 decode 阶段，每步只产生一个新 token。注意力计算的输入：
- Q：`(batch, 1, num_q_heads, head_dim)` — 当前 token 的 query，query length = 1
- K/V：`(batch, seq_len, num_kv_heads, head_dim)` — 历史 KV cache

对每个 (batch, q_head) 对，计算：
```
output = softmax(Q · K^T / sqrt(head_dim)) · V
```

其中 K/V 的 head 数量可以少于 Q（GQA），seq_len 随生成步数增加。

### 为什么要用 Triton 而不是直接调 flash_attn

flash_attn 是生产级黑盒，了解它的性能特征需要自己实现一个对照。自己写 kernel 能回答：
1. "你对 flash_attn 的理解是什么？" — 不只是 API 调用，能讲清楚 online softmax 和 tiling
2. "这个操作是 compute-bound 还是 memory-bound？" — 从 roofline 分析而非直觉回答
3. "与 flash_attn 的差距在哪里？" — 能量化且有根因

---

## 二、核心原理

### Online Softmax（Milakov & Gimelshein, 2018）

标准 softmax 需要两次遍历全部 scores：第一次找 max，第二次算 exp/sum。这意味着所有 scores 要存在显存里（O(seq_len) 额外空间），或者读两遍 K/V（两倍带宽）。

Online softmax 用三个运行状态 (m, l, acc) 在一次遍历中完成：

```
初始状态：m = -∞，l = 0，acc = 0（shape = head_dim）

处理第 i 个 K/V block（大小 BLOCK_N）：
  scores = Q · K_block^T / scale          # (BLOCK_N,)
  m_new = max(m, max(scores))

  # 关键：修正历史 accumulator
  rescale = exp(m - m_new)
  acc = acc * rescale                      # 历史 V 的贡献按新 max 缩放
  l   = l * rescale + sum(exp(scores - m_new))

  # 累积当前 block
  exp_s = exp(scores - m_new)             # (BLOCK_N,)
  acc  += sum(exp_s[:, None] * V_block, axis=0)   # (head_dim,)
  m = m_new

最终：acc / l                              # (head_dim,)
```

**数学正确性**：每次 block 处理后，`acc / l` 恒等于前 i 个 block 的完整 softmax attention 输出。最终结果与一次性计算完全等价。

**SRAM 要求**：只需存 Q（head_dim）、当前 K/V block（BLOCK_N × head_dim × 2）、acc（head_dim）和状态（m, l），与 seq_len 无关。

### Roofline 分析

以 RTX 4090，Qwen2.5-7B 参数（28Q/4KV heads，head_dim=128，batch=8, seq_len=128）为例：

| 指标 | 计算 | 数值 |
|------|------|------|
| K/V 读取量 | batch × seq_len × num_kv_heads × head_dim × 2 × 2 bytes | ~2.1 MB |
| Q/Out 读写 | batch × 1 × num_q_heads × head_dim × 2 × 2 bytes | ~0.1 MB |
| 总内存访问 | K/V + Q + Out | ~2.2 MB |
| FLOPs | batch × num_q_heads × (2×seq_len×head_dim + 3×seq_len + 2×seq_len×head_dim) | ~14.8 MFLOPs |
| 算术强度 | FLOPs / bytes | ~7 FLOPs/Byte |
| RTX 4090 ridge point | ~82 TFLOPS / ~1008 GB/s | ~82 FLOPs/Byte |

AI = 7 << 82，**decode attention 是 memory-bound**。

**推论**：优化 FLOPs 对性能几乎没帮助；提升带宽利用率才是关键（向量化 load、减少 cache miss、prefetch pipeline）。

---

## 三、工程实现方式

### Grid 设计

```python
grid = (batch_size, num_q_heads)
```

每个 program = 一个 (batch_idx, q_head_idx) 对，完整处理整个 seq_len 的 KV，输出一个 head_dim 长度的向量。

独立 program 无需同步，是最简单的并行分解。代价是 GQA 下多个 Q heads 的 K/V 重复加载（每组 7 个 Q heads 会各自独立加载同一 K/V head）。

### Stride 传递

Kernel 不接受 tensor 对象，只接受裸指针和 stride。以 Q（shape `(batch, 1, num_q_heads, head_dim)`）为例：

```python
# Python 端
_decode_attn_kernel[grid](
    q, ...
    q.stride(0),  # stride_qb：batch 维
    q.stride(2),  # stride_qh：head 维（跳过 seq 维，decode 永远取 pos=0）
    q.stride(3),  # stride_qd：dim 维
    ...
)

# Kernel 端
q_ptr = Q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh
q = tl.load(q_ptr + tl.arange(0, HEAD_DIM) * stride_qd)
```

### 关键 Triton 语法

```python
# 构造 2D 指针矩阵（BLOCK_N 行 × HEAD_DIM 列）
k_ptrs = k_base + block_range[:, None] * stride_kn + d_range[None, :] * stride_kd

# 带 mask 的 load（处理最后一个 partial block）
k_block = tl.load(k_ptrs, mask=mask[:, None], other=0.0)

# 带 mask 的点积
scores = tl.sum(q[None, :] * k_block, axis=1)   # (BLOCK_N,)
scores = tl.where(mask, scores, -1e38)
```

### Encoding 不匹配的修复

```python
# ❌ 错误：tl.max(scores, axis=0) 返回 scalar，与 m_i:[1] encoding 不兼容
block_max = tl.max(scores, axis=0)
m_new = tl.maximum(m_i, block_max)   # 编译报错

# ✅ 正确：升维后 reduce，保持 [1] blocked encoding
block_max = tl.max(scores[None, :], axis=1)   # scores: (BLOCK_N,) → [1,BLOCK_N] → [1]
m_new = tl.maximum(m_i, block_max)           # [1] vs [1]，encoding 一致
```

---

## 四、设计取舍

### BLOCK_N = 64 vs 更大/更小

| BLOCK_N | 单个 K/V block 大小 | 优点 | 缺点 |
|---------|-----------------|------|------|
| 32 | 8 KB K + 8 KB V = 16 KB | SRAM 压力小 | 循环次数更多，overhead 更大 |
| **64** | **16 KB K + 16 KB V = 32 KB** | **cache line 对齐（128B），循环适中** | — |
| 128 | 32 KB K + 32 KB V = 64 KB | 循环次数少 | 占 SM L1 cache 的 1/2，可能影响 occupancy |

选 64 是一个平衡点：符合 128-byte cache line 对齐（64 elements × 2 bytes = 128 bytes），SM 内可同时容纳多个 block。

### Dense KV vs Paged KV

本实现使用 dense `(batch, seq_len, num_kv_heads, head_dim)` KV tensor，不支持 block_table。

- **选择 dense 的原因**：实验性对比，不替换 flash_attn，接口简单
- **Paged KV 的实现方式**：需要传入 `block_table: (batch, max_blocks)` 和 `cache_seqlens: (batch,)`，kernel 内根据 `block_table[batch_idx, block_idx]` 计算物理 K/V 地址。这是 flash_attn_with_kvcache 支持的接口，也是 vLLM 的生产路径。

### float32 内部计算 vs float16 全程

Kernel 接受 float16 输入，内部用 float32 做 softmax 和累积（`tl.load(...).to(tl.float32)`），输出转回 float16。

全程 float16 节省寄存器但有数值精度风险（exp 的上溢下溢）。float32 内部计算多约 2× 寄存器，但精度有保障。实测 max_diff ≤ 1.5e-5，远低于 1e-2 阈值，方案可行。

---

## 五、常见误区

**误区 1："decode attention 是 compute-bound，优化 FLOPs 有用"**
→ AI ≈ 7 FLOPs/Byte，远低于 ridge point（82），是 memory-bound。减少 FLOPs 对延迟几乎没帮助，提升带宽利用率才有效。

**误区 2："Triton 首版能接近 flash_attn 性能"**
→ seq_len=128 时差距 1.23× 看起来不错，但 seq_len=2048 时差距达 9.46×。差距随 KV 读取量增大线性扩大，因为访存效率的差距在长序列下被放大。

**误区 3："online softmax 和普通 softmax 结果不一样"**
→ 数学上完全等价，实测 max_diff ≤ 1.5e-5。差异来自 float16 精度，不是算法。

**误区 4："mask 的 other=0.0 会污染 scores"**
→ K 加载时 masked 位置填 0，scores 计算后再用 `tl.where(mask, scores, -1e38)` 覆盖为 -∞，softmax 后对应 exp 趋近于 0，不影响结果。

**误区 5：用 `python -c "..."` 测试 @triton.jit kernel**
→ `inspect.getsource()` 无法读取 inline 代码，会报 `OSError: could not get source code`。必须写 `.py` 文件。

---

## 六、和 mini-infer 的关系

Phase 6.5 新增文件：

| 文件 | 作用 |
|------|------|
| `mini_infer/triton_attn.py` | Triton decode attention kernel（实验性，不替换 attention.py） |
| `tests/test_triton_attn.py` | 数值正确性测试（8个，含 dry-run） |
| `benchmarks/benchmark_triton.py` | 与 flash_attn 的 latency 对比 benchmark |

`triton_attn.py` 与 `attention.py`（Phase 6 的 flash_attn 集成）**解耦**：前者是实验和学习路径，后者是 LLMEngine 的生产路径。

---

## 七、进一步阅读

- Milakov & Gimelshein (2018), "Online normalizer calculation for softmax" — online softmax 原始论文
- Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" — flash attention 的 tiling + online softmax 工程实现
- Triton 官方 Tutorial: "Fused Attention" — `https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html`
- vLLM `csrc/attention/attention_kernels.cuh` — 生产级 paged attention CUDA kernel，对比参考
- 实验数据：`本地资料/实验记录/2026-03-20-phase6.5-benchmark.md`
- 代码：`mini_infer/triton_attn.py`
