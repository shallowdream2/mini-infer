# 用 Triton 写一个 decode attention kernel：从零到能跑

> 系列文章第 7 篇。前 6 篇从零实现了 Paged KV Cache、Continuous Batching、向量化 gather、双卡扩展、Profiling 分析，最终在 Phase 6 用 flash_attn block_tables 达到了 100% HF baseline。Phase 6.5 做一件事：自己写一个 decode attention kernel。

---

## 一、为什么要自己写 kernel

Phase 6 结束后，mini-infer 的 decode 路径已经没有明显的 Python 开销。decode 的每一步耗时分布约为：

- `model_forward`：~17.9ms（含 28 层 flash_attn_with_kvcache）
- `gather_kv / write_kv`：~0（Phase 6 消除）
- Python 调度：< 0.5ms

继续提升性能意味着进入 GPU kernel 层。但 flash_attn 是一个高度优化的 C++/CUDA 黑盒。要真正理解性能瓶颈在哪里，唯一的方法是自己实现一个功能等价的 kernel，做直接对比，用数据说话。

这不是为了替换 flash_attn，而是为了学习：
- Triton 编程模型的实际使用
- Decode attention 的计算特征（memory-bound 还是 compute-bound）
- 与高度优化实现的差距来自哪里

---

## 二、问题定义：decode attention 的特点

Decode 阶段，每一步只有一个新 token（query length = 1），需要 attend 到所有之前的 KV（seq_len 个 token）。

输入：
- Q：`(batch, 1, num_q_heads, head_dim)` — 当前 token 的 query
- K/V：`(batch, seq_len, num_kv_heads, head_dim)` — 历史 KV cache（dense tensor）

输出：
- `(batch, 1, num_q_heads, head_dim)` — attention 结果

计算：对每个 (batch_idx, q_head_idx) 对，计算 `softmax(Q · K^T / scale) · V`，其中 K/V 可能对应多个 Q heads（GQA）。

### Roofline 分析：这是一个 memory-bound 问题

在 RTX 4090 上（以 Qwen2.5-7B 配置为例，batch=8, seq_len=128）：

| 指标 | 数值 |
|------|------|
| 读写量（K/V + Q/Out） | ~2.1 MB |
| FLOPs（QK matmul + softmax + AV） | ~14.8 MFLOPs |
| 算术强度（AI） | ~7 FLOPs/Byte |
| RTX 4090 ridge point | ~82 FLOPs/Byte |

AI = 7 << 82，**decode attention 是典型的 memory-bound 操作**。优化目标不是减少 FLOPs，而是最大化内存带宽利用率。

这个结论决定了 kernel 的设计方向：tile size 的选择要让 K/V 数据尽量在 L1/L2 cache 中复用，而不是追求 FLOPs 密度。

---

## 三、方案设计

### Grid 设计

每个 Triton program 处理一个 `(batch_idx, q_head_idx)` 对：

```python
grid = (batch, num_q_heads)
```

对 Qwen2.5-7B（28 Q heads）、batch=8，grid = (8, 28) = 224 个并发 program。每个 program 负责：
1. 加载这一 Q head 的 query 向量（128 维）
2. 分块迭代整个 KV 序列
3. 用 online softmax 累积结果

### GQA 映射

Qwen2.5-7B 使用 GQA（Grouped Query Attention）：28 个 Q heads 共享 4 个 KV heads，每 7 个 Q heads 对应 1 个 KV head。

```python
kv_head_idx = q_head_idx * num_kv_heads // num_q_heads
```

这行代码把 q_head_idx ∈ [0, 28) 映射到 kv_head_idx ∈ [0, 4)。

### Online Softmax（Milakov & Gimelshein, 2018）

Softmax 不能两步完成（先算 max，再算 exp），因为那需要把所有 scores 存在 SRAM 里。当 seq_len 很长时放不下。

Online softmax 的核心思路：维护运行状态 (m_i, l_i, acc)，每处理一个 K/V block 就更新：

```
m_new = max(m_i, max(scores_block))
l_new = l_i * exp(m_i - m_new) + sum(exp(scores_block - m_new))
acc   = acc * exp(m_i - m_new) + exp(scores_block - m_new) · V_block
```

最终 `acc / l_i` 就是归一化结果。只需要 O(BLOCK_N × head_dim) 的 SRAM，与 seq_len 无关。

### Tile Size 选择

**BLOCK_N=64**，**HEAD_DIM=128**（constexpr，编译期固定）。

选 BLOCK_N=64 的依据：
- 每个 K/V block 的内存：64 × 128 × 2 bytes（float16）= 16 KB
- 两个 block（K + V）共 32 KB，对应 RTX 4090 每 SM 约 128 KB L1 shared memory 的 1/4
- 保留足够空间给 Q 向量（128 × 4 bytes = 0.5 KB）和累积器 acc（128 × 4 bytes = 0.5 KB）
- 64 元素 × 2 bytes = 128 bytes，恰好对应 GPU 的 128-byte cache line，利于内存访问对齐

---

## 四、实现细节

### Kernel 框架

```python
@triton.jit
def _decode_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qd,   # Q: (batch, 1, num_q_heads, head_dim)
    stride_kb, stride_kn, stride_kh, stride_kd,  # K: (batch, seq_len, num_kv_heads, head_dim)
    stride_vb, stride_vn, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    seq_len, num_kv_heads, num_q_heads, scale,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_idx  = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx * num_kv_heads // num_q_heads  # GQA

    d_range = tl.arange(0, HEAD_DIM)
    q_ptr = Q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh
    q = tl.load(q_ptr + d_range * stride_qd).to(tl.float32)

    m_i = tl.full([1], -1e38, dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    k_base = K_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + kv_head_idx * stride_vh

    for block_start in range(0, seq_len, BLOCK_N):
        block_range = block_start + tl.arange(0, BLOCK_N)
        mask = block_range < seq_len

        k_ptrs  = k_base + block_range[:, None] * stride_kn + d_range[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(mask, scores, -1e38)

        block_max  = tl.max(scores[None, :], axis=1)   # [1]
        m_new      = tl.maximum(m_i, block_max)
        exp_scores = tl.exp(scores - m_new)
        l_new      = l_i * tl.exp(m_i - m_new) + tl.sum(exp_scores[None, :], axis=1)

        acc = acc * tl.exp(m_i - m_new)

        v_ptrs  = v_base + block_range[:, None] * stride_vn + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        acc     = acc + tl.sum(exp_scores[:, None] * v_block, axis=0)

        m_i, l_i = m_new, l_new

    out_ptr = Out_ptr + batch_idx * stride_ob + q_head_idx * stride_oh
    tl.store(out_ptr + d_range * stride_od, (acc / l_i).to(tl.float16))
```

### Stride 传递

Q 的 shape 是 `(batch, 1, num_q_heads, head_dim)`，kernel 访问 `Q[batch_idx, 0, q_head_idx, :]`：

```python
q_ptr = Q_ptr + batch_idx * Q.stride(0) + q_head_idx * Q.stride(2)
q = tl.load(q_ptr + d_range * Q.stride(3))
```

省略了 seq 维度（`stride(1)`），因为 decode 永远只取位置 0。

K/V 类似：`k_base = K_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh`，然后在 seq 维度上分块。

---

## 五、踩坑：Triton encoding 不匹配

这是实现过程中最不直觉的一个问题，值得单独说清楚。

### 问题现象

```python
m_i = tl.full([1], -1e38, dtype=tl.float32)
# ...
block_max = tl.max(scores, axis=0)           # 返回 scalar
m_new     = tl.maximum(m_i, block_max)       # 编译报错！
```

报错信息：
```
RuntimeError: PassManager::run failed
...
'triton_gpu.cmpf' op requires the same encoding for all operands and results
```

### 根因

Triton 在 MLIR 层面区分两种类型：
- **blocked tensor**：`tl.full([1], ...)` 创建的 shape `[1]` tensor，有 blocked layout encoding
- **scalar**：`tl.max(tensor, axis=0)` 规约到 0 维后产生的标量，没有 layout encoding

`tl.maximum(blocked_tensor, scalar)` 要求两边 encoding 一致，因此编译失败。

### 修复

```python
# ❌ 错误：tl.max 返回 scalar，与 m_i 的 blocked encoding 不兼容
block_max = tl.max(scores, axis=0)
m_new = tl.maximum(m_i, block_max)

# ✅ 修复：先升维为 [1, BLOCK_N]，再沿 axis=1 规约，结果保持 [1] blocked encoding
block_max = tl.max(scores[None, :], axis=1)   # [1]，encoding 与 m_i 一致
m_new = tl.maximum(m_i, block_max)            # ✅ 编译通过
```

同理，`tl.sum(exp_scores, axis=0)` 也要改成 `tl.sum(exp_scores[None, :], axis=1)` 才能与 `l_i: [1]` 相加。

这个规则不在 Triton 文档的显眼位置，需要从编译器报错里倒推。核心结论：**在 Triton kernel 中，凡是需要对 1D tensor 做 reduce 并把结果与另一个 `[1]` tensor 做运算，必须保持 encoding 一致，用 `[None, :]` 升维后再 reduce。**

### 另一个坑：`@triton.jit` 与 `python -c` 不兼容

在前置条件验证阶段，想用 `python -c "..."` 跑一个最小 Triton kernel 测试。结果报错：

```
OSError: could not get source code
```

原因：`@triton.jit` 内部调用 `inspect.getsource()` 来获取 kernel 源代码，但 `python -c "..."` 没有对应的源文件，`inspect` 无法读取。

解决方案：把测试代码写到临时文件后用 `subprocess` 执行，或者直接写到 `.py` 文件里运行。**不能用 `python -c` 测试 Triton kernel。**

---

## 六、验证：数值正确性

写完 kernel 不能直接上 benchmark，先验证数值正确性。对比三种实现：

1. `triton_decode_attention`：本文实现的 Triton kernel
2. `reference_decode_attention`：PyTorch float32 参考实现（ground truth）
3. `flash_decode_attention`：`flash_attn_with_kvcache`

验证配置：Qwen2.5-7B 参数（28Q/4KV heads，head_dim=128），不同 batch/seq_len。

```
tests/test_triton_attn.py::test_vs_reference_basic   max_diff = 4e-6   PASSED
tests/test_triton_attn.py::test_vs_flash             max_diff = 1.5e-5 PASSED
tests/test_triton_attn.py::test_gqa_qwen_config      max_diff = 0      PASSED
tests/test_triton_attn.py::test_batch8               max_diff < 1e-4   PASSED
tests/test_triton_attn.py::test_long_seq             max_diff = 0      PASSED  # seq_len=2048
tests/test_triton_attn.py::test_seq_not_multiple_of_block max_diff = 0 PASSED  # seq_len=100
```

max_diff 最高 1.5e-5，远低于 1e-2 阈值（fp16 精度差异的合理上界）。数值正确。

---

## 七、性能实验

### 实验环境

- GPU：NVIDIA GeForce RTX 4090
- PyTorch：2.1.2+cu121，Triton：2.1.0，flash_attn：2.5.9.post1
- 测量方式：`triton.testing.do_bench(fn, warmup=25, rep=100)`，返回 median 延迟

### batch=1 延迟对比

| seq_len | triton (μs) | flash (μs) | 比值 | 理论 BW bound (μs) |
|---------|------------|-----------|------|------------------|
| 128 | 14.30 | 11.66 | 1.23× | 0.27 |
| 512 | 45.58 | 11.87 | 3.84× | 1.05 |
| 1024 | 85.25 | 13.81 | 6.17× | 2.09 |
| 2048 | 168.26 | 17.78 | 9.46× | 4.18 |

### batch=8 延迟对比

| seq_len | triton (μs) | flash (μs) | 比值 | 理论 BW bound (μs) |
|---------|------------|-----------|------|------------------|
| 128 | 20.83 | 12.54 | 1.66× | 2.19 |
| 512 | 59.36 | 22.47 | 2.64× | 8.44 |
| 1024 | 100.74 | 35.19 | 2.86× | 16.76 |
| 2048 | 182.84 | 60.01 | 3.05× | 33.40 |

### 解读

**seq_len=128，batch=1 时差距只有 1.23×。** 这说明 kernel 本身的基础正确性没问题，在 KV 量少时两者接近。

**seq_len 增大后差距扩大到 9.46×（batch=1, seq_len=2048）。** 这里差距来自内存访问效率，而不是算法差异。理论 BW bound 是 4.18 μs，flash_attn 实际是 17.78 μs（4.25× 理论值），Triton 是 168 μs（40× 理论值）。两者都远未达到硬件带宽上限，但 flash_attn 更接近。

**batch=8 时差距缩小到 2.6×–3.1×。** batch 增大后 GPU 并行度更高，Triton 的 kernel launch overhead 被摊薄，部分内存访问延迟也被隐藏。

---

## 八、差距根因分析

Triton 首版比 flash_attn 慢的根本原因不是算法问题，是工程优化层面的差距：

### 1. 无向量化 global memory load

Triton 首版用 stride 方式逐元素加载 K/V：

```python
k_ptrs = k_base + block_range[:, None] * stride_kn + d_range[None, :] * stride_kd
k_block = tl.load(k_ptrs, ...)
```

每次 load 的粒度取决于 `stride_kd`（通常为 1，即逐 float16 元素）。flash_attn 使用 `float4` 或 `float2` 向量化 load，单次取 128-bit，是我们的 4×/8× 带宽效率。

### 2. 无 K/V 预取流水线

Triton 首版在循环中串行执行：load K → compute QK → load V → compute AV。

```
load K block → compute scores → load V block → accumulate
(等待)          (计算)          (等待)          (计算)
```

flash_attn 通过软件流水线（prefetch）将下一个 block 的 load 与当前 block 的计算重叠，内存延迟被隐藏在计算之后。

### 3. GQA 没有 cache line 复用优化

本文实现中，每个 (batch, q_head) program 独立加载对应的 KV head。28 个 Q heads 共享 4 个 KV heads，实际上会有 7 个 program 读取同一份 K/V 数据。

flash_attn 对 GQA 有专门的 kernel，多个 Q head 共享 K/V 的加载，cache line 复用率更高。

### 结论

这些都是**可以优化的工程问题**，不是方案本身的问题。如果继续推进：

1. 用 `tl.make_block_ptr` + `tl.load(..., eviction_policy="evict_first")` 改善向量化
2. 在循环内手动 prefetch 下一个 block 的 K/V 指针
3. 合并共享同一 KV head 的多个 Q head，减少重复加载

但这超出了本阶段"走通实现路径"的目标范围。

---

## 九、总结

这篇文章记录了从零实现一个 Triton decode attention kernel 的完整过程。

**做到了什么：**
- 一个 220 行的 `triton_attn.py`，包含 kernel、PyTorch wrapper 和 flash_attn 对照组
- 8 个测试全部通过，max_diff ≤ 1.5e-5，满足 fp16 精度要求
- 在 seq_len=128 时与 flash_attn 差距仅 1.23×，说明基础实现路径是正确的
- 用 roofline 分析定量说明了 decode attention 的 memory-bound 特性（AI=7 FLOPs/Byte）

**学到了什么：**
- Triton 的 blocked encoding 约束：reduce 操作必须保持 encoding 一致，否则编译报错
- Online softmax 的 tile 化实现：比概念理解多了很多工程细节
- "memory-bound" 不是一个空话，它意味着优化方向是内存访问效率，而非 FLOPs 减少
- 与 flash_attn 的差距完全来自工程优化层面（向量化 load、prefetch pipeline、GQA 专项优化），算法本身没有差距

**还差什么：**
- 向量化 float4 load（最关键的优化点）
- K/V prefetch 流水线
- GQA 的 K/V 共享加载
- Paged KV（block_table 路径），本文实现仅针对 dense KV tensor

从工程角度，flash_attn 的领先来自多年积累的精细优化，而不是算法上的神秘性。理解了这一点，Triton kernel 开发就有了清晰的优化路径，而不是面对黑盒束手无策。

---

*环境：Ubuntu 24.04，RTX 4090，PyTorch 2.1.2+cu121，Triton 2.1.0，flash_attn 2.5.9.post1*
*代码：[mini-infer/mini_infer/triton_attn.py](https://github.com/psmarter/mini-infer)*
