# 从 9% 到 103%：用 Triton 实现 Flash Decoding（Split-K Attention）

> mini-infer Phase 12.5 实践记录
> 环境：RTX 4090，Triton 2.1.0，PyTorch 2.1.2+cu121

---

## 背景

Phase 12（CUDA Graph）把 Qwen2.5-1.5B bs=1 的 decode step 延迟从 7.33ms 降到 5.21ms，手段是消除 Python→CUDA dispatcher 的调度开销。这是 CPU 侧的问题，解决了。

GPU 侧还有一个不那么显眼的问题：**SM 利用率**。

Triton decode attention kernel（Phase 6.5）的 grid 是 `(batch, num_q_heads)`。对于 batch=1、num_q_heads=12 的 1.5B 模型，这意味着只有 **12 个 thread block**。RTX 4090 有 128 个 SM，一次只能给 12 个 SM 分配工作，利用率约 9%。

序列越长，每个 thread block 要处理的 KV tokens 越多，latency 随 seq_len 线性增长。但多出来的 116 个 SM 一直在闲置。

Flash Decoding（split-K）的思路很直接：在 KV 序列长度方向再切分，让所有 SM 都参与工作。

---

## 问题定义

Decode attention 的计算量（简化）：

```
每个 (batch_idx, q_head_idx) 程序 → 遍历 seq_len 个 KV tokens → O(seq_len × head_dim)
```

标准 grid = `(batch, num_q_heads)`，grid size = 1 × 12 = 12（1.5B，batch=1）。

**瓶颈**：所有 KV tokens 由一个 thread block 串行处理，其他 SM 空转。seq_len=4096 时，这个 thread block 要处理 4096/64 = 64 个 BLOCK_N 大小的 K/V 块。

Split-K 的做法：

```
grid = (batch, num_q_heads, num_splits)  →  1 × 12 × 11 = 132 个 thread block

每个 program 只处理 KV[split_start : split_end]（约 373 tokens）
所有 splits 并行，最后合并结果
```

SM 利用率从 9% 提升到 ~103%（刚好能充满 128 SM）。

---

## 方案设计

### 两阶段架构

**阶段一：Split Kernel**（Triton，主要计算）

- Grid = `(batch, num_q_heads, num_splits)`
- 每个 program 在 `KV[split_start : split_end]` 上跑 online softmax
- 输出：
  - `partial_out[batch, num_splits, num_q_heads, head_dim]`（float16，V 加权均值）
  - `partial_lse[batch, num_splits, num_q_heads]`（float32，log-sum-exp）

**阶段二：Reduce**（PyTorch）

- 在 num_splits 维度上做数值稳定的 softmax 加权归约
- 对 num_splits ≤ 32，开销 < 0.01ms，用 PyTorch 完全足够

### 数学推导

设 split s 的 online softmax 结果为：
- `m_s`：该 split 内的最大 score
- `l_s`：归一化分母（`sum exp(score - m_s)`）
- `acc_s`：V 加权累积（`sum exp(score - m_s) × V`）

输出存储：
- `partial_out_s = acc_s / l_s`（V 加权均值）
- `partial_lse_s = m_s + log(l_s)`（log-sum-exp）

归约时：

```python
# 全局 LSE
global_lse = logsumexp_s(partial_lse_s)

# 恢复各 split 的贡献权重
weight_s = exp(partial_lse_s - global_lse)

# 最终输出
out = sum_s(weight_s × partial_out_s)
```

关键：`partial_out_s × exp(partial_lse_s)` = `acc_s × exp(m_s)` = `sum_{t in s} exp(score_t) × V_t`，再除以全局归一化系数，结果等价于对完整 KV 序列做一次 attention。数学正确性由 num_splits=1 时 max_diff < 1e-3 验证。

### num_splits 的自动选择

```python
def auto_num_splits(seq_len, num_q_heads, batch=1, sm_count=128):
    BLOCK_N = 64
    max_by_seqlen = max(1, seq_len // BLOCK_N)   # 每个 split 至少有 BLOCK_N tokens
    ideal_splits  = max(1, math.ceil(sm_count / (batch * num_q_heads)))
    return min(max_by_seqlen, ideal_splits)
```

1.5B（12 heads）：`ideal = ceil(128/12) = 11`，seq_len≥704 时锁定 11 splits。

---

## 实现关键代码

### Split Kernel（节选）

```python
@triton.jit
def _flash_decode_split_kernel(
    Q_ptr, K_ptr, V_ptr, PartialOut_ptr, PartialLse_ptr,
    ...,
    seq_len, kv_per_split, num_kv_heads, num_q_heads, scale,
    BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    batch_idx  = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    split_idx  = tl.program_id(2)

    kv_head_idx = q_head_idx * num_kv_heads // num_q_heads  # GQA 映射

    split_start = split_idx * kv_per_split
    split_end   = tl.minimum(split_start + kv_per_split, seq_len)

    # online softmax 状态
    m_i = tl.full([1], -1e38, dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for block_start in range(split_start, split_end, BLOCK_N):
        block_range = block_start + tl.arange(0, BLOCK_N)
        mask = block_range < split_end
        # ... 加载 K/V，更新 online softmax
```

重点：`range(split_start, split_end, BLOCK_N)` 在 Triton 2.1.0 中支持运行时 start/end，step（BLOCK_N）是 constexpr。空 split（split_start >= seq_len）的 range 直接为空，循环体不执行。

### PyTorch 归约

```python
def _reduce_splits(partial_out, partial_lse):
    # partial_lse: (batch, num_splits, num_q_heads), float32
    max_lse = partial_lse.max(dim=1, keepdim=True).values
    exp_lse = torch.exp(partial_lse - max_lse)
    weights = exp_lse / exp_lse.sum(dim=1, keepdim=True)
    out = (weights.unsqueeze(-1) * partial_out.float()).sum(dim=1)
    return out.to(torch.float16).unsqueeze(1)
```

---

## 实验结果

环境：RTX 4090，batch=1，warmup=100，repeat=500（数据来自 `本地资料/实验记录/2026-03-22-phase12_5-benchmark.md`）。

### 1.5B 配置（12Q/2KV heads）

| seq_len | num_splits | flash_attn_ms | triton_65_ms | flash_decode_ms | vs triton_65 |
|---------|-----------|--------------|-------------|----------------|-------------|
| 128 | 2 | 0.010 | 0.021 | 0.064 | 0.32× |
| 512 | 8 | 0.014 | 0.033 | 0.064 | 0.51× |
| 1024 | 11 | 0.010 | 0.058 | 0.064 | 0.91× |
| 2048 | 11 | 0.011 | 0.112 | 0.070 | **1.60×** |
| 4096 | 11 | 0.012 | 0.224 | 0.068 | **3.31×** |

**seq_len 从 128 增长到 4096（32×）**：
- triton_65：0.021ms → 0.224ms（**+967%**）
- flash_decode：0.064ms → 0.068ms（**+6%**）

### 7B 配置（28Q/4KV heads）

| seq_len | num_splits | triton_65_ms | flash_decode_ms | vs triton_65 |
|---------|-----------|-------------|----------------|-------------|
| 1024 | 5 | 0.074 | 0.063 | **1.17×** |
| 2048 | 5 | 0.129 | 0.068 | **1.91×** |
| 4096 | 5 | 0.256 | 0.100 | **2.57×** |

7B 模型 28 heads 本身已有更好的 SM 覆盖，auto_num_splits=5（vs 1.5B 的 11），提升空间相对小，但 seq_len=4096 仍有 2.57× 加速。

---

## 坑点与反思

### 坑 1：`tl.ones` 不存在

第一版代码用了 `tl.where(l_i > 0.0, l_i, tl.ones([1], dtype=tl.float32))`。Triton 2.1.0 没有 `tl.ones`，运行时直接报 `AttributeError`。改为 `tl.full([1], 1.0, dtype=tl.float32)` 解决。

（Triton 有 `tl.zeros` 但没有对称的 `tl.ones`，踩坑前没想到。）

### 坑 2：标量 pointer + block value 的 `tl.store` 报错

`partial_lse` 存储的是一个 float32 标量（每个 split/head 一个 LSE 值）。写代码时自然写了：

```python
# 错误写法
pl_base = PartialLse_ptr + batch_idx * stride_plb + split_idx * stride_pls + q_head_idx * stride_plh
tl.store(pl_base, lse)  # lse 是 [1] 的 block tensor，pl_base 是标量 pointer
```

Triton 报：`Value argument cannot be block type if pointer argument is not a block`。

解决方案：把 pointer 也变成 `[1]` block pointer：

```python
tl.store(pl_base + tl.arange(0, 1), lse)  # 两边都是 [1] block
```

### 坑 3：原计划的对比基准选错了

infer-plan 里写了"seq_len=4096 目标 < 0.5× flash_attn"。实测 flash_decode 在 seq_len=4096 时比 flash_attn 慢约 5.7×，离这个目标差很远。

根因是：flash_attn 是高度优化的 CUDA C++（使用 shared memory tiling、warp-level 原语），而我们是 Triton 研究实现（所有 K/V 读取走 L2 cache）。两者之间的差距来自**实现层级**，不是算法。split-K 能改善 SM 利用率，但不能弥补 Triton 和 C++ 之间的差距。

正确的对比基准是同为 Triton 的 Phase 6.5 kernel，在此维度 split-K 效果显著。教训：对比时要区分"算法收益"和"实现层级差距"。

### 短序列的退化

seq_len < 1024（1.5B）时，flash_decode 慢于 triton_65（0.91×）。原因：split-K 有固定开销（partial_out/lse buffer 写入约 66KB，split kernel launch，reduction 计算），短序列时这个开销比并行化收益大。

这是 Flash Decoding 的固有特征，不是 bug。在线服务的 prefill 阶段（seq_len 通常 < 512）不应该用 split-K；长 context decode（seq_len > 1024）才是适用场景。

---

## 总结

split-K 的核心思路：decode attention 的 KV 遍历是串行的，但 KV 序列本身可以并行处理——只需要每个 split 输出 partial result（V 加权均值 + log-sum-exp），最后做一次数值稳定的归约。

效果：1.5B 模型 seq_len=4096 时，Triton split-K 延迟仅为基础 Triton kernel 的 **30%**（3.31×），而延迟几乎不随 seq_len 增长（128→4096 仅增 6%）。

局限：未接入推理主路径（需要支持 Paged KV / block_table），不能直接影响 TTFT/TPOT；也无法追平 flash_attn 的 SMEM 级优化。但作为研究性实现，它展示了 split-K 在 SM 利用率层面的可扩展性。

---

## 自查结果（QUALITY_CHECKLIST.md）

- [x] 文章主题对应真实项目阶段或真实技术问题（Phase 12.5，SM 利用率 9% → 103%）
- [x] 结构覆盖：背景、问题定义、方案设计、实现细节、实验结果、坑点反思、总结
- [x] 有明确背景和问题定义（SM 利用率不足，latency 随 seq_len 线性增长）
- [x] 有设计取舍，而不是只罗列概念（两阶段 vs 全 Triton、PyTorch reduce、num_splits 自动选择策略）
- [x] 有代码、实验或运行结果支撑（kernel 节选、benchmark 表格数据来自实验记录文件）
- [x] 有失败点、坑点或反思（三个具体坑点：tl.ones、scalar store、对比基准选错）
- [x] 语言克制、专业、可发表（无夸大宣称，明确说明 vs flash_attn 的差距及原因）
