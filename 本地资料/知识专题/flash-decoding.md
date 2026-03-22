# 知识专题：Flash Decoding（Split-K Attention）在 LLM 推理中的应用

---

## 一、问题定义

LLM 自回归 decode 的每个 step，attention 计算的特征是：query length = 1，KV cache length = seq_len（随生成步数增长）。

标准 Flash Attention 的 grid = `(batch, num_q_heads)`。decode 场景下 batch 通常为 1（在线服务的单请求），num_q_heads 由模型决定（如 1.5B = 12，7B = 28）。对于 RTX 4090（128 SM）：

```
1.5B，batch=1：grid = 12 个 thread block → SM 利用率 = 12/128 ≈ 9%
7B，batch=1：  grid = 28 个 thread block → SM 利用率 = 28/128 ≈ 22%
```

**结果**：绝大多数 SM 空转，attention latency 随 seq_len 线性增长，无法被并行化吸收。

这与 prefill 不同——prefill 的 query length = prompt_len，grid = `(batch, num_heads, seq_len/BLOCK_M)` 可以充满 SM。decode 的低 SM 利用率是独有问题。

---

## 二、核心原理

### Flash Decoding 的直觉

Attention 的计算可以分解为对 KV 序列的归约：

```
out = softmax(Q @ K^T / scale) @ V
    = (sum_t exp(score_t) * V_t) / (sum_t exp(score_t))
```

这个归约是**结合性**的：可以先对 KV 序列的子区间 `[a, b)` 计算 partial result，再把所有 partial results 合并。只要每个 partial result 携带足够的信息（partial sum + log-sum-exp），合并就是精确的。

### Online Softmax 与 Log-Sum-Exp

每个 split s 对 `KV[split_start : split_end]` 运行 online softmax，输出：

- `partial_out_s`：该区间的 V 加权均值 `= (sum_{t in s} exp(score_t) * V_t) / (sum_{t in s} exp(score_t))`
- `partial_lse_s`：`= m_s + log(sum_{t in s} exp(score_t - m_s))`，即区间内分母的 log

合并时：

```python
global_lse = logsumexp_s(partial_lse_s)        # 全局 log-sum-exp
w_s = exp(partial_lse_s - global_lse)          # 归一化权重，和为 1
out = sum_s(w_s * partial_out_s)               # 加权合并
```

这是数值稳定的精确合并，不是近似。

---

## 三、工程实现方式

### 两阶段架构

```
阶段一（split kernel）
  Grid: (batch, num_q_heads, num_splits)
  每个 program 对 KV[split_start : split_end] 做 online softmax
  输出：partial_out[batch, num_splits, num_q_heads, head_dim]（float16）
        partial_lse[batch, num_splits, num_q_heads]（float32）

阶段二（reduce）
  Grid: (batch, num_q_heads)  或  PyTorch element-wise ops
  归约 num_splits 维度
  输出：out[batch, 1, num_q_heads, head_dim]（float16）
```

### num_splits 选择

```python
def auto_num_splits(seq_len, num_q_heads, batch=1, sm_count=128):
    BLOCK_N = 64
    max_by_seqlen = max(1, seq_len // BLOCK_N)     # 每 split 至少 BLOCK_N tokens
    ideal_splits  = ceil(sm_count / (batch * num_q_heads))
    return min(max_by_seqlen, ideal_splits)
```

1.5B（12 heads）：ideal = ceil(128/12) = 11，覆盖 132 SM 槽位（刚好充满 128 SM）。

### Triton 2.1.0 实现注意点

- `range(split_start, split_end, BLOCK_N)`：start/end 可为运行时值，step 需 constexpr
- 空 split（split_start >= seq_len）：range 直接为空，循环体不执行，lse 输出 -1e38（归约时权重 ≈ 0）
- 标量 pointer 写 `[1]` tensor：`tl.store(ptr + tl.arange(0, 1), val)` 把两边统一为 `[1]` block

---

## 四、设计取舍

### 阶段二用 PyTorch 还是 Triton

| 方案 | 优点 | 缺点 |
|------|------|------|
| Triton reduce kernel | 纯 GPU，无 CPU→GPU 数据流 | 实现复杂（需 constexpr MAX_SPLITS），调试难 |
| PyTorch ops | 实现简单，3 行代码 | 有 CPU 调度开销 |

实测：partial_out buffer 约 66KB（1.5B，num_splits=11），PyTorch reduce 开销 < 0.01ms。相比 split kernel 的 0.06ms，可以忽略。**结论：用 PyTorch，可读性更高，性能无损失。**

### partial_out 存 float16 还是 float32

存 float16 可以减少 partial buffer 的显存占用和 DRAM 带宽。V 加权均值的数值范围与模型激活值相同（float16 可表示），精度损失可接受（实测 max_diff < 1e-2 vs float32 reference）。

### 短序列的处理策略

split-K 在短序列（< 512）下反而更慢，因为 partial buffer 写入 + reduce 开销 > 并行化收益。实践中应根据 seq_len 动态选择：

- seq_len < ~1000：使用标准 Flash Attention（Phase 6.5 kernel 或 flash_attn）
- seq_len > ~1000：使用 Flash Decoding（split-K）

或者更简单：直接用 flash_attn_with_kvcache（始终是最快的选项），split-K 的价值在于研究和理解。

---

## 五、常见误区

**误区 1：split-K 比 flash_attn 快**

我们的 Triton split-K 实现比 flash_attn_with_kvcache 慢 5-7×。flash_attn 使用 shared memory tiling（SRAM 带宽 ≫ L2/DRAM），warp-level 指令，精心调优的 pipeline。Triton 研究实现走 L2 cache，无 SMEM tiling。split-K 是对**同等实现层级**下的非 split-K 有优势，不是对高度优化 C++ 实现有优势。

**误区 2：split-K 对所有 batch size 有益**

batch=8 时，grid = 8 × 12 = 96 个 thread block，SM 利用率已达 75%，split-K 的额外并行化收益很小，反而引入 partial buffer 开销。split-K 主要针对 batch=1 的在线推理场景。

**误区 3：num_splits 越多越好**

num_splits 超过 `seq_len // BLOCK_N` 会产生空 split（split_start >= seq_len），这些 split 仍占用 SM 资源（kernel launch，partial buffer 写入）。`auto_num_splits` 的上界 `max_by_seqlen` 就是防止这种情况。

**误区 4：partial_lse 合并可以直接求和**

不能。`sum(partial_lse_s)` 没有意义。正确做法是 `logsumexp(partial_lse_s) = max(lse) + log(sum(exp(lse - max(lse))))`，再用 `exp(partial_lse_s - global_lse)` 作为归约权重。

---

## 六、和 mini-infer 的关系

Phase 12.5 实现见 `mini_infer/triton_flash_decode.py`：
- `flash_decode_triton()`：公开接口，支持 GQA，auto_num_splits，dense KV
- `auto_num_splits()`：基于 SM 数量和 head 数的自动配置
- benchmark 见 `benchmarks/benchmark_flash_decode.py`（seq_len sweep，vs flash_attn / triton_65）

实测（RTX 4090，1.5B，batch=1）：
- seq_len=2048：vs triton_65 加速 1.60×
- seq_len=4096：vs triton_65 加速 3.31×
- 延迟平坦性：128→4096（32×）增幅仅 +6%（vs triton_65 的 +967%）

Phase 12.5 的 kernel 未接入推理主路径（需要 block_table/Paged KV 支持），是独立实验模块。

---

## 七、进一步阅读

- Flash-Decoding 原始博文（Dao-AILab）：`https://crfm.stanford.edu/2023/10/12/flashdecoding.html`
- FlashAttention-2 论文（Tri Dao, 2023）——split-K 的归约数学
- vLLM 实现参考：`vllm/attention/backends/flash_attn.py`（生产级 Paged KV + split-K 融合版本）
- Triton 官方教程：`https://triton-lang.org/main/getting-started/tutorials/`（online softmax kernel）
