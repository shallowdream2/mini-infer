# 知识专题：Paged Attention 与 Batch Decode

## 主题

大语言模型推理中的 KV Cache 分页管理与批次解码机制：原理、工程实现方式、设计取舍，以及在 mini-infer Phase 2 中的具体落地与局限。

---

## 一、问题定义

### 1.1 为什么需要 Paged KV Cache

在 Transformer 的自回归推理中，每个 decode step 需要访问所有历史 token 的 Key 和 Value。标准做法是将历史 KV 存在 `past_key_values` 里，每步 forward 后追加新 KV。

这种方式有两个根本问题：

**问题一：显存无上限**。序列越长，每个请求占用的显存越多，无法在启动前预测峰值，也无法跨请求统一调度。

**问题二：碎片化**。每个请求的 `past_key_values` 是一块连续内存，且随着 decode 推进需要频繁 realloc 或追加。多个请求的 KV 分散在 GPU 内存各处，空闲内存不连续，实际可用量远小于总量。

vLLM 的 PagedAttention 论文（2023）借鉴操作系统的虚拟内存分页思想解决了这两个问题。

### 1.2 为什么需要 Batch Decode

在串行 decode 中，每条请求独立做一次 `model(input_ids=[[token]], past_kv=...) forward`。对于 batch=N 的场景，这意味着 N 次 GPU kernel 调用，每次只有 1 个 token 的计算量，GPU 严重利用不足（compute-bound 计算太少，kernel launch overhead 占比大）。

batch decode 把 N 个请求的最后 token 拼成 `[N, 1]` 的 input_ids 做一次 forward，GPU compute 被充分利用。

---

## 二、核心原理

### 2.1 Block Table 与物理块

Paged KV Cache 的核心数据结构是两层映射：

```
逻辑 token 位置 pos
    → 逻辑块号:  block_idx = pos // block_size
    → 物理块号:  phys_blk  = block_table[request_id][block_idx]
    → 块内偏移:  slot_idx  = pos % block_size
    → KV 位置:   k_cache[l][phys_blk, slot_idx]
```

物理块由 FreeBlockPool 统一管理，所有请求共享同一个 pool，分配和归还都是 O(1)。

### 2.2 Batch Decode 的 KV 对齐

Batch decode 需要把不同长度的请求的 KV 对齐到同一个 tensor（模型 forward 要求固定形状）。

**为什么用左填充而不是右填充？**

Qwen2 使用 RoPE 位置编码，position_ids 不是固定的，而是从 attention_mask 动态推导：

```
position_ids[b, i] = (attention_mask[b, :i+1].sum()) - 1
                   = attention_mask[b, :i+1].cumsum()[-1] - 1
```

decode 阶段 input_ids 只有一个新 token，它的 position_ids 必须等于当前序列长度（从 0 开始计的位置）。

- **左填充**：填充区 mask=0 不参与 cumsum，真实 token 的 position_ids 从 0 连续累积，新 token 的 position = seq_len，正确
- **右填充**：新 token 在前面，position_ids = 填充位数（一般是 0 或 1），与实际序列位置不符，RoPE 错乱

### 2.3 Continuous Batching 的调度逻辑

静态 batching（HF generate）：一批请求一起开始，一起结束，等最慢的请求才能腾出位置。

Continuous batching：每个 decode step 后检查是否可以接入新请求。先完成的请求立即释放 KV 块，等待队列中的请求可以在下一步就加入。

```
while has_waiting or running:
    admit()       # 新请求进 running
    prefill()     # 新请求跑 prefill
    decode_batch()  # 所有 running 一次 batch forward
    cleanup()     # 完成的请求释放 KV 块
```

好处：GPU 不因等待"队尾最慢请求"而空转；长请求和短请求可以混合调度，整体利用率提高。

---

## 三、工程实现方式

### 3.1 mini-infer 的实现（Python 层）

mini-infer Phase 2 的 `gather_batch_kv()` 实现：

```python
def gather_batch_kv(self, request_ids):
    seq_lens = [self._seq_lens[rid] for rid in request_ids]
    max_seq_len = max(seq_lens)

    for l in range(self.num_layers):
        k_layer = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, ...)
        for b, (rid, seq_len) in enumerate(zip(request_ids, seq_lens)):
            pad = max_seq_len - seq_len
            for blk_idx, phys_blk in enumerate(self._block_tables[rid]):
                # 从 block pool 复制到 dense tensor
                k_layer[b, :, pad+start:pad+end, :] = k_cache[l][phys_blk, :n].permute(1,0,2)
```

每个 decode step 都执行一次完整复制，复制量 = O(batch × max_seq_len × num_layers × head_dim × kv_heads)。

### 3.2 vLLM 的实现（CUDA kernel）

真正的 PagedAttention 不复制 KV：

```
block_table: [batch, max_num_blocks]  (GPU tensor)
↓
自定义 CUDA kernel (paged_attention_v1/v2)
    - 每个 thread block 负责一个 query token 的一个 attention head
    - 通过 block_table 查找物理块号，直接在 block pool 上做 attention
    - 不需要把 KV 聚合到 dense tensor
```

代价是需要写 CUDA 或 Triton kernel，不能用标准 PyTorch attention API。

---

## 四、设计取舍

| 设计点 | mini-infer 选择 | vLLM 选择 | 取舍说明 |
|--------|---------------|---------|---------|
| KV gather | Python 层复制到 dense tensor | CUDA kernel 直接在 block pool 做 attention | mini-infer 实现简单，但每步复制开销大 |
| block_size | 固定参数（默认 16）| 固定参数（默认 16） | 太小则 BlockTable 变长，太大则内碎片严重 |
| 填充方式 | 左填充 | 左填充（连续 batching 标准） | 保证 RoPE position_ids 正确 |
| 准入估算 | `ceil((prompt + max_out) / block_size)` | 动态估算 | 保守估计，避免 mid-decode OOM |
| OOM 处理 | 快速失败 raise RuntimeError | 抢占（preemption）或 swap | 简化设计，不做请求抢占 |

### block_size 的选择

block_size 是一个隐藏的重要参数：
- 太小（如 4）：block table 很长，管理开销大，且每次 `gather_batch_kv` 的 block 迭代次数多
- 太大（如 256）：最后一个 block 的内碎片大（最坏情况浪费 block_size-1 个 slot）
- 16 是 vLLM 的默认值，是对两个方向的经验折中

---

## 五、常见误区

### 误区 1：Paged KV Cache 一定比普通 KV Cache 快

不一定。Paged KV Cache 的优势是**显存利用率高**（允许更多并发请求）和**无碎片**，不是单请求的延迟更低。

对于单请求、短序列的场景，普通 KV cache 可能更快（无复制开销）。

### 误区 2：Continuous Batching 等于 Dynamic Batching

Dynamic batching 通常是指在 batch 凑齐后统一发起推理（等待一定时间或凑够 N 条请求）。Continuous batching 是在 decode 层面的动态调度，粒度是每个 decode step，而不是每个请求。

### 误区 3：`gather_batch_kv()` 是 PagedAttention 的核心

不是。`gather_batch_kv()` 是一种把 paged KV 转换为 dense KV 再走普通 attention 的**过渡实现**，它额外引入了复制开销。真正的 PagedAttention 是用 CUDA kernel 直接在 block pool 上做 attention，省掉这次复制。

### 误区 4：左填充会导致 attention 计算浪费

左填充的填充区 mask=0，模型对这些位置做 attention 时得到的权重会被 mask 掉（softmax 后趋近 0）。但计算本身还是发生了（矩阵乘法仍包含填充部分），所以对于长短差异很大的请求 batch，填充开销确实不可忽略。

---

## 六、和 mini-infer 的关系

Phase 2 实现了 Paged KV Cache 的完整架构（BlockTable、FreeBlockPool、block tensor），以及基于 Python 层 `gather_batch_kv()` 的 batch decode。

当前主要局限：

1. **吞吐量低于 HF**（batch=8 时只有 49%）：根因是 `gather_batch_kv()` 每步复制，Phase 3 方向是换用 FlashAttention 或自定义 kernel 消除复制。

2. **DynamicCache 兼容性**：当前向 HF 传 tuple 格式的 `past_key_values` 会触发弃用警告，Phase 3 需要迁移。

3. **未测试 Continuous Batching 优势**：当前 benchmark 用同质 prompt，优势未体现。需要混合长度、混合到达速率的测试。

真实数据（RTX 4090，float16）：

| batch | mini-infer Phase 2 | HF baseline |
|-------|--------------------|-------------|
| 1 | 49.4 tok/s (TTFT 18.7ms) | 56.2 tok/s (TTFT 19.0ms) |
| 4 | 135.2 tok/s | 210.5 tok/s |
| 8 | 201.0 tok/s | 408.9 tok/s |

---

## 七、进一步阅读

- vLLM 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention*（Kwon et al., 2023）
- Flash Attention 论文：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（Dao et al., 2022）
- HuggingFace DynamicCache API 文档（transformers 4.40+）
- Qwen2.5 模型架构：`config.json` 中 `num_hidden_layers`、`num_key_value_heads`、`num_attention_heads`
