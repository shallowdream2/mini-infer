# 知识专题：Autoregressive Decode 与 KV Cache

这个文件深入解释 decoder-only 模型推理的核心机制，以及 KV Cache 的设计原理和实现方式，结合 mini-infer Phase 1 的真实代码。

## 主题

Decoder-only 大模型（如 Qwen2.5、LLaMA）的 autoregressive 推理中，KV Cache 如何工作，为什么能加速 decode，以及当前实现的局限性。

---

## 一、问题定义

Transformer 的 self-attention 在每一层计算：

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
```

在 autoregressive decode 中，每生成一个新 token，模型需要对所有之前的 token（包括 prompt）计算 attention。如果每步都从头计算，seq_len=512 时计算量是 seq_len=1 的 512×，整体 decode 复杂度是 O(n²)，完全不实用。

**KV Cache 解决的问题**：把历史 token 的 K/V 矩阵缓存下来，每步 decode 只计算新 token 的 K/V，然后和缓存拼接，把 decode 复杂度降到 O(n)。

---

## 二、核心原理

### Prefill 阶段

输入整个 prompt `[t₁, t₂, ..., tₙ]`，模型做完整 forward：

```
每一层：
  K_cache[1..n] = W_K · X[1..n]
  V_cache[1..n] = W_V · X[1..n]
  Attn = softmax(Q[1..n] · K_cache[1..n]^T / sqrt(d)) · V_cache[1..n]
```

输出：最后一位置的 logits（用于采样第一个新 token）+ 所有层的 K_cache/V_cache。

### Decode 阶段（第 i 步）

输入上一步生成的 token `tₙ₊ᵢ`：

```
每一层：
  k_new = W_K · x_{n+i}          # 只计算新 token 的 K/V
  v_new = W_V · x_{n+i}
  K_full = concat(K_cache, k_new)  # 拼接历史
  V_full = concat(V_cache, v_new)
  Attn = softmax(q_new · K_full^T / sqrt(d)) · V_full  # 新 Q attend to 所有历史
  更新 K_cache ← K_full
```

每步只计算 1 个 token 的 Q/K/V，attention 是 1×(n+i) 的矩阵乘法，O(n) 复杂度。

---

## 三、工程实现方式

### HuggingFace past_key_values（Phase 1 实现）

HF 的实现最简单：模型 forward 时 `use_cache=True`，输出里包含 `past_key_values`：

```python
# 类型：tuple[tuple[Tensor, Tensor], ...]
# 外层 tuple：num_layers 个元素
# 内层 tuple：(K, V)，shape [batch, num_kv_heads, seq_len, head_dim]

out = model(input_ids=input_ids, use_cache=True)
past_kv = out.past_key_values   # 存起来

# decode 时传回去
out = model(
    input_ids=[[last_token]],
    past_key_values=past_kv,
    use_cache=True
)
past_kv = out.past_key_values   # 每步都是新的（包含这步的 K/V）
```

**优点**：零代码改动，直接用模型内置实现。
**缺点**：
- 每个请求独立存一份完整 KV，无法共享物理内存
- 不同请求的 seq_len 不同，无法直接 batch 到一起做 decode
- 没有显存上限，长序列会 OOM
- 每步 decode 返回的是全新的 tuple（内存复制），而不是 in-place 更新

### Paged KV Cache（Phase 2 目标）

预先分配一个固定大小的 GPU tensor 池（`num_blocks × block_size × num_kv_heads × head_dim`），用 BlockTable 管理每个请求对物理 block 的映射：

```
逻辑地址：request_id + token_position
物理地址：block_id + slot_id（block_id * block_size + slot_id = token_position）
```

优点：显存有上限，可以统一管理，支持不同长度请求共享同一个 block pool，支持 batch decode。

---

## 四、设计取舍

| 方案 | 显存控制 | Batch Decode | 实现复杂度 |
|------|---------|-------------|-----------|
| HF past_key_values | 无上限 | 困难（seq_len 对齐问题）| 极低 |
| Paged KV Cache（gather 方案）| 有上限 | 中等（gather 后 batch）| 中等 |
| 真正 PagedAttention（vLLM）| 有上限 | 高效（直接 block 寻址）| 高（需自定义 CUDA kernel）|

mini-infer Phase 2 选择"Paged KV Cache + gather 做 batch decode"：正确性和内存管理是 Phase 2 的目标，零拷贝的高性能 attention 是 Phase 3+。

---

## 五、常见误区

**误区 1：KV Cache 越大越好**

不对。KV Cache 占用的显存和 `batch_size × seq_len × num_layers × 2 × num_kv_heads × head_dim × dtype_bytes` 成正比。对 Qwen2.5-7B（28 层，8 KV heads，128 head_dim，float16），单个请求 1024 token 的 KV 约占 0.46 GB。同时跑 32 个请求则是 15 GB，直接 OOM。Paged KV Cache 的价值不是扩大 KV，而是精确控制并复用碎片化的显存块。

**误区 2：Prefill 和 Decode 只是先后顺序问题**

不对。它们是计算特征完全不同的两个阶段：
- Prefill 是 compute-bound（大矩阵乘法）
- Decode 是 memory-bound（主要时间在读权重和 KV cache）

混合到一个 batch 里会互相干扰，这是 Prefill/Decode 分离调度的动机。

**误区 3：batch decode 只是把 input_ids stack 在一起**

不完全对。还需要把不同长度请求的 KV 对齐（padding 或动态 mask），确保每个请求的 attention 只看到自己的历史，不能跨请求。这需要正确的 `attention_mask`。

---

## 六、和 mini-infer 的关系

Phase 1：`ModelRunner._past_kv` 是一个 `dict[request_id, tuple]`，每个请求独立存 HF 格式的 past_key_values。这是最简实现，允许验证链路正确性，但 decode 只能串行。

Phase 2：`KVCacheManager` 将升级为真实 GPU block tensor pool + BlockTable。`decode_step` 将从 `for state: model(1 token)` 改为 `model([batch of 1 tokens], gathered_past_kv)`，实现真正的 batch decode。

---

## 七、进一步阅读

- vLLM 论文：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- HuggingFace Transformers 源码：`modeling_qwen2.py` 的 `Qwen2Attention.forward()` — 看 `past_key_value` 是怎么拼接的
- Flash Attention 2 论文：理解如何在 tiled 方式下高效做 attention，是 vLLM 后续的优化基础
