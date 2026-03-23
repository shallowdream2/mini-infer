# Multi-head Latent Attention（MLA）

## 主题

DeepSeek-V2/V3/R1 的 KV cache 压缩机制：通过低秩 latent 投影将 KV cache 从 O(num_heads × head_dim) 压缩到 O(kv_lora_rank)，在相同显存下支持更多并发请求。

---

## 一、问题定义

标准 MHA/GQA 的 KV cache 随序列长度线性增长：

```
GQA（Qwen2.5-7B）：4 KV heads × 128 dim × 2（K+V）× 2 bytes = 2,048 bytes/token/layer
```

长序列、大 batch 场景下 KV cache 成为显存瓶颈。MLA 的目标：**在不损失模型质量的前提下，将 KV cache 压缩到更小的低秩空间**。

---

## 二、核心原理

### 2.1 KV 低秩压缩

```
# 压缩：hidden → latent
C_KV = X @ W_dkv                    # (seq, kv_lora_rank)，d_c << num_heads × head_dim

# 展开：latent → K/V（推理时按需）
K_nope = C_KV @ W_uk                # (seq, num_heads, qk_nope_head_dim)
V      = C_KV @ W_uv                # (seq, num_heads, v_head_dim)
```

只缓存 `C_KV`（低秩 latent），推理时即时展开 K/V。

### 2.2 RoPE 分量单独处理

K 被拆成两部分：
- `K_nope`：从 latent 展开，不含位置信息
- `K_pe`：直接从 hidden 投影 + RoPE，所有 head 共享（MQA 风格）

```
K_pe = X @ W_kr + RoPE              # (seq, 1, qk_rope_head_dim)
K = concat(K_nope, K_pe)            # (seq, num_heads, qk_nope_head_dim + qk_rope_head_dim)
```

cache 存储：`C_KV（kv_lora_rank 维）+ K_pe（qk_rope_head_dim 维）`

### 2.3 DeepSeek-V2-Lite 超参

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_size | 2048 | Transformer 隐藏层维度 |
| num_heads | 16 | Q attention head 数 |
| q_lora_rank | None | V2-Lite 不压缩 Q |
| qk_nope_head_dim | 128 | K/Q 非 RoPE 分量维度 |
| qk_rope_head_dim | 64 | K/Q RoPE 分量维度 |
| kv_lora_rank | 512 | KV latent 维度（d_c） |
| v_head_dim | 128 | V head 维度 |

### 2.4 压缩比计算

```
MLA latent cache = (kv_lora_rank + qk_rope_head_dim) × 2 bytes
                 = (512 + 64) × 2 = 1,152 bytes/token/layer

GQA（4KV, 128dim）= 4 × 128 × 2 × 2 = 2,048 bytes/token/layer

压缩比 = 1,152 / 2,048 = 56.25%
```

---

## 三、工程实现方式

### 3.1 三种实现策略

**MLAAttentionNaive**（与 HF 等价）
- 每次 forward 展开完整 K/V，缓存 key_states + value_states
- cache 大小：10,240 bytes/token/layer（V2-Lite）
- 适合：验证正确性，对比基准

**MLAAttentionLatentCache**
- 只缓存 compressed_kv + k_pe（1,152 bytes/token/layer）
- 每个 decode step 对全部历史 latent 做 kv_b_proj 展开
- 适合：显存受限场景，接受额外计算开销

**MLAAttentionAbsorbed**（矩阵吸收）
- 预计算 `W_k_absorbed = W_uk^T`，`W_v_absorbed = W_uv^T`
- decode 时直接 `compressed_kv_normed @ W_k_absorbed` 得到等价于 k_nope 的中间量
- 跳过显式 k_nope 展开，理论上在长序列大 batch 下更快
- cache 大小同 latent cache

### 3.2 关键实现细节

**kv_a_layernorm 的位置**：必须在 attention 计算前对 compressed_kv 做 norm，cache 存储 raw（未 norm）值：

```python
# cache 存 raw
new_cache = MLAKVCacheLatent(compressed_kv=compressed_kv, k_pe=k_pe)
# attention 时 norm
compressed_kv_normed = self.kv_a_layernorm(compressed_kv)
```

**k_pe 广播**：k_pe 是 MQA 风格（1 head），在拼接 key_states 时广播到 num_heads：

```python
key_states[:, :, :, qk_nope_head_dim:] = k_pe  # (bsz, 1, seq, rope_dim) → broadcast
```

---

## 四、设计取舍

| 策略 | cache 大小 | decode 计算量 | 适用场景 |
|------|-----------|-------------|---------|
| naive | 10,240 B/token/layer | 低（K/V 已展开） | 计算资源充足，显存宽裕 |
| latent | 1,152 B/token/layer | 高（每步展开全部历史） | 显存受限，序列不太长 |
| absorbed | 1,152 B/token/layer | 中（einsum，理论更优） | 大 batch 长序列 |

**矩阵吸收的实际效果**：在 batch=1、seq_len≤1024 的测试中，absorbed 比 naive 慢 16~28%（einsum overhead）。矩阵吸收的优势需在 seq_len >> kv_lora_rank=512 且 batch 较大时才能体现。

---

## 五、常见误区

**误区 1：MLA 比 GQA 节省更多**

MLA latent（1,152）比 GQA（2,048）节省 43.75%，但比 MLA naive（10,240）节省 88.75%。MLA naive 比 GQA 大 5×，因为 MLA 有更多 Q heads（16 vs 4）且 q_head_dim 更大（192 vs 128）。

**误区 2：矩阵吸收总是更快**

矩阵吸收避免了 k_nope 展开，但引入了 einsum。在小 batch 下 einsum 的 overhead 超过了节省的计算量。只有在 batch 较大、seq_len 远大于 kv_lora_rank 时才有优势。

**误区 3：cache 存 normed latent**

应存 raw（未 norm）的 compressed_kv，norm 在 attention 时即时做。存 normed 值会导致 kv_a_layernorm 参数更新后 cache 失效。

**误区 4：CPU 可以用 fp16**

PyTorch CPU 不支持 fp16 matmul（`addmm_impl_cpu_ not implemented for Half`），测试时需用 float32。

---

## 六、和 mini-infer 的关系

Phase 14 实现了三个版本的 MLA attention，用于：
1. 理解 DeepSeek-V2 的 KV cache 压缩机制
2. 验证 latent cache 的压缩比（56.25% vs GQA）
3. 实现矩阵吸收优化并验证数学等价性

**未接入主链路**：DeepSeek-V2-Lite 是 MoE 模型，接入 LLMEngine 需要实现 MoE routing，超出 Phase 14 范围。

相关文件：
- `mini_infer/mla_attention.py`：三种实现
- `tests/test_mla_attention.py`：8 个测试（CPU + GPU）
- `benchmarks/benchmark_mla.py`：理论对比 + 真实显存 + 延迟对比

---

## 七、进一步阅读

- DeepSeek-V2 技术报告（arXiv:2405.04434）Section 2.1：MLA 数学推导
- DeepSeek-V2 技术报告 Section 2.1.2：矩阵吸收优化（Efficient Inference）
- `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664.../modeling_deepseek.py`：HF 参考实现
