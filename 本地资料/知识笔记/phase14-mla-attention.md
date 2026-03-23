# Phase 14 知识笔记：MLA（Multi-head Latent Attention）

## 核心思路

MLA 把 KV 投影到低秩 latent 空间，只缓存 latent 向量而非完整 K/V：

```
C_KV = X @ W_dkv          # 压缩：hidden → latent（kv_lora_rank 维）
K_nope = C_KV @ W_uk      # 展开：latent → K（推理时按需）
V      = C_KV @ W_uv      # 展开：latent → V
K_pe   = X @ W_kr + RoPE  # RoPE 分量单独处理，所有 head 共享
```

## 压缩比（DeepSeek-V2-Lite）

| 策略 | bytes/token/layer | 相对 GQA |
|------|-------------------|---------|
| GQA（4KV, 128dim） | 2,048 | 100% |
| MLA naive | 10,240 | 500% |
| MLA latent | **1,152** | **56.25%** |

相同 32 GB VRAM：GQA 606 × seq=1024，MLA latent 1,078 × seq=1024（1.78×）

## 三种实现

- **naive**：每步展开完整 K/V，cache 大（10,240 B），计算少
- **latent**：只缓存 latent（1,152 B），每步展开全部历史，计算多
- **absorbed**：预计算 W_k_absorbed/W_v_absorbed，直接用 latent 算 score，cache 同 latent

## 矩阵吸收

```python
# 预计算（一次性）
W_k_absorbed = W_uk.T  # (num_heads, kv_lora_rank, qk_nope_head_dim)

# decode 时（替代 kv_b_proj 展开）
kv_for_score = einsum("bsd,hde->bhse", compressed_kv_normed, W_k_absorbed)
score_nope = q_nope @ kv_for_score.T
```

实测：batch=1 下 absorbed 比 naive 慢 16~28%（einsum overhead），大 batch 长序列才有优势。

## 关键细节

- `kv_a_layernorm` 必须在 attention 前应用，cache 存 raw latent
- `k_pe` 是 MQA 风格（1 head），广播到 num_heads
- V2-Lite 的 `q_lora_rank=None`（不压缩 Q），V2/V3 有 Q 压缩

## 踩坑

1. CPU 不支持 fp16 matmul → 用 float32
2. HF DeepseekV2Attention 强制 attention_mask 非 None → 构造 causal mask
3. absorbed 版漏掉 kv_a_layernorm → max diff 0.165，加上后 < 1e-4
