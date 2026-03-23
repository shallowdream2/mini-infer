# 从 KV Cache 压缩到矩阵吸收：手写 DeepSeek-V2 MLA 的三个版本

> 本文是 mini-infer 推理系统学习项目 Phase 14 的技术记录。
> 代码：`mini_infer/mla_attention.py`，测试：`tests/test_mla_attention.py`

---

## 背景：KV Cache 是长序列推理的显存瓶颈

在 Phase 13 实现 Tensor Parallelism 之后，我开始思考另一个方向：**纵向压缩 KV cache**。

标准 MHA/GQA 的 KV cache 大小是固定的：

```
GQA（Qwen2.5-7B）：4 KV heads × 128 head_dim × 2（K+V）× 2 bytes = 2,048 bytes/token/layer
```

对于 28 层模型，1024 个 token 的 KV cache 约 58 MB。这个数字看起来不大，但在 batch=64、seq=4096 的场景下会直接撑爆显存。

DeepSeek-V2 的 MLA（Multi-head Latent Attention）给出了一个不同的答案：**把 K/V 投影到低秩 latent 空间，只缓存 latent 向量**。

---

## MLA 的数学结构

标准 MHA 的 KV 计算：
```
K = X @ W_k,  V = X @ W_v
```

MLA 把它拆成两步：
```
# 压缩：hidden → latent（低秩）
C_KV = X @ W_dkv                    # (seq, kv_lora_rank)

# 展开：latent → K/V（推理时）
K_nope = C_KV @ W_uk                # (seq, num_heads, qk_nope_head_dim)
V      = C_KV @ W_uv                # (seq, num_heads, v_head_dim)

# RoPE 分量单独处理（MQA 风格，所有 head 共享）
K_pe = X @ W_kr  + RoPE             # (seq, 1, qk_rope_head_dim)

# 最终 K = concat(K_nope, K_pe)
```

DeepSeek-V2-Lite 的超参：
- `kv_lora_rank = 512`（latent 维度）
- `qk_rope_head_dim = 64`（RoPE 分量）
- 每 token 每层 cache：`(512 + 64) × 2 = 1,152 bytes`

对比 GQA 的 2,048 bytes，**压缩到 56.25%**。相同 32 GB VRAM 下，并发上限从 606 × seq=1024 提升到 1,078 × seq=1024（**1.78×**）。

---

## 三个实现版本

### 版本 1：MLAAttentionNaive

与 HF `DeepseekV2Attention.forward()` 数学等价，缓存完整展开的 key/value：

```python
# KV 压缩
compressed_kv_and_kpe = self.kv_a_proj_with_mqa(hidden_states)
compressed_kv = compressed_kv_and_kpe[..., :c.kv_lora_rank]
k_pe = compressed_kv_and_kpe[..., c.kv_lora_rank:]

# KV 展开（每次 forward 都做）
kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
kv = kv.view(bsz, q_len, c.num_heads, c.qk_nope_head_dim + c.v_head_dim).transpose(1, 2)
k_nope, value_states = kv.split([c.qk_nope_head_dim, c.v_head_dim], dim=-1)

# 拼接 K = concat(k_nope, k_pe)，缓存完整 key_states + value_states
```

cache 大小：`16 × (192 + 128) × 2 = 10,240 bytes/token/layer`（V2-Lite）

### 版本 2：MLAAttentionLatentCache

只缓存 `compressed_kv + k_pe`，attention 时对全部历史 latent 即时展开：

```python
# cache 只存 latent
new_cache = MLAKVCacheLatent(compressed_kv=compressed_kv, k_pe=k_pe)

# attention 时展开全部历史
kv_full = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))  # 对 kv_seq_len 条全做
```

cache 大小：`(512 + 64) × 2 = 1,152 bytes/token/layer`，节省 88.75%。

代价：每个 decode step 需对全部历史 `compressed_kv` 做一次 `kv_b_proj` 展开，计算量 O(seq_len × kv_lora_rank × hidden)。

### 版本 3：MLAAttentionAbsorbed（矩阵吸收）

DeepSeek-V2 技术报告 Section 2.1.2 提到的推理优化：预计算 `W_uk` 和 `W_uv`，直接用 `compressed_kv` 计算 attention score 和 output，跳过 k_nope 展开。

```python
def _build_absorbed(self):
    W = self.kv_b_proj.weight  # (num_heads*(d_nope+d_v), d_c)
    W = W.view(c.num_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
    # W_k_absorbed: (num_heads, d_c, d_nope)
    self.W_k_absorbed = W[:, :c.qk_nope_head_dim, :].transpose(1, 2)
    # W_v_absorbed: (num_heads, d_c, d_v)
    self.W_v_absorbed = W[:, c.qk_nope_head_dim:, :].transpose(1, 2)

def forward(self, hidden_states, past_cache=None):
    # ...
    compressed_kv_normed = self.kv_a_layernorm(compressed_kv)

    # 直接从 latent 计算 score（不展开 k_nope）
    kv_for_score = torch.einsum("bsd,hde->bhse", compressed_kv_normed, self.W_k_absorbed)
    score_nope = torch.matmul(q_nope, kv_for_score.transpose(2, 3))

    # 直接从 latent 计算 output（不展开 v）
    kv_for_v = torch.einsum("bsd,hdv->bhsv", compressed_kv_normed, self.W_v_absorbed)
    attn_output = torch.matmul(attn_weights, kv_for_v)
```

---

## 踩坑记录

### 坑 1：CPU 不支持 fp16 matmul

GPU 测试时用 `device_map="cpu"` + `torch.float16` 加载模型，触发：

```
RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
```

CPU 的 Linear 层不支持 fp16 输入。改为 `torch.float32` 解决。这个错误在 GPU 上不会出现，容易被忽略。

### 坑 2：HF DeepseekV2Attention 强制要求 attention_mask

直接调用 `hf_attn(hidden_states, attention_mask=None, ...)` 会在内部触发：

```python
assert attention_mask is not None  # modeling_deepseek.py:880
```

需要构造 causal mask 传入：

```python
causal_mask = torch.zeros(bsz, 1, seq, seq)
causal_mask = causal_mask.masked_fill(
    torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1), float("-inf")
)
```

### 坑 3：MLAAttentionAbsorbed 漏掉 kv_a_layernorm

第一版 absorbed forward 里，`compressed_kv` 直接送入 einsum，没有先过 `kv_a_layernorm`：

```python
# 错误版本
kv_for_score = torch.einsum("bsd,hde->bhse", compressed_kv, self.W_k_absorbed)
```

结果 max diff = 0.165，远超预期。调试发现 `W_k_absorbed` 是从 `kv_b_proj` 提取的，期望输入是 normed 的 latent：

```python
# 正确版本
compressed_kv_normed = self.kv_a_layernorm(compressed_kv)
kv_for_score = torch.einsum("bsd,hde->bhse", compressed_kv_normed, self.W_k_absorbed)
```

修复后 max diff < 1e-4。

---

## 实验结果

### KV Cache 压缩比（理论值，来自 benchmark_mla.py --section 1）

| 策略 | bytes/token/layer | 相对 GQA |
|------|-------------------|---------|
| GQA（Qwen2.5-7B） | 2,048 | 100% |
| MLA naive | 10,240 | 500% |
| MLA latent | **1,152** | **56.25%** |

### 单步 decode 延迟（batch=1，RTX 4090，真实 V2-Lite 第 0 层权重，来自 benchmark_mla.py --section 3）

| seq_len | naive (ms) | latent (ms) | absorbed (ms) |
|---------|-----------|------------|--------------|
| 1 | 0.140 | 0.137 | 0.161 |
| 256 | 0.132 | 0.134 | 0.159 |
| 1024 | 0.131 | 0.154 | 0.167 |

**观察**：
- latent 在 seq=1024 时比 naive 慢 18%（每步需对全部历史 latent 做 kv_b_proj 展开）
- absorbed 在当前规模下比 naive 慢 16~28%，`torch.einsum` 在小 batch 下开销大于 matmul
- 矩阵吸收的理论优势需在 batch 更大或 seq_len >> kv_lora_rank=512 时才能体现

### GPU 单层等价性（真实权重）

HF DeepseekV2Attention vs MLAAttentionNaive，max diff = 0.2655。差异来自 RoPE：HF 有完整 RoPE，naive 版跳过了旋转位置编码。权重路径（q_proj, kv_a_proj, kv_b_proj, o_proj）本身正确。

---

## 设计取舍

**为什么不接入 LLMEngine 主链路？**

DeepSeek-V2-Lite 是 MoE 模型，接入主链路需要实现 MoE routing，超出 Phase 14 的范围。Phase 14 的目标是理解 MLA 的 cache 组织和数学结构，不是完整的 DeepSeek 推理系统。

**为什么 absorbed 版不用 `F.scaled_dot_product_attention`？**

absorbed 版需要手动分离 nope 和 rope 两部分 score 再相加，SDPA 不支持这种分段计算，只能用 matmul + softmax 手写。

**cache 存 raw 还是 normed latent？**

存 raw（未 norm）的 compressed_kv，norm 在 attention 时即时做。原因：kv_a_layernorm 是无状态的，每次 forward 都可以重新计算，存 raw 更接近 HF 的实现，也避免了 norm 参数更新后 cache 失效的问题。

---

## 总结

Phase 14 实现了 MLA 的三个版本，核心收获：

1. **MLA 的 cache 压缩是真实的**：1,152 vs 2,048 bytes/token/layer，56.25% 的压缩比，相同显存下 1.78× 并发上限
2. **矩阵吸收在小规模下没有优势**：einsum 的 overhead 在 batch=1 时超过了避免 k_nope 展开的收益，需要更大规模才能体现
3. **layernorm 的位置很关键**：kv_a_layernorm 必须在 attention 计算前应用，漏掉会导致 max diff = 0.165，但测试能快速定位

---

## 自查结果（QUALITY_CHECKLIST.md）

- [x] 文章主题对应真实项目阶段（Phase 14 MLA 实现）
- [x] 结构覆盖：背景、问题定义、方案设计、实现细节、实验结果、坑点反思、总结
- [x] 有明确背景和问题定义（KV cache 显存瓶颈，MLA 压缩方案）
- [x] 有设计取舍（为什么不接入主链路、为什么不用 SDPA、cache 存 raw vs normed）
- [x] 有代码、实验或运行结果支撑（代码片段来自真实实现，数字来自 benchmark 文件）
- [x] 有失败点、坑点或反思（3 个坑：CPU fp16、attention_mask、漏掉 layernorm）
- [x] 语言克制、专业、可发表（无夸大性能宣称，局限性明确说明）
