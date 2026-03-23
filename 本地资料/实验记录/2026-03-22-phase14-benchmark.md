# Phase 14 MLA Benchmark 实验记录

**日期**：2026-03-22
**环境**：Ubuntu 24.04，2 × RTX 4090，PyTorch 2.1.2+cu121，transformers 4.43.4
**模型**：DeepSeek-V2-Lite（16B MoE，2.4B active，27层，hidden=2048）
**命令**：
```bash
conda run -n ai-infra python benchmarks/benchmark_mla.py --section 1
conda run -n ai-infra python benchmarks/benchmark_mla.py --section 2
conda run -n ai-infra python benchmarks/benchmark_mla.py --section 3
```

---

## Section 1：理论 KV Cache 大小对比

对比基准（fp16，seq_len=1024，num_layers=27）：

| 策略 | bytes/token/layer | 相对 GQA |
|------|-------------------|---------|
| GQA（Qwen2.5-7B，4KV heads，head_dim=128） | 2,048 | 100.00% |
| MLA naive（16 heads，q_head_dim=192，v_head_dim=128） | 10,240 | 500.00% |
| MLA latent（kv_lora_rank=512，rope_dim=64） | 1,152 | 56.25% |

全局 KV cache（seq_len=1024，27 layers）：
- GQA：58.0 MB
- MLA naive：289.9 MB
- MLA latent：32.6 MB

**压缩比**：MLA latent / GQA = 56.25%，MLA latent / MLA naive = 11.25%

**并发上限估算（32 GB 可用 VRAM）**：
- GQA：606 × seq=1024
- MLA latent：1,078 × seq=1024（**1.78×**）

---

## Section 2：真实模型 GPU 显存测量

- 模型加载峰值显存：**15.20 GB**（fp16，device_map=auto，2×4090）
- 生成 64 tokens 峰值显存：**15.21 GB**
- 增量（KV cache + activations）：**0.009 GB**（极小，因 MoE 模型 KV cache 本身很小）
- 生成内容：正常中文输出，内容连贯

---

## Section 3：三种实现单步 decode 延迟对比

测试条件：batch=1，GPU（RTX 4090），真实 V2-Lite 第 0 层权重，fp16，warmup=10，repeat=50

| seq_len | naive (ms) | latent (ms) | absorbed (ms) | absorbed/naive |
|---------|-----------|------------|--------------|---------------|
| 1 | 0.140 | 0.137 | 0.161 | 1.16× |
| 64 | 0.136 | 0.136 | 0.162 | 1.19× |
| 256 | 0.132 | 0.134 | 0.159 | 1.21× |
| 1024 | 0.131 | 0.154 | 0.167 | 1.28× |

**观察**：
- naive vs latent：seq_len=1024 时 latent 慢 18%（每步需对全部历史 compressed_kv 做 kv_b_proj 展开）
- absorbed 在当前测试规模下比 naive 慢 16~28%，原因是 `torch.einsum` 在小 batch 下比 `matmul` 开销大
- 矩阵吸收的理论优势（避免 k_nope 展开）在更大 batch 或更长序列（seq_len >> kv_lora_rank=512）下才能体现

**口径说明**：
- 仅测单层 attention 延迟，不含 MoE routing、FFN、embedding 等
- 无 causal mask（三种实现一致，不影响相对对比）
- throughput / TTFT / TPOT：N/A（Phase 14 为架构理解阶段，不做端到端吞吐测试）

---

## 验收标准检查

| 标准 | 结果 |
|------|------|
| test_absorbed_equivalence 通过（atol < 1e-4） | ✅ max diff < 1e-4 |
| MLA latent / GQA 压缩比 < 60% | ✅ 56.25% |
| test_gpu_layer_equivalence 通过（max diff < 0.5） | ✅ max diff = 0.2655 |
| benchmark_mla.py --section 3 输出延迟数据 | ✅ |
| pytest tests/test_mla_attention.py 全部通过 | ✅ 8 passed |

---

## 局限性

1. Section 3 只测单层，不代表完整模型推理性能
2. absorbed 版在当前规模下无性能优势，需更大 batch 才能体现矩阵吸收收益
3. GPU 等价性测试（test_gpu_layer_equivalence）max diff=0.2655 来自 RoPE 差异（HF 有 RoPE，naive 无），权重路径本身正确
4. MLA 未接入 LLMEngine 主链路，无法与 Qwen2.5 做端到端吞吐对比
