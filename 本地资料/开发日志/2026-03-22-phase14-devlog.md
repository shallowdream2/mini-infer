# Phase 14 开发日志：MLA（Multi-head Latent Attention）

**日期**：2026-03-22
**阶段**：Phase 14

---

## 背景

Phase 13 完成 Tensor Parallelism 后，开始 Phase 14：理解 DeepSeek-V2 的 MLA 架构，实现三种 KV cache 策略并用真实模型验证。

---

## 主要事件

### 模型下载

- 启动 DeepSeek-V2-Lite 后台下载（~31.4 GB）
- 误以为下载停止，重启了第二个进程，导致两个进程同时下载同一文件
- 发现后 kill 重复进程（PID 527538/527563），保留原始进程（PID 514244）
- 下载速度约 8-18 KB/s（梯子对 HF CDN 限速），历时数小时完成
- 最终 30 GB 下载完成，0 个 .incomplete shard

### Track A：CPU 实现

- 实现 `MLAConfig`、`MLAKVCacheNaive`、`MLAKVCacheLatent`、`RMSNorm`
- 实现 `MLAAttentionNaive`（与 HF 等价，缓存完整 K/V）
- 实现 `MLAAttentionLatentCache`（只缓存 latent，即时展开）
- 实现 `compute_kv_cache_bytes`（理论压缩比计算）
- 写 5 个 CPU 测试，全部通过

### Track B：GPU 验证（模型下载完成后）

- 发现 CPU fp16 matmul 不支持，改为 float32
- 发现 HF DeepseekV2Attention 强制要求 attention_mask 非 None，构造 causal mask
- `test_gpu_layer_equivalence` 通过，max diff = 0.2655（RoPE 差异）

### Track C：矩阵吸收优化

- 实现 `MLAAttentionAbsorbed`，预计算 `W_k_absorbed`/`W_v_absorbed`
- 第一版漏掉 `kv_a_layernorm`，max diff = 0.165，调试后修复
- 修复后 max diff < 1e-4，两个 absorbed 测试通过
- 新增 `benchmark_mla.py --section 3`，三种实现延迟对比

### Benchmark 结果

- MLA latent / GQA 压缩比：56.25%（1,152 vs 2,048 bytes/token/layer）
- 相同 32 GB VRAM：GQA 606 × seq=1024，MLA latent 1,078 × seq=1024（1.78×）
- absorbed 在 batch=1 下比 naive 慢 16~28%（einsum overhead）

---

## 关键决策

1. **不接入 LLMEngine 主链路**：DeepSeek-V2-Lite 是 MoE 模型，接入需要实现 MoE routing，超出范围
2. **cache 存 raw latent**：kv_a_layernorm 在 attention 时即时做，避免 norm 参数更新后 cache 失效
3. **absorbed 版用 einsum 而非 matmul**：需要分离 nope/rope 两部分 score，SDPA 不支持

---

## 遗留问题

- `_absorbed_built` 在 `load_state_dict` 后可能失效（建议修复，非阻塞）
- absorbed 版在小 batch 下无性能优势，需更大规模验证
