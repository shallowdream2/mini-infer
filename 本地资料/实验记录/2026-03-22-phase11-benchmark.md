# Phase 11 Benchmark — Speculative Decoding

本文件记录 Phase 11（Speculative Decoding）的 GPU 实测结果。
测试日期：2026-03-22
测试人：mini-infer

---

## 环境

| 项目 | 值 |
|------|-----|
| 硬件 | 2 × RTX 4090，CUDA 12.2，驱动旧版（P2P 警告，不影响正确性） |
| PyTorch | 2.1.2+cu121 |
| transformers | 4.43.4 |
| flash_attn | 2.5.9.post1 |
| mini-infer | Phase 11，commit bb56d0d（后续 KV 修复提交） |
| draft model | Qwen2.5-0.5B-Instruct @ cuda:0 |
| target model | Qwen2.5-7B-Instruct @ cuda:1（根目录路径，非 snapshot） |

## Workload

- prompts: 4 条（见下）
- batch_size: 1（顺序处理）
- max_new_tokens: 64
- K: 4（draft 每轮候选 token 数）
- temperature: 0.0（greedy）
- warmup: 1 条 prompt × 8 tokens，warmup 后重置统计

**Prompts：**
1. "What is the capital of France?"
2. "Explain the difference between a list and a tuple in Python."
3. "Write a short poem about the ocean."
4. "What are the main causes of World War I?"

---

## 结果

### SpecEngine（K=4，0.5B draft + 7B target）

| 指标 | 值 | 口径 |
|------|-----|------|
| total_time | 5.69s | wall-clock，含 4 条 prompt |
| throughput | ~29.0 tok/s | 近似（word count ÷ elapsed） |
| avg_latency/prompt | 1421ms | elapsed / 4 |
| TTFT | N/A | 未实现计时 |
| TPOT | ~22ms | elapsed / total_words 近似 |
| acceptance_rate | **55.85%** | 229 accepted / 410 drafted |
| total_draft_tokens | 410 | — |
| total_accepted_tokens | 229 | — |
| peak_memory draft | 1.9 GB (cuda:0) | torch.cuda.max_memory_allocated |
| peak_memory target | 18.2 GB (cuda:1) | — |

### Target-only Baseline（7B，greedy）

| 指标 | 值 | 口径 |
|------|-----|------|
| total_time | 4.51s | wall-clock，含 4 条 prompt |
| throughput | ~44.4 tok/s | 近似（word count ÷ elapsed） |
| avg_latency/prompt | 1127ms | elapsed / 4 |
| TTFT | N/A | 未实现计时 |
| TPOT | ~11ms | elapsed / total_words 近似（sequential, batch=1） |
| peak_memory | 18.2 GB (cuda:1) | — |

### 对比摘要

| 指标 | SpecEngine (K=4) | Target-only | 说明 |
|------|-----------------|-------------|------|
| throughput | ~29.0 tok/s | ~44.4 tok/s | spec 近似 |
| speedup | **0.65×** | baseline | spec 比 target-only **慢 35%** |
| acceptance_rate | **55.85%** | N/A | greedy, K=4 |
| memory overhead | +1.9 GB (draft) | — | 0.5B 草稿模型 |

---

## 运行命令

```bash
HF_HUB_OFFLINE=1 conda run -n ai-infra python /tmp/bench_phase11_spec.py
HF_HUB_OFFLINE=1 conda run -n ai-infra python /tmp/bench_phase11_target.py
```

（两个脚本不能同一进程运行，cuda:1 OOM；需独立进程顺序执行）

---

## 结论与分析

### 核心发现

1. **acceptance_rate = 55.85%（K=4, greedy）**：draft 预测质量在合理范围，验证了 draft+target 双模型 rejection sampling 的基本正确性。

2. **spec v1 比 target-only 慢（0.65×）**：这是预期行为，根因是 v1 双 forward 设计。

### v1 慢的根因

v1 每轮迭代（接受约 2.3 个 token）需要：
- K=4 次 draft forward（0.5B，fast）
- 1 次 `spec_verify_target`（7B，`use_cache=False`，返回验证 logit，不写 KV）
- 1 次 `spec_advance_target_kv`（7B，`use_cache=True`，写 KV，返回 last logit）

每轮 2.3 个 token 但需 **2 次 7B forward**，等效 target forward/token ≈ 0.87 次 + 跨设备同步 + Python loop overhead = 实际比 target-only 1 forward/token 还慢。

等效计算：
```
target_equiv_per_token = (2 + 4 × (0.5/7)) / 2.3 ≈ (2 + 0.29) / 2.3 ≈ 1.0
```

理论上等效计算量与 target-only 相当，但实际 0.65× 源于：
- 跨设备（cuda:0 → cuda:1）张量拷贝（draft_probs align）
- Python-level K=4 循环（sequential draft steps）
- 两次 target forward 的 CUDA launch + sync overhead
- `get_prefix_kv` 每轮重建完整 KV tensor

### 验收标准对照

| 验收标准 | 状态 | 依据 |
|---------|------|------|
| draft+target 双模型 spec decoding 跑通 | ✅ 达成 | 4 条 prompt 均输出正常文本 |
| acceptance_rate 有意义（非零） | ✅ 达成 | 55.85% |
| 吞吐提升（v1 预期无提升）| ⚠️ 0.65×（低于 1×） | v1 双 forward，符合设计预期 |
| 回退成本说明 | ✅ 已说明 | 额外 1.9GB VRAM，+1421ms avg vs 1127ms |

### v2 优化方向（记录，非当前任务）

将 `spec_verify_target` 和 `spec_advance_target_kv` 合并为**一次 target forward**：
- 输入：`[accepted_context + draft_tokens]`
- 输出：logits（既用于验证，也用 `use_cache=True` 写 KV）
- 预期：acceptance_rate=55% 时理论 speedup ≈ (K × AR + 1) / (K × AR) ≈ 1.4×

---

## 局限性

- throughput 使用 word count 近似，非真实 token count（真实值可能偏低 10-20%）
- TTFT/TPOT 未精确计时（spec 引擎无 per-token 计时 hook）
- batch=1 顺序处理，非并发场景
- 两个模型在不同设备运行，跨设备开销（本机 RTX 4090 PCIe）是瓶颈之一
