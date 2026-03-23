# Phase 15 PD 解耦 Benchmark 实验记录

**日期**：2026-03-23
**环境**：Ubuntu 24.04，RTX 4090 × 2，PyTorch 2.1.2+cu121，transformers 4.43.4
**模型**：Qwen2.5-1.5B-Instruct（fp16，28层，2 KV heads，head_dim=128）
**命令**：
```bash
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_pd_disagg.py --section 1
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_pd_disagg.py --section 2
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_pd_disagg.py --section 3
conda run -n ai-infra python -m pytest tests/test_pd_disagg.py -v
```

---

## Section 1：理论 KV 传输大小

**口径**：fp16，K+V 两路（GQA），MLA 只存一路压缩向量

| 模型 | seq_len | KV传输大小 | 架构 |
|------|---------|------------|------|
| Qwen2.5-1.5B | 128 | 3.50 MB | GQA |
| Qwen2.5-1.5B | 512 | 14.00 MB | GQA |
| Qwen2.5-1.5B | 1024 | 28.00 MB | GQA |
| Qwen2.5-7B | 128 | 7.00 MB | GQA |
| Qwen2.5-7B | 512 | 28.00 MB | GQA |
| Qwen2.5-7B | 1024 | 56.00 MB | GQA |
| DeepSeek-V2-Lite MLA | 128 | 3.80 MB | MLA latent+rope（单向量）|
| DeepSeek-V2-Lite MLA | 512 | 15.19 MB | MLA latent+rope（单向量）|
| DeepSeek-V2-Lite MLA | 1024 | 30.38 MB | MLA latent+rope（单向量）|

**结论**：同机传输即使 7B seq=1024 也仅 ~56 MB，共享内存延迟预期 < 5ms。

---

## Section 2：真实 GPU 端到端正确性验证

- 模型：Qwen2.5-1.5B，prompt="What is the capital of France?"，max_new_tokens=32
- LLMEngine 输出：`' The capital of France is Paris. It is located in the north of the country and i'`
- PDEngine 输出：`' The capital of France is Paris. It is located in the north of the country and i'`
- **结果一致：✓**（greedy 采样，两进程各自加载相同模型，输出完全相同）

---

## Section 3：TTFT 分解（prefill / transfer / decode）

**Workload**：3 个 prompt，batch=1，max_new_tokens=64，n_warmup=1，greedy
**Prompts**：attention 机制解释 / Python 2 vs 3 差异 / 神经网络架构描述

### 计时口径说明

| 字段 | 含义 | 来源 |
|------|------|------|
| 端到端时间 | `generate()` wall-clock | PDEngine 主进程 |
| prefill_time | tokenize + HF forward + extract_kv | PrefillWorker 内部 perf_counter |
| transfer_time（近似） | total - prefill - decode，含 pickle + Queue IPC | 近似估算 |
| decode_time | rebuild_cache + decode loop | DecodeWorker 内部 perf_counter |

### 结果（3 prompt 平均）

| 指标 | Unified LLMEngine | PDEngine |
|------|-------------------|----------|
| 平均端到端时间 (ms) | 459.2 | 546.0 |
| prefill 时间 (ms) | N/A | 12.3 |
| transfer 时间·近似 (ms) | N/A | 14.7 |
| decode 时间 (ms) | N/A | 519.0 |
| 相对开销 | 1.00× | **1.19×** |

---

## 测试结果

```
tests/test_pd_disagg.py::test_kv_transfer_payload    PASSED
tests/test_pd_disagg.py::test_kv_transfer_queue      PASSED
tests/test_pd_disagg.py::test_extract_kv_from_past   PASSED
tests/test_pd_disagg.py::test_rebuild_dynamic_cache  PASSED
tests/test_pd_disagg.py::test_pd_engine_dry_run_single PASSED
tests/test_pd_disagg.py::test_pd_engine_dry_run_batch  PASSED
tests/test_pd_disagg.py::test_worker_dry_run          PASSED

7 passed in 5.71s
```

---

## 分析与结论

1. **正确性满足**：greedy 输出 token 级别完全一致，KV 传输 + DynamicCache 重建数学路径验证正确

2. **TTFT 分解**：prefill 本身只需 12.3ms（1.5B 短 prompt），transfer 近似 14.7ms（含 pickle IPC），decode 占绝大部分（519ms = 64 tokens × ~8ms/token）

3. **1.19× 开销**：PDEngine 比 Unified LLMEngine 慢 19%。来源：
   - 两进程各自加载模型（启动时间在预热中摊销）
   - pickle 序列化 KV tensor（28层 × seq_len 的 fp16 tensor）
   - IPC Queue 额外内存拷贝

4. **生产差距**：本实现用 pickle+Queue，真实 PD 解耦系统（Mooncake）用 RDMA 或 GPU-Direct，transfer_time 可降至 < 1ms，开销比应接近 1.00×

---

## 局限性说明

- 两进程各自加载一份 1.5B 模型，GPU 显存 2×（~6 GB vs 统一 ~3 GB）
- transfer_time 是近似值（total - prefill - decode），并非直接测量
- 当前串行发送请求，未测试真正的 prefill/decode overlap（生产优势：prefill 处理请求 i+1 时，decode 已在处理请求 i）
- 未测试 throughput（当前只测 latency，吞吐需要多请求并发才有意义）
- TPOT / Peak Memory 口径：N/A（架构验证阶段不测完整 TPOT/memory）
