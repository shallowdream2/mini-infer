# FAQ

---

## 环境与安装

### 如何安装 flash-attn？

`flash-attn` 需要编译，不能通过 `pyproject.toml` 的 extras 自动安装：

```bash
pip install "flash-attn>=2.5.0" --no-build-isolation
```

`--no-build-isolation` 使用系统已有的 PyTorch，避免重新编译。

### 不安装 flash-attn 能跑吗？

可以。dry-run 模式和部分 benchmark 不依赖 flash-attn。
仅 True PagedAttention（Phase 6+）路径需要 `flash_attn_with_kvcache` 的 `block_table` 参数。

### 必须用 Conda 吗？

不必须。`pip install -e ".[serve,dev]"` 在任何 Python 3.10+ 虚拟环境中都可以工作。
Makefile 默认使用 `conda run -n ai-infra`，外部用户可以通过 `PYTHON=python make test-fast` 覆盖。

---

## 模型与权重

### 支持哪些模型？

当前主要验证过 Qwen2.5 系列（0.5B / 1.5B / 7B）和 DeepSeek-V2-Lite（Phase 14 MLA）。
架构参数（`num_hidden_layers` / `num_kv_heads` / `head_dim`）需在 `EngineConfig` 中明确指定。

### 7B 模型路径为什么不用 snapshot 子目录？

Qwen2.5-7B-Instruct 的 HuggingFace 缓存目录中，snapshot 子目录的部分 shard 软链接缺失，需使用根目录路径。

### 如何防止 transformers 联网检查？

设置环境变量：

```bash
export HF_HUB_OFFLINE=1
```

---

## 工程与架构

### block_size 为什么必须是 256 的倍数？

`flash_attn_with_kvcache` 对 KV block 对齐有要求。当前默认 `block_size=256`，不建议修改。

### Triton kernel（Phase 6.5 / 12.5）有接入主推理链路吗？

没有。两个 Triton kernel 都是**独立实验性实现**，通过各自的 benchmark 脚本验证，不替换主链路的 `flash_attn`。

### SpecEngine / PDEngine / TPEngine 和 LLMEngine 是什么关系？

它们是**独立的引擎扩展**，共享相同的 `ModelRunner` 和 `KVCacheManager`，但各自维护独立的生命周期。例如：
- `SpecEngine`：在 `LLMEngine` 基础上增加 draft model step
- `TPEngine`：用 `mp.spawn` + NCCL 替换单进程 forward
- `PDEngine`：拆分成两个独立的 Worker 进程

### MoE benchmark 为什么是 synthetic workload？

Phase 17–21 的 EP benchmark 基于 **synthetic MoE layer**（单层 forward），不包含完整 LLM serving 链路。原因：
1. 控制变量：隔离通信 vs 计算 overhead
2. 不依赖真实 MoE 模型权重（需要数十 GB）
3. 便于精确测量 ep_packed_bytes、ep_ideal_bytes 等通信口径

---

## Benchmark 与数据

### 如何复现 batch=8 吞吐达到 100% HF 的结论？

```bash
export MODEL=/path/to/Qwen2.5-7B-Instruct && export HF_HUB_OFFLINE=1
python benchmarks/benchmark_hf.py --model $MODEL --batch-size 8 --max-new-tokens 128
python benchmarks/benchmark_flash.py --model $MODEL --batch-size 8 --compare
```

### W8A8 量化的 greedy match 71.8% 是什么意思？

与 FP16 基线**逐 token 比较**的一致率。71.8% 表示约 72% 的生成 token 和 FP16 完全相同，其余有量化误差。
当前 decode 路径以 mixed fallback（int8 权重 + float32 activation）为主，精度高于纯 W8A8。

### 想看到更多数字，去哪里找？

完整 benchmark 数据与复现命令见 [docs/benchmarks.md](benchmarks.md)。

---

## 项目方向

### 这个项目会继续维护吗？

是的。当前计划是继续按阶段演进技术主线，同时逐步改善工程呈现质量。

### 可以贡献代码吗？

欢迎。建议先通过 `make test-fast` 验证环境可用，然后参考各 phase 的模块结构进行修改。
请确保改动不破坏现有测试，并为新功能添加对应的 dry_run 测试。

### 和 vLLM 的差距在哪里？

| 能力 | mini-infer | vLLM |
|------|-----------|------|
| Paged KV Cache | ✅ | ✅ |
| Continuous Batching | ✅ | ✅ |
| PagedAttention | ✅（flash_attn） | ✅（自研 CUDA kernel） |
| Chunked Prefill | ✅ | ✅ |
| Prefix Caching | ✅ | ✅ |
| Speculative Decoding | ✅ | ✅ |
| Tensor Parallelism | ✅（Megatron-LM 风格） | ✅ |
| MoE Expert Parallelism | ✅（synthetic，2-GPU） | ✅（真实模型） |
| 量化 | ✅（W8A8 原型） | ✅（AWQ/SmoothQuant/FP8） |
| 多模型 / 多租户 | ❌ | ✅ |
| production SLA | ❌ | ✅ |
| 真实 CUDA kernel | ❌（使用 flash_attn） | ✅（xFormers / FlashInfer） |
| 完整 RLHF / LoRA serving | ❌ | ✅ |

mini-infer 的核心价值不是功能覆盖度，而是**每个机制的实现路径和 benchmark 分析都清晰可追踪**。
