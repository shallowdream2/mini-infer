# Qwen2.5-7B Baseline Benchmark — Phase 1 vs HF

这个文件记录 Qwen2.5-7B-Instruct 在 HuggingFace Transformers baseline 和 mini-infer Phase 1 下的首次真实 benchmark 结果。

## 环境

- 日期：2026-03-19
- 系统：Ubuntu 24.04，RTX 4090 × 2（单卡 cuda:0）
- CUDA：12.2，PyTorch：2.1.2+cu121，transformers：4.43.4
- 模型：`Qwen/Qwen2.5-7B-Instruct`（local float16）
- 命令：
  ```bash
  python benchmarks/benchmark_hf.py   --model <path> --batch-size {1,4,8} --max-new-tokens 128 --device cuda:0 --dtype float16
  python benchmarks/benchmark_mini.py --model <path> --batch-size {1,4,8} --max-new-tokens 128 --device cuda:0 --dtype float16
  ```

## Workload

- Prompt：固定 8 条中文推理问题（PROMPTS 列表），按 batch_size 截取
- Output：max_new_tokens=128，greedy（do_sample=False）
- HF baseline：同一 batch 内 padding 对齐，整批 generate
- mini-infer Phase 1：请求逐条串行 prefill + decode

## 结果

### HF Baseline（transformers generate）

| batch_size | prompt_len(padded) | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | Peak Mem (GB) |
|-----------|-------------------|-----------|--------------|-------------------|---------------|
| 1 | 10 tokens | 19.0 | 17.78 | 56.2 | 15.78 |
| 4 | 11 tokens | 20.8 | 18.99 | 210.4 | 15.81 |
| 8 | 15 tokens | 23.6 | 19.53 | 408.9 | 15.88 |

### mini-infer Phase 1（串行 past_key_values）

| batch_size | 总耗时 (s) | Throughput (tok/s) | Peak Mem (GB) | TTFT | TPOT |
|-----------|-----------|-------------------|---------------|------|------|
| 1 | 2.27 | 56.4 | 15.78 | 未测 | 未测 |
| 4 | 9.08 | 56.4 | 15.81 | 未测 | 未测 |
| 8 | 18.17 | 56.3 | 15.84 | 未测 | 未测 |

## 分析

### 关键观察

**1. batch=1 时两者吞吐几乎相同（56.2 vs 56.4 tok/s）。**
单请求时两者执行路径等价，差异在测量噪声范围内。

**2. mini-infer Phase 1 的吞吐随 batch 完全不增长，维持在 ~56 tok/s。**
原因：`engine.py` 中当前的 `decode_step` 是 `for state in states: model(single_token)`，每条请求独立执行一次 forward，没有合并成一个 batch forward。batch=4 时总耗时是 batch=1 的 4 倍（9.08 vs 2.27s），线性增长。

**3. HF baseline 吞吐随 batch 近线性增长（56→210→408）。**
因为 HF 的 `generate` 在同一个 batch 内所有序列共享同一次 GPU forward，计算效率随 batch 提升。

**4. 显存两者几乎相同。**
Phase 1 使用 HF 的 `past_key_values`，存储结构与 HF baseline 一致。显存差异 < 0.1 GB 属于正常波动。

### 吞吐差距

| batch_size | HF baseline | mini-infer Ph1 | 差距倍数 |
|-----------|------------|---------------|---------|
| 1 | 56.2 | 56.4 | ~1× |
| 4 | 210.4 | 56.4 | 3.7× |
| 8 | 408.9 | 56.3 | 7.3× |

差距随 batch 线性扩大，完全符合串行 decode 的预期行为。

## 数据口径说明

- HF baseline 的 Throughput = `output_len × batch_size / total_time`
- mini-infer 的 Throughput = `tokenizer.encode(output) 的 token 数 / total_time`（逐条 decode 后汇总）
- 两者口径不完全一致，但在相同 output_len 下误差很小
- mini-infer Phase 1 **未测 TTFT / TPOT**（benchmark_mini.py 缺少分阶段计时，待 Phase 2 补齐）

## 局限性

- prompt 极短（10-15 tokens padded），不代表真实场景
- 未测长 prompt（512+ tokens）或长输出（512+ tokens）
- mini-infer 无 TTFT/TPOT 测量，无法与 HF baseline 逐指标对比
- 单次测量，未做多次平均

## 结论

Phase 1 的串行 decode 设计已被数据确认：**单请求性能与 HF 持平，多请求时吞吐不随 batch 增长，差距随 batch 线性扩大**。

这是 Phase 2 要解决的核心问题：把串行 `for state: model(1 token)` 改为真正的 batch decode，并配合 Paged KV Cache 管理不同长度请求的内存。
