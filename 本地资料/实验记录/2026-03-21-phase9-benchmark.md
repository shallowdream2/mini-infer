# Phase 9 Chunked Prefill Benchmark

## 环境

| 项目 | 值 |
|------|----|
| 日期 | 2026-03-21 |
| 机器 | Ubuntu 24.04 + RTX 4090 × 2（单卡运行） |
| CUDA | 12.2 |
| PyTorch | 2.1.2+cu121 |
| transformers | 4.43.4 |
| flash_attn | 2.5.9.post1 |
| Python | 3.10.19 |
| 模型 | Qwen2.5-7B-Instruct（float16，~14.3 GB，单卡 cuda:0） |

## 测试对象

mini-infer `LLMEngine`（`add_request` / `step` 接口），Phase 9 Chunked Prefill 实现。

对照组 A：`chunk_prefill_size=0`（禁用分块，Phase 8 行为）
实验组 B：`chunk_prefill_size=128` 或 `chunk_prefill_size=256`

## Workload

| 参数 | 值 |
|------|----|
| 短请求并发数 | 7（已进入 running/decode 阶段） |
| 短请求 prompt 长度 | ~32 tokens（"a " × 32） |
| 短请求 max_new_tokens | 128 |
| 长请求 prompt 长度 | ~1024 tokens（"a " × 1024） |
| 长请求 max_new_tokens | 32 |
| KV 块大小 | 256 tokens/block |
| GPU 块数 | 200 |
| max_batch_size | 8 |

## 测量口径

- **max_itl_spike_ms**：长请求到达后至完成 prefill 前，所有短请求经历的最大相邻 token 间隔（跨 warmup 阶段与 long-arrival 阶段的时间戳连续记录，包含 `t_long_start` 跨越间隔）
- **ttft_long_ms**：`step()` 首次返回长请求 token 的时刻 − `add_request(long_prompt)` 时刻
- **throughput_toks**：总生成 token 数 / 总 wall time（全程，含 warmup）
- **TPOT**：N/A（未单独测量每步解码时间）
- **peak memory**：N/A（未测量，单卡 float16 场景已有 Phase 6 数据：~16 GB）

## 运行命令

```bash
export MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
export HF_HUB_OFFLINE=1
conda run -n ai-infra python benchmarks/benchmark_chunked_prefill.py --model $MODEL --chunk-size 128
conda run -n ai-infra python benchmarks/benchmark_chunked_prefill.py --model $MODEL --chunk-size 256
```

## 结果

### chunk_prefill_size=128（对照 chunk=0）

| 指标 | 对照组 A（chunk=0） | 实验组 B（chunk=128） | 变化 |
|------|-------------------|---------------------|------|
| 短请求最大 ITL spike | 138.4 ms | 45.9 ms | **−66.8%** |
| 长请求 TTFT | 138.4 ms | 399.3 ms | +260.9 ms |
| 总吞吐 | 290.0 tok/s | 296.8 tok/s | +2.3%（无回归） |

### chunk_prefill_size=256（对照 chunk=0）

| 指标 | 对照组 A（chunk=0） | 实验组 B（chunk=256） | 变化 |
|------|-------------------|---------------------|------|
| 短请求最大 ITL spike | 137.9 ms | 58.6 ms | **−57.5%** |
| 长请求 TTFT | 137.9 ms | 269.8 ms | +131.9 ms |
| 总吞吐 | 288.8 tok/s | 301.3 tok/s | +4.3%（无回归） |

### 不同 chunk size 的权衡对比

| chunk_size | ITL spike 降低 | 长请求 TTFT 增加 |
|-----------|--------------|----------------|
| 0（无分块） | 基准 138 ms | 基准 138 ms |
| 128 | −66.8%（45.9 ms） | +261 ms（399 ms 总计） |
| 256 | −57.5%（58.6 ms） | +132 ms（270 ms 总计） |

规律：chunk size 越小 → ITL spike 越低（更平滑）→ 长请求 TTFT 增加越多（需要更多轮 chunk）。

## 单元测试

```
conda run -n ai-infra python -m pytest tests/test_chunked_prefill.py -v
7 passed in 0.72s
```

全部通过（dry_run 路径）。

## 结论

1. **验收标准达成**：max_itl_spike 降低 ≥ 30%
   - chunk=128：−66.8% ✅（大幅超过目标）
   - chunk=256：−57.5% ✅（超过目标）

2. **吞吐无回归**：两组实验组吞吐均略高于对照组（+2-4%，属正常波动），无回归。

3. **TTFT 增加是设计内的权衡**：
   - chunk=256 时 TTFT 从 138 ms 增至 270 ms（+1×），可接受
   - chunk=128 时 TTFT 增至 399 ms（+2.9×），适合对 ITL 平稳性要求极高的场景

4. **Phase 9 的核心收益**：长 prefill 请求到达时，运行中的短请求不再遭遇 ~138 ms 的完整 prefill 阻塞。使用 chunk=256 可将最大 ITL 从 138 ms 降至 58 ms，用户感知到的流式输出不中断。

## 局限性

- 测试场景固定（7 短 + 1 长），未覆盖多个长请求并发到达的场景
- 短请求 max_new_tokens=128 较短，实际生产中更长的 decode 序列可能呈现不同 spike 分布
- TPOT 和 peak memory 未测量
- swap（preemption）与 chunked prefill 同时活跃的场景未测试
- 两次 chunk=0 对照组数据（137.9 ms 和 138.4 ms）一致，说明测量稳定
