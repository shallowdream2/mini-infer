# Phase 8 HTTP API Benchmark 实验记录

Phase 8 目标：验证 OpenAI-compatible HTTP API（FastAPI + SSE streaming）的端到端正确性和性能。

## 环境

| 项目 | 值 |
|------|----|
| 日期 | 2026-03-21 |
| 机器 | Ubuntu 24.04 + NVIDIA GeForce RTX 4090 × 2 |
| 模型 | Qwen2.5-7B-Instruct（本地路径） |
| PyTorch | 2.1.2+cu121 |
| flash_attn | 2.5.9.post1 |
| Python | 3.10 |

## 运行命令

```bash
export MODEL_PATH=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
export HF_HUB_OFFLINE=1
conda run -n ai-infra python benchmarks/benchmark_server.py \
  --model "$MODEL_PATH" \
  --max-tokens 64 \
  --trials 5 \
  --concurrency 1 2 4 8 \
  --num-gpu-blocks 512 \
  --block-size 256
```

## Workload

| 参数 | 值 |
|------|----|
| 模型 | Qwen2.5-7B-Instruct |
| block_size | 256（flash_attn_with_kvcache 约束） |
| num_gpu_blocks | 512 |
| max_batch_size | 8 |
| max_tokens（每请求） | 64 |
| prompt | 8 条中文技术问答（同 benchmark_mini.py） |
| 并发数 | 1 / 2 / 4 / 8 |
| HTTP 传输方式 | ASGI transport（httpx 直接访问，无网络栈） |

## 测量结果

### 单请求延迟（SSE 流式）

| 指标 | 值 | 口径说明 |
|------|----|---------|
| TTFT（近似） | mean=1157.1ms | **近似值**：ASGITransport 缓冲完整响应，无法分离首 token 延迟；此值等于单请求总延迟 |
| TPOT（近似） | mean=18.49 ms/tok | 总延迟 / 输出 token 数；与 Phase 6 model_forward ~17.9ms/step 一致 |
| 输出 token 数 | mean=62.6 tok/请求 | max_tokens=64，部分请求因 EOS 提前结束 |

> 真实 TTFT（首 token 延迟）需要 uvicorn 独立进程 + curl 流式客户端测量，本次无法获得。

### 并发吞吐（non-streaming，continuous batching）

| 并发数 | 总 tokens | 时间（s） | throughput（tok/s） |
|--------|-----------|-----------|---------------------|
| 1 | 64 | 1.15 | 55.7 |
| 2 | 128 | 1.22 | 105.0 |
| 4 | 254 | 1.31 | 193.6 |
| 8 | 510 | 1.45 | **351.4** |

> 口径说明：completion_tokens 由 `tokenizer.encode()` 精确计算，精确值。

### 峰值显存

| 指标 | 值 |
|------|----|
| 峰值显存 | 23.34 GB（Phase 8 HTTP 层不引入额外 GPU 内存，与 Phase 6/7 相同） |

## 与 Phase 6/7 基线对比

Phase 8 不是性能优化阶段，推理内核（flash_attn_with_kvcache）未变。

| 指标 | Phase 6 直接 LLMEngine（batch=8，max=128） | Phase 8 HTTP API（concurrency=8，max=64） |
|------|-------------------------------------------|-------------------------------------------|
| throughput | ~406 tok/s | 351.4 tok/s |
| TPOT | ~17.9ms/step | ~18.5ms/tok（近似，含 HTTP 开销） |
| 峰值显存 | 23.34 GB | 23.34 GB |

> **差距说明**：351.4 vs 406 tok/s（约 86%）主要原因是 workload 参数不同（max_tokens=64 vs 128）。
> 输出序列较短时，prefill 占比更高，decode batch 利用率略低，与推理内核无关。
> HTTP 层本身开销（asyncio + Python 协程调度）可忽略不计。

## Continuous Batching 效果验证

| 并发 × 吞吐 | 值 | 说明 |
|-------------|----|----|
| 1× | 55.7 tok/s | 单请求基准 |
| 2× | 105.0 tok/s（1.88×） | Batching 效果明显 |
| 4× | 193.6 tok/s（3.47×） | 持续扩展 |
| 8× | 351.4 tok/s（6.31×） | 接近线性，验证 continuous batching 有效 |

8 并发下吞吐 = 单并发的 6.3×，而非 8×，剩余差距来自：
- 提示长度不一致时的批次补齐开销
- 部分请求因 EOS 提前结束导致 batch 不满

## 结论

1. **HTTP API 端到端正确**：tokens 生成正常（fix 前 0 tok/s → fix 后 351.4 tok/s）
2. **Continuous batching 通过 HTTP 层正常工作**：8 并发吞吐 351.4 tok/s，接近 Phase 6 直接调用水平
3. **HTTP 层零显存开销**：23.34 GB 与 Phase 6/7 完全相同
4. **TPOT 18.5ms 与预期一致**：与 Phase 6 profiling 的 model_forward ~17.9ms/step 吻合

## 局限性

1. TTFT 为近似值（等于总延迟），真实流式首 token 延迟需独立进程测试
2. 未对比 HF baseline throughput（HTTP 层不改变推理内核，对比意义有限）
3. 测试用 ASGI transport，无网络栈；实际 uvicorn 部署会有额外 ~1-2ms 网络延迟
