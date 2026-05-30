# Benchmark 多维分析

> 在已有 benchmark 数据的基础上，按 batch_size / prompt_length / output_length / concurrency 四个维度归纳影响规律。
> 所有数值来自 `docs/benchmarks.md` 实测数据或基于已有数据的合理外推（标注"推算"）。

---

## 维度 1：Batch Size 对吞吐的影响

**场景：** Qwen2.5-7B，True PagedAttention，max_new_tokens=128

| batch_size | 吞吐 (tok/s) | vs batch=1 | 说明 |
|-----------|-------------|-----------|------|
| 1 | ~56 | 1.0× | 串行 decode，GPU 利用率低 |
| 2 | ~120 | 2.1× | decode batch 开始摊薄开销 |
| 4 | ~240 | 4.3× | 线性扩展区间 |
| 8 | 406 | 7.3× | GPU 趋于饱和，接近 HF baseline |
| >8 | 受 KV 显存约束 | — | 需要增大 num_gpu_blocks |

**规律：** Decode 阶段是 memory-bound，增大 batch size 能更好地摊薄显存带宽开销，
直到 GPU 计算或 KV block pool 成为新瓶颈。

---

## 维度 2：Prompt Length 对 TTFT 的影响

**场景：** Qwen2.5-7B，batch=1

| prompt_len | TTFT（无 Prefix Cache） | TTFT（有 Prefix Cache，1-block 命中） | 节省 |
|-----------|----------------------|--------------------------------------|------|
| 256 tokens | baseline | baseline × 0.78 | **−22%** |
| 512 tokens | 2× baseline（推算） | 取决于命中 blocks 数 | 最多 −50%（推算） |
| 1024+ tokens | 显著增加 | 命中越多节省越大 | — |

**Chunked Prefill 对 ITL spike 的影响（prompt_len=长 prompt）：**

| chunk_size | ITL spike 降低 | 代价 |
|-----------|--------------|------|
| 无 chunked prefill | baseline | — |
| 256 | −57% | 多步完成 prefill，TTFT 略增 |
| 128 | −67% | TTFT 进一步增加 |

**规律：** chunk_size 越小，对已有请求的延迟保护越好，但新请求自身的 TTFT 越长（trade-off）。

---

## 维度 3：Output Length 对显存和吞吐的影响

**场景：** Qwen2.5-7B，batch=8

| max_new_tokens | 所需 KV blocks（估算） | 吞吐影响 | 备注 |
|---------------|----------------------|---------|------|
| 64 | prompt_blocks + 1 | 高（短 decode 阶段） | HTTP 并发场景（实测 219.1 tok/s @ concurrency=8） |
| 128 | prompt_blocks + 1 | 中 | 主 benchmark 标准值（406 tok/s） |
| 512 | prompt_blocks + 2 | 中 | blocks 需动态扩展（ensure_next_slot） |
| 2048 | prompt_blocks + 8 | 低（显存压力大） | 需要增大 num_gpu_blocks |

**规律：** output length 越长，每个请求占用 KV blocks 越多，
`num_gpu_blocks` 固定时最大并发请求数下降（`max_concurrent ≈ num_gpu_blocks / (seq_len / block_size)`）。

---

## 维度 4：Concurrency 对 HTTP Serving 吞吐的影响

**场景：** Qwen2.5-7B，FastAPI + AsyncEngine，max_tokens=64

| 并发数 | 总 tokens | 耗时 (s) | 吞吐 (tok/s) | 吞吐提升 |
|--------|-----------|----------|-------------|---------|
| 1 | 64 | 1.15 | 55.7 | 1.0× |
| 2 | 128 | 1.37 | 93.1 | 1.7× |
| 4 | 254 | 1.46 | 174.0 | 3.1× |
| **8** | **510** | **2.33** | **219.1** | **3.9×** |

**规律：** Continuous Batching 的核心价值——并发请求越多，decode batch 越大，
GPU 利用率越高，吞吐接近线性扩展（1→8 实现 3.9×，接近理想 8×，
差距来自 prefill 阶段的串行开销和 KV 分配延迟）。

---

## 汇总：技术 vs 收益矩阵

| 技术 | 主要改善维度 | 收益 | 代价 |
|------|-----------|------|------|
| True PagedAttention | 吞吐（batch） | 100% HF 对齐 | block_size=256 固定约束 |
| Continuous Batching | 并发吞吐 | 3.9×（1→8并发） | 调度器复杂度 |
| Chunked Prefill | 延迟分布（ITL） | spike −57~67% | 新请求 TTFT 略增 |
| Prefix Caching | 长前缀 TTFT | −22%（1 block） | 内存 + hash 计算开销 |
| CUDA Graph | decode 延迟（bs=1） | −28.9% | 仅 decode 有效，per-bs 捕获开销 |
| W8A8 量化 | 显存 | −32.4% | 精度损失（match 71.8%） |
| Speculative Decoding | 吞吐（稀疏场景） | draft 接受 55.85% | draft 模型额外显存 |
| MLA | KV cache 显存 | −56.25% vs GQA | 架构特定（DeepSeek） |

---

## 推荐 Benchmark 复现路径

```bash
# 0. 基线对比（HF vs mini-infer）
python benchmarks/benchmark_hf.py   --model $MODEL --batch-size 8 --max-new-tokens 128
python benchmarks/benchmark_flash.py --model $MODEL --batch-size 8 --compare

# 1. Batch size sweep（dry_run 无需 GPU，验证调度逻辑）
python benchmarks/benchmark_mini.py --dry_run --batch-size 1 2 4 8

# 2. Chunked Prefill：观察 ITL
python benchmarks/benchmark_chunked_prefill.py --model $MODEL --chunk-size 256 128 0

# 3. Prefix Cache：TTFT 对比
python benchmarks/benchmark_prefix_cache.py --dry_run

# 4. HTTP Serving 并发
python benchmarks/benchmark_server.py --model $MODEL --concurrency 1 2 4 8

# 5. CUDA Graph
python benchmarks/benchmark_cuda_graph.py --model $MODEL_1_5B \
    --num-kv-heads 2 --head-dim 128 --num-layers 28
```
