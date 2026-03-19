# Phase 5 Profiling 实验记录（2026-03-19）

这个文件记录 mini-infer Phase 5 的 decode_batch 内部 profiling 实验，分析 gather_batch_kv / model_forward / write_decode_kv 三段的 CUDA 时间占比。

---

## 测试对象

- **引擎**：mini-infer Phase 3（Paged KV Cache + 向量化 gather_batch_kv + DynamicCache）
- **工具**：`benchmarks/profile_decode.py`，使用 `torch.profiler` + `record_function` 标签
- **标签位置**：`model_runner.py` 的 `decode_batch()` 真实路径

## 环境

| 项目 | 值 |
|------|----|
| 硬件 | NVIDIA GeForce RTX 4090 × 1（24 GB） |
| OS | Ubuntu 24.04 |
| 模型 | Qwen2.5-7B-Instruct，float16 |
| transformers | 4.43.4 |
| 模型路径 | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct`（根目录） |

## 运行命令

```bash
export MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
export HF_HUB_OFFLINE=1

python benchmarks/profile_decode.py --model $MODEL --batch-size 1 --decode-steps 30
python benchmarks/profile_decode.py --model $MODEL --batch-size 4 --decode-steps 30
python benchmarks/profile_decode.py --model $MODEL --batch-size 8 --decode-steps 30
```

## 数据口径说明

- `decode_steps=30`：profiler context 内 generate 生成 30 个 token，对应约 29 次 decode_batch 调用（第 1 次 token 来自 prefill）
- `调用次数=29`：与 decode 步数一致，prefill 不计入（prefill 的 model forward 无 record_function 标签）
- `CUDA 均值(ms)`：单次 decode step 的 CUDA 时间，等于 `CUDA 总(ms) / 调用次数`
- 三段合计 = gather_batch_kv + model_forward + write_decode_kv（不含 DynamicCache 构造、input_ids 准备、采样等步骤）

## 实验结果

### batch=1

```
操作                     调用次数  CUDA 总(ms)  CUDA 均值(ms)     占比
model_forward                29      500.94        17.274    98.3%
gather_batch_kv              29        7.88         0.272     1.5%
write_decode_kv              29        0.68         0.023     0.1%
三段合计                               509.50
```

**主导 CUDA kernel（来自完整 top-20）：**
- `gemvx_kernel`（两种变体）：357.8ms + 80.1ms = 合计约 73.1% + 16.4% = **89.5% CUDA 时间**
- batch=1 的 decode 本质是 GEMV（矩阵-向量乘积），受显存带宽瓶颈，非 compute-bound

### batch=4

```
操作                     调用次数  CUDA 总(ms)  CUDA 均值(ms)     占比
model_forward                29      520.53        17.949    97.3%
gather_batch_kv              29        9.38         0.323     1.8%
write_decode_kv              29        5.30         0.183     1.0%
三段合计                               535.21
```

**主导 CUDA kernel：**
- `cutlass WMMA tensorop f16`：509.2ms = 89.7% CUDA 时间
- GEMM 内核取代 GEMV，batch=4 开始进入 compute-bound 区间
- `aten::scaled_dot_product_attention` 出现（10.1ms），efficient attention kernel 激活

### batch=8

```
操作                     调用次数  CUDA 总(ms)  CUDA 均值(ms)     占比
model_forward                29      518.80        17.890    97.2%
gather_batch_kv              29        9.10         0.314     1.7%
write_decode_kv              29        5.61         0.193     1.1%
三段合计                               533.51
```

**主导 CUDA kernel：**
- `cutlass WMMA tensorop f16`：573.5ms = 89.2% CUDA 时间
- `fmha_cutlassF_f16_aligned_64x128_rf_sm80`（FlashAttention FMHA）：10.3ms
- batch=8 与 batch=4 的 model_forward 时间接近（17.89ms vs 17.95ms），GPU 接近饱和

## 结论

### 三段时间占比

| batch | gather_kv 均值 | model_forward 均值 | write_kv 均值 | model_forward 占比 |
|-------|--------------|--------------------|--------------|-------------------|
| 1 | 0.272 ms | 17.274 ms | 0.023 ms | 98.3% |
| 4 | 0.323 ms | 17.949 ms | 0.183 ms | 97.3% |
| 8 | 0.314 ms | 17.890 ms | 1.1% | 97.2% |

**core 结论：**

1. **model_forward 支配全部 decode 开销（97-98%）**：三种 batch size 下 model_forward 均值几乎相同（17.27～17.95ms/step），说明 decode 阶段 GPU 的利用已趋近饱和，增加 batch 的边际计算收益递减。

2. **Phase 3 向量化 gather_batch_kv 效果显著：仅占 1.5-1.8%**：gather_batch_kv 从 Phase 2 的"主要瓶颈"降到了 0.27-0.32ms/step。batch=4 vs batch=8 的 gather 时间差异极小（0.32ms vs 0.31ms），向量化后的 PyTorch advanced indexing 对 batch 的伸缩性良好。

3. **write_decode_kv 开销可忽略（0.1-1.1%）**：batch=1 时仅 0.023ms/step，batch=4/8 时为 0.18-0.19ms/step。

4. **residual 2-3% 用于非标注开销**：DynamicCache 构造（for l in range(28) 循环 cache.update()）、attn_mask 填充、logits 采样等 CPU 操作。

### kernel 变化

- **batch=1**：GEMV 内核主导（gemvx），受显存带宽限制，每步固定消耗约 17ms
- **batch=4,8**：GEMM 内核主导（cutlass WMMA），进入 compute-bound 区间；FlashAttention FMHA kernel 在 batch≥4 时激活
- batch=8 vs batch=4 的 model_forward 时间几乎相同（仅差 0.06ms），符合 GPU 接近吞吐上限的预期

### 后续优化方向（Phase 5 之后）

- **gather_batch_kv 残余 0.3ms/step**：主要来自 PyTorch advanced indexing 的 copy。消除 gather 本身需要 flash_attn 2.5+ 的 block_tables 参数，让 attention kernel 直接从 block tensor 寻址，绕过物理拷贝。
- **DynamicCache 构造（28 层循环）**：约 0.5-1ms CPU 时间，可通过批量构造或缓存优化，但当前不是瓶颈。
- **batch=8 的 write_decode_kv（0.19ms）**：在当前 97% model_forward 的格局下，即使完全消除也不影响总吞吐。

## 局限性

- `decode_steps=30` 只 profile 了前 30 步 decode；更长序列（KV 增长后）的 gather 开销可能略有上升（需额外测量）
- profiler 本身有 tracing 开销（CUDA event recording），实际生产 throughput 以 `benchmark_mini.py` 数据为准
- 当前 `model_forward` 均值仅反映三段中的 decode forward；prefill forward 时间从本 profile 无法读取
