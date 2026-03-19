# 知识专题：LLM Decode 性能解剖——Profiling 方法与瓶颈层次

本文深度解析如何对 LLM decode 阶段进行 profiling，理解 batch size 对 CUDA kernel 类型的影响，以及 Paged KV Cache 格式转换代价的量化方法，结合 mini-infer Phase 5 的真实实验数据。

---

## 主题

**LLM Decode 的 CUDA 时间分布：为什么 model_forward 占 97%，剩余 3% 在哪里**

---

## 一、问题定义

当 LLM 推理引擎的吞吐优化接近瓶颈时，需要量化回答两个问题：

1. **各个操作的实际 CUDA 时间占比是多少？** 避免凭直觉判断哪里是瓶颈
2. **和 baseline 的差距来自哪里？** 指导下一步优化方向

对 mini-infer 来说，Phase 3 优化后已经达到 HF baseline 的 88.4%，剩余 12% 的根因需要定位。

---

## 二、核心原理

### 2.1 torch.profiler 的工作方式

`torch.profiler` 通过 CUDA Event API 记录 GPU 操作的开始/结束时间，在 Python 层用 `record_function` 上下文管理器打标签：

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    engine.generate(prompts, max_new_tokens=30)
    torch.cuda.synchronize()
```

关键特性：
- **`record_function` 未激活时是 no-op**：不影响正常性能，可以留在生产代码里
- **CPU time ≠ CUDA time**：CUDA 是异步的，CPU time 反映 Python 调度延迟，CUDA time 反映真实 GPU 计算时间
- **`cuda_time_total` vs `cuda_time`**：前者是所有调用的总 CUDA 时间，后者是均值（total / count）

### 2.2 decode 阶段的 GEMV 与 GEMM

decode 阶段每个 step 的核心计算是：对 input_ids（每个请求 1 个 token）做 Transformer forward。

- **batch=1**：`[1, 1, hidden] × [hidden, hidden]` → 矩阵-向量乘（GEMV）
  - 受**显存带宽**限制：每步需要把 ~7B 个参数的权重矩阵从 HBM 读出来
  - kernel：`gemvx_kernel`（专门为 GEMV 优化的 kernel）
  - 特征：无论如何调整，单步时间由 HBM 带宽决定

- **batch≥4**：`[batch, 1, hidden] × [hidden, hidden]` → 矩阵-矩阵乘（GEMM）
  - **compute-bound**：Tensor Core 利用率高，CUDA core 可以复用权重
  - kernel：`cutlass WMMA tensorop f16`（Warp Matrix Multiply-Accumulate）
  - 特征：batch 从 1 增加到合适大小，throughput 近线性提升，因为每次读取的权重被多个请求复用

FlashAttention FMHA kernel 在 batch≥4 时激活（单请求时注意力退化为简单点乘，不走 FMHA 路径）。

### 2.3 GPU 饱和的识别

GPU 饱和的标志：增加 batch size，per-step CUDA 时间不再增加（或增加很小）。

理论上：
- 未饱和时：增加 batch，GPU 更充分利用，per-step 时间不变，throughput 线性增加
- 饱和后：增加 batch，GPU 计算时间增加，per-step 时间上升，throughput 增加放缓

mini-infer 实测：
- batch=4 → batch=8：per-step model_forward 时间差 0.06ms（17.95ms → 17.89ms），几乎不变
- 说明 RTX 4090 在 batch=4 时对 Qwen2.5-7B decode 已接近饱和

---

## 三、工程实现方式

### 在关键段加 record_function

在 `model_runner.py` 的 `decode_batch()` 中，对三个主要阶段打标签：

```python
# 阶段 1：从 block tensor 聚合 KV
with torch.profiler.record_function("gather_batch_kv"):
    k_batch, v_batch, seq_lens = self.kv_cache.gather_batch_kv(request_ids)

# 阶段 2：DynamicCache 构造（此段未标注，是未标注开销的一部分）
cache = DynamicCache()
for l in range(num_layers):
    cache.update(k_batch[l], v_batch[l], l)  # ← 28 次 aten::cat

# 阶段 3：Transformer batch forward
with torch.profiler.record_function("model_forward"):
    with torch.no_grad():
        out = self.model(...)

# 阶段 4：写回新 KV
with torch.profiler.record_function("write_decode_kv"):
    self.kv_cache.write_decode_kv(request_ids, k_new, v_new)
```

注意：DynamicCache 构造在 gather 和 model_forward 之间，**没有被任何标签覆盖**，它的 CUDA 时间（28 次 aten::cat）出现在"未标注"的部分。

### profiling 脚本

```bash
HF_HUB_OFFLINE=1 python benchmarks/profile_decode.py \
    --model $MODEL --batch-size 8 --decode-steps 30
```

输出三段时间 + 占比，以及 top-20 完整 CUDA 操作表。

---

## 四、设计取舍

### 标注粒度：粗还是细

**粗标注（三段）**：简单，输出清晰，适合验证整体格局（谁是主要开销）。缺点：DynamicCache 构造被漏掉。

**细标注**：在 DynamicCache 构造循环、attn_mask 构造、采样等步骤也加标签。优点：100% 的时间都有归属。缺点：代码里的标签太多，影响可读性，且 no-op 特性仍然保证不影响性能。

本项目选择粗标注（3 段），因为主要目的是验证 gather 和 model_forward 的占比，不需要完整的时间分解。

### 何时用 profiler，何时用 CUDA Event

- **CUDA Event 计时**：更轻量，适合精确测量两个 GPU 操作之间的时间间隔
- **torch.profiler**：功能完整，可以捕获所有 CUDA kernel 调用，适合探索性分析

本项目用 torch.profiler，因为需要知道 model_forward 内部哪些 kernel 主导（cutlass vs gemvx vs FlashAttention），CUDA Event 只能测整体时间。

---

## 五、常见误区

**1. "profiler overhead 很大，不能用于生产代码"**

不对。`record_function` 在 profiler 未激活时是 no-op，开销为零。torch.profiler 的 overhead 只在 profiler 激活时存在（CUDA event recording 约 1-2%）。标签本身可以永久留在代码里。

**2. "batch 越大，per-step CUDA 时间越短"**

不一定。batch=1 → batch=4 时，GEMV 切换到 GEMM，per-step 时间变化不大，但 throughput 提升 3.6×（因为每步处理了 4 个 token 而不是 1 个）。batch=4 → batch=8 时，per-step 时间几乎不变，throughput 近 2×。

**3. "gather 只占 1.7%，优化它意义不大"**

这个说法本身是对的，但背后的推论有陷阱：gather 从"瓶颈（Phase 2）"降到"1.7%（Phase 3）"，说明向量化是有效的。但"与 HF 差 12%"的差距**不是来自 gather 的 1.7%**，而是来自 paged→dense 格式转换的整体代价（gather + DynamicCache 构造）。光看 gather 的 1.7% 会错过 DynamicCache 构造的 ~0.5ms。

**4. "CPU time 和 CUDA time 用哪个？"**

LLM inference 的性能以 CUDA time 为主。CPU time 包含 Python 调度延迟，在 GPU 异步执行时会严重虚高（CPU 发了 kernel 就继续跑，但 CUDA time 记录的是 GPU 实际执行的时间）。分析 GPU 瓶颈一律用 CUDA time。

---

## 六、和 mini-infer 的关系

Phase 5 的 profiling 数据：

| batch | gather (ms/step) | model_forward (ms/step) | write_kv (ms/step) | model_forward 占比 |
|-------|-----------------|------------------------|-------------------|-------------------|
| 1 | 0.272 | 17.274 | 0.023 | 98.3% |
| 4 | 0.323 | 17.949 | 0.183 | 97.3% |
| 8 | 0.314 | 17.890 | 0.193 | 97.2% |

三段之外（未标注）约 3.8ms/step（batch=8），主要来自 DynamicCache 预填充（28 次 aten::cat）和 attention_mask/input_ids 构造。

**四个阶段的瓶颈迁移总结：**

| 阶段 | 主要瓶颈 | 表现 |
|------|---------|------|
| Phase 1 | 串行 decode（Python for 循环） | batch=8 吞吐 ≈ batch=1，不随 batch 增长 |
| Phase 2 | gather_batch_kv 的 Python 嵌套循环 | batch=8 吞吐只有 HF 的 49.1% |
| Phase 3 | paged→dense 格式转换的物理拷贝 | batch=8 达到 HF 的 88.4% |
| Phase 5（测量）| DynamicCache 预填充 + 其他系统开销 | 差距约 3.8ms/step，各有 1ms 量级 |

彻底消除 paged→dense 代价需要 flash_attn 2.5+ 的 block_tables：attention kernel 直接从 block tensor 按物理地址寻址，不再需要 gather 或 DynamicCache 构造。

---

## 七、进一步阅读

- torch.profiler 官方文档：`torch.profiler.profile` 和 `record_function` 的使用
- FlashAttention-2 论文（Dao et al., 2023）：FMHA kernel 设计和 block_tables 接口
- vLLM 论文中的 PagedAttention 章节：attention kernel 直接接受 block_tables 的实现
- NVIDIA Nsight Systems / Nsight Compute：更底层的 GPU kernel profiling 工具
- cutlass WMMA tensor core 文档：理解 GEMM 内核的 Tensor Core 使用方式
