# Phase 5 知识笔记：Decode Profiling 与 GPU 性能解剖

本文件是 mini-infer Phase 5 的核心概念速记，供自己回顾。

---

## torch.profiler.record_function

- 在代码段入口/出口加标签，profiler 激活时记录 CUDA 和 CPU 时间
- **profiler 未激活时是 no-op**，不影响正常性能，可以直接留在生产代码里
- `key_averages()` 返回聚合统计，可按 key 过滤自己打的标签
- 三个关键字段：`cuda_time_total`（总 CUDA 时间）、`cuda_time`（均值，即 total/count）、`count`（调用次数）

---

## batch=1 vs batch≥4：GEMV 和 GEMM

- **batch=1 decode**：每层的矩阵运算是 `[hidden] × [hidden, hidden]`，本质是矩阵-向量乘（GEMV）
  - 受限于**显存带宽**：需要把整个权重矩阵从 HBM 读出来
  - kernel：gemvx（RTX 4090 上约 17ms/step）
- **batch≥4 decode**：`[batch, hidden] × [hidden, hidden]`，变成矩阵-矩阵乘（GEMM）
  - **compute-bound**：可以更好利用 Tensor Core / CUDA core
  - kernel：cutlass WMMA tensorop f16
  - batch=4 开始激活 FlashAttention FMHA kernel

**实测数字（Qwen2.5-7B, RTX 4090）**：
- batch=1：17.27ms/step（gemvx，带宽瓶颈）
- batch=4：17.95ms/step（GEMM，计算瓶颈）
- batch=8：17.89ms/step（几乎同 batch=4，GPU 接近饱和）

这解释了 throughput 为什么 batch 1→4 提升 3.6×，但 4→8 只提升 1.86×。

---

## GPU 饱和的特征

- **per-step 时间趋于稳定**：batch=4 和 batch=8 的 model_forward 每步时间差 0.06ms（≈0.3%）
- **throughput 近线性增加**：batch 翻倍，step 时间不变，总 token 数翻倍 → throughput 接近 2×
- **边际收益递减**：batch=1→4 是 GEMV→GEMM 的质变（3.6×），batch=4→8 是 GEMM 内部的量变（1.86×）

---

## Paged KV Cache 的格式转换代价

decode_batch 的完整流程：
1. `gather_batch_kv`：block tensor（paged）→ dense tensor（需要 KV copy）
2. DynamicCache 构造：`for l: cache.update(k, v, l)` → 28 × aten::cat
3. model_forward（带 DynamicCache）
4. `write_decode_kv`：新 token KV 写回 block tensor

向量化后 gather 本身只 0.3ms，但整个 paged→dense 流程（gather + DynamicCache 构造）约 1-2ms。HF 没有这个开销（past_key_values 直接原地 append）。

**消除路径**：flash_attn 2.5+ 的 block_tables 参数，让 attention kernel 直接从 block tensor 按物理地址寻址，绕过物理拷贝。

---

## 残余差距分解（batch=8）

| 阶段 | mini-infer (ms/step) | HF 等效 (ms/step) | 差 |
|------|---------------------|--------------------|-----|
| gather_batch_kv | 0.31 | — | +0.31 |
| DynamicCache 构造 | ~0.3-0.5 | ~0.1 | +0.2-0.4 |
| model_forward | 17.89 | ≈17.89 | ≈0 |
| 其他（mask/sampling）| ~2-3 | ~1-2 | +1 |
| **总计** | **22.2** | **19.6** | **+2.6 (12%)** |

注：HF 的 past_key_values 原地 append（aten::cat in model.forward，每层每步 1 次），mini-infer 还多一次预填充（28 次 cat）。
