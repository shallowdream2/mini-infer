# 解剖 Decode：gather 降到 1.7%，还差 12% 在哪里

这篇文章是 mini-infer 项目的第五篇技术复盘。Phase 3 做完向量化 gather_batch_kv 后，throughput 从 49.1% HF 涨到 88.4% HF，但距离 100% 还有 12%。Phase 5 用 torch.profiler 对 decode_batch 的三个阶段进行计时，试图把这 12% 的位置找出来。

结果比预期更有意思。

---

## 一、要测什么

Phase 3 之后，我们对 decode_batch 的直觉是：主要时间在 model_forward，gather 已经不是瓶颈了。但"直觉"不够，需要数字。

`decode_batch()` 的主要结构是三段：

```
gather_batch_kv      ← 从 block tensor 聚合各请求历史 KV
model_forward        ← 一次 batch Transformer forward
write_decode_kv      ← 把新 token 的 KV 写回 block tensor
```

在 `model_runner.py` 的三段入口加上 `torch.profiler.record_function` 标签：

```python
with torch.profiler.record_function("gather_batch_kv"):
    k_batch, v_batch, seq_lens = self.kv_cache.gather_batch_kv(request_ids)

with torch.profiler.record_function("model_forward"):
    with torch.no_grad():
        out = self.model(...)

with torch.profiler.record_function("write_decode_kv"):
    self.kv_cache.write_decode_kv(request_ids, k_new, v_new)
```

`record_function` 在 profiler 未激活时是 no-op，不影响正常推理性能。

---

## 二、数据

环境：Ubuntu 24.04，RTX 4090，Qwen2.5-7B-Instruct float16，decode 30 步。

| batch | gather (ms/step) | model_forward (ms/step) | write_kv (ms/step) | model_forward 占比 |
|-------|-----------------|------------------------|-------------------|-------------------|
| 1 | 0.272 | 17.274 | 0.023 | **98.3%** |
| 4 | 0.323 | 17.949 | 0.183 | **97.3%** |
| 8 | 0.314 | 17.890 | 0.193 | **97.2%** |

三段合计：batch=1 约 509ms，batch=4 约 535ms，batch=8 约 534ms（30 步中 29 步是 decode）。

---

## 三、97% 说明了什么

model_forward 占 97-98%，这和预判一致，但数字有几个细节值得看。

**第一个细节：model_forward 的均值在 batch=4 和 batch=8 之间几乎相同（17.95ms vs 17.89ms，差 0.06ms）。**

这说明 GPU 在 batch=4 时就已经接近吞吐上限。batch 从 4 增加到 8，每个 decode 步骤的等待时间没有明显增加——这是 batch 对 throughput 有效但对单步延迟几乎无效的直接证据。

对应 benchmark 数据：Phase 3 batch=4 是 194.2 tok/s，batch=8 是 361.3 tok/s。两者的比值约为 1.86×，非线性增长——就是因为 GPU 在 batch=4 时已经高效运转，batch=8 能做到接近 2× 但不完全，恰好体现了 GPU 从半饱和到接近饱和的过渡。

**第二个细节：batch=1 和 batch=4 的 model_forward 时间差不多（17.27ms vs 17.95ms），但 throughput 相差 3.6×（54 vs 194 tok/s）。**

这是因为 batch=1 的 decode 是矩阵-向量乘（GEMV），batch=4 是矩阵-矩阵乘（GEMM）。从 profiler 的完整 top-20 可以看到：

- batch=1：`gemvx_kernel` 主导，占 89.5% CUDA 时间
- batch=4/8：`cutlass WMMA tensorop f16` 主导，占 89-90% CUDA 时间

GEMV 和 GEMM 对 GPU 的利用方式完全不同。GEMV 受限于显存带宽（把模型权重从 HBM 读出来），而 GEMM 可以更好地利用 CUDA core 的计算吞吐。这解释了为什么 batch 从 1 → 4 的 throughput 提升（3.6×）远大于 batch 从 4 → 8（1.86×）：前者是从 GEMV 到 GEMM 的质变，后者是同一 GEMM 内的效率提升。

FlashAttention FMHA kernel（`fmha_cutlassF_f16_aligned_64x128_rf_sm80`）只在 batch≥4 时出现，batch=1 时注意力退化为简单的点乘，不走 FlashAttention 路径。

---

## 四、gather 降到 1.7%，已经不是瓶颈了

Phase 2 的 `gather_batch_kv` 是 3 层 Python 嵌套循环（layers × batch × seq_len）。以 batch=8、seq_len=128 为例，是 28 × 8 × 128 = 28672 次标量 GPU 索引，实测下来每步约占 decode 总时间的一半。

Phase 3 将其替换为 PyTorch advanced indexing——仍然有 `for l in range(28)` 的层循环，但层内部是一次 `k_cache[l][phys_blocks, slot_indices]`，对应一次 CUDA kernel 调用。

现在 profiler 给出：batch=8 的 gather 是 **0.314ms/step**，占三段合计的 **1.7%**。

更能说明问题的数据：batch=4 vs batch=8 的 gather 时间几乎没变（0.323ms vs 0.314ms）。向量化后的 gather 对 batch 的伸缩性很好，加倍请求数只让 CUDA kernel 处理的 tensor 稍大一点。

对比 Phase 2 的状况：那时 gather 是瓶颈，是因为 Python 层面的调度开销，而不是 GPU 的计算开销。消除 Python 循环之后，gather 的代价回落到应有的水平——0.3ms 的 tensor indexing，而不是几十毫秒的 Python 调度。

---

## 五、还差 12% 在哪里

写到这里，profiling 的数字逻辑应该是闭合的：gather 1.7%，model_forward 97%，write_kv 1%，合计 ~100%，性能接近 HF 了。但事实上 mini-infer batch=8 是 361 tok/s，HF 是 409 tok/s，差距 12%。

这不矛盾，但需要一点计算来理解。

throughput = (batch × 1) / time_per_step

- mini-infer batch=8：361 tok/s → time_per_step ≈ 8/361 = 22.2ms
- HF batch=8：409 tok/s → time_per_step ≈ 8/409 = 19.6ms

但我们的 profiler 显示三段合计只有 534ms / 29 = **18.4ms**。总时间 22.2ms 和标注段 18.4ms 差了 **3.8ms**，这部分是什么？

### 三段之外的 3.8ms

主要来自两处：

**1. DynamicCache 预填充（gather 和 model_forward 之间）**

decode_batch 在调用 model 之前，需要把 gathered KV 装入 DynamicCache：

```python
cache = DynamicCache()
for l in range(num_layers):  # 28 次
    cache.update(k_batch[l], v_batch[l], l)
```

`DynamicCache.update()` 内部会调用 `torch.cat` 把旧 KV 和新 KV 拼接。这发生在 `gather_batch_kv` 和 `model_forward` 之间，没有 record_function 标签，因此不计入三段合计。

在 profiler 的 top-20 里，`aten::cat` 在 batch=8 时总计 9.17ms，每步约 0.32ms。这只是 cat 调用本身，不包含 DynamicCache 构造时的 tensor 分配和 Python 循环开销。

**2. attention_mask 构造、input_ids 构造、logits 采样**

这些 CPU 操作本身很快，但会造成 GPU-CPU 同步点，在总时间上留下痕迹。

### HF 为什么没有这些开销

HF 在 prefill 之后直接持有 DynamicCache 对象，后续每个 decode 步骤只在 model.forward() 内部对当前层做一次 cat（append 新 token）。不需要 gather，不需要预填充——past_key_values 一直是同一个对象在原地更新。

mini-infer 使用 Paged KV Cache，KV 存储在 block tensor 里（`k_cache[layer][block_id, slot_id, ...]`），每个 decode 步都需要把分散的 block 重新 gather 成 dense tensor，然后包装成 DynamicCache 传给模型。这是 paged 存储和 HF 接口之间的"阻抗不匹配"——paged 格式是为了高效管理显存碎片，但目前的 HF 模型 forward 接口不支持直接从 paged 格式读取 KV。

**所以剩余 12% 的差距，是 paged KV cache 的结构性代价**，不是某个可以轻松消除的局部 bug：

| 代价来源 | 粗略估计（ms/step, batch=8） | 备注 |
|---------|--------------------------|------|
| gather_batch_kv | 0.31 | 向量化后已很小 |
| DynamicCache 预填充（cat） | ~0.3-0.5 | 28层 × aten::cat |
| 其他 CPU/同步开销 | ~3.0 | attention_mask、input_ids、采样等 |
| **合计** | **~3.8** | vs HF 的约 2.6ms 差距 |

消除这层代价的正确路径是：让 attention kernel 直接接受 block_tables 作为输入，在 attention 计算内部按 block 地址寻址 KV——这正是 vLLM 使用 flash_attn 2.5+ PagedAttention kernel 所做的事。当前 mini-infer 使用的是 flash_attn 2.3.6，不支持这个接口。

---

## 六、一个工程细节

profiling 脚本本身有一个低级 bug 被 review 抓出来：

```python
# 错误：EngineConfig 没有 max_num_blocks 字段
config = EngineConfig(
    model_name=args.model,
    max_num_blocks=2048,  # ← 拼写错误
    ...
)
```

`EngineConfig` 使用 `@dataclass(slots=True)`，不接受未声明的 keyword argument，运行时立即报 `TypeError`。正确字段名是 `num_gpu_blocks`。

这种 bug 在没有单元测试覆盖的 benchmark 脚本里很常见：script 从来没有在无 GPU 环境下跑通，字段名拼写错误只有在真正执行时才暴露。教训：即使是 benchmark 脚本，至少应该能在 `dry_run=True` 模式下跑通 EngineConfig 构造。

---

## 七、小结

Phase 5 用 profiler 量化了 decode_batch 的内部结构：

- **gather_batch_kv 已经不是瓶颈（1.5-1.7%）**：Phase 3 向量化的效果得到数字确认，0.27-0.32ms/step，batch 扩展无增量。
- **model_forward 占 97-98%，且 batch=4→8 时间几乎不变**：GPU 在 batch=4 时接近饱和，增加 batch 对单步延迟无影响，对吞吐有正收益但次线性。
- **剩余 12% 差距的根因是 paged 格式和 HF 接口的阻抗不匹配**：gather 只占 1.7%，但 gather 背后的 DynamicCache 预填充、KV 从 block tensor 到 dense tensor 的物理拷贝，合计消耗约 3-4ms/step，这是 paged KV cache 在当前接口下不可避免的代价。

从更大的视角看，Phase 3 把性能损失从"Python 解释器是瓶颈"（Phase 2：2× 慢）推进到"paged→dense 转换是代价"（Phase 3：12% 慢）。每次优化都在压缩瓶颈的量级，直到剩下的差距需要改变底层 kernel 接口才能消除——那就是 PagedAttention 的真正意义：attention kernel 原生支持非连续 KV 地址，而不是在 kernel 外面做一次 gather。
