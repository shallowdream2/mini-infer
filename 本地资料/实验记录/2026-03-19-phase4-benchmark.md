# Phase 4 Benchmark 实验记录

本文件记录 mini-infer Phase 4（双卡扩展）的真实性能数据，对比三种配置：single（单卡）、replica（双卡数据并行）、tp2（HF Pipeline Parallel）。

## 实验环境

- 硬件：Ubuntu 24.04 + RTX 4090 × 2（各 24 GB），双卡 P2P 驱动旧版警告（不影响本次结果）
- 模型：Qwen/Qwen2.5-7B-Instruct，float16
- 模型路径：~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct（本地，HF_HUB_OFFLINE=1）
- Python：3.10.19，transformers 4.43.4，torch 2.1.2，accelerate（HF device_map 依赖）
- 环境：conda ai-infra

## 测试命令

```bash
MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct

# single
HF_HUB_OFFLINE=1 python benchmarks/benchmark_multi_gpu.py \
  --model $MODEL --mode single --batch-size 8 --max-new-tokens 128

# replica
HF_HUB_OFFLINE=1 python benchmarks/benchmark_multi_gpu.py \
  --model $MODEL --mode replica --batch-size 8 --max-new-tokens 128

# tp2
HF_HUB_OFFLINE=1 python benchmarks/benchmark_multi_gpu.py \
  --model $MODEL --mode tp2 --batch-size 8 --max-new-tokens 128
```

## 数据口径

| 指标 | 口径 |
|------|------|
| Throughput | 输出 token 总数（decode 后重新 tokenize）/ 总耗时 |
| TTFT | engine.generate(max_new_tokens=1) 端到端耗时（single/replica）；tp2 为 model.generate(max_new_tokens=1) |
| Peak Mem | torch.cuda.max_memory_allocated()（GB），主 benchmark 前 reset |
| 热身 | single/tp2：1 条请求 4 token；replica：2 条请求 4 token |

注：TTFT 是近似值，包含 tokenization、engine 初始化后首次调用的一次性开销。tp2 的 TTFT 还包含 HF sampling loop overhead，与 single/replica 不完全可比。

## benchmark 结果（batch=8，max_new_tokens=128）

| 模式 | Throughput | TTFT（近似） | Peak Mem GPU0 | Peak Mem GPU1 |
|------|-----------|-------------|---------------|---------------|
| single（Phase 3 基线） | 361.4 tok/s | 19.4 ms | 16.42 GB | — |
| replica（双卡数据并行） | 376.1 tok/s | 23.1 ms | 16.31 GB | 16.31 GB |
| tp2（HF Pipeline Parallel） | 361.5 tok/s | 21.4 ms | 7.00 GB | 8.97 GB |

## 分析

### Replica（数据并行）

batch=8 分到两卡各 4 条请求，吞吐从 361.4 → 376.1 tok/s，仅 **+4.1%**，远低于理论 ~2×。

根因：

1. **每卡 batch=4 的效率低于单卡 batch=8**。Phase 3 数据：单卡 batch=4 = 194.2 tok/s，单卡 batch=8 = 361.3 tok/s。两卡各跑 batch=4 理论上限为 388.4 tok/s（与 Phase 3 batch=4 的 2× 吻合），但单卡 batch=8 已经接近这个上限（93%），所以 replica 的收益空间很小。

2. **Replica 适合总 batch 远大于单卡最优 batch 的场景**。如果总请求量是 32 条（每卡 16 条），单卡 batch=32 可能因 KV cache 显存不足而无法运行，replica 就有实质收益。在 batch=8 的测试规模下，单卡仍能高效运行，replica 没有发挥空间。

3. **Python GIL 对两线程并发的影响**。ThreadPoolExecutor 并发两个引擎，CUDA 操作在 GPU 上并行，但 Python 层面（tokenize、调度循环等）受 GIL 序列化。实测两卡总 throughput 376.1 tok/s 约等于单卡 batch=4 的 2× (194.2×2=388.4)，说明 GPU 端确实在并行，只是 batch=4 本身不如 batch=8 高效。

### tp2（HF Pipeline Parallel）

吞吐与单卡几乎相同（361.5 vs 361.4），**主要收益是显存分布**：

- 单卡：16.42 GB 集中在 GPU0
- tp2：7.00 GB（GPU0）+ 8.97 GB（GPU1）= 15.97 GB，每卡约减半

对于 Qwen2.5-7B 这种单卡能放下的模型，tp2 没有吞吐收益。真正的价值场景是：70B/72B 级别的模型，单卡 24 GB 放不下，用 device_map="balanced" 可以在双卡上跑。

tp2 吞吐不提升的原因：Pipeline Parallel 是层间串行流水——GPU0 跑完前 14 层，输出激活传给 GPU1 跑后 14 层，总计算量不变，只是分在两卡上。对于 batch=8 的批推理，这是纯粹的时序串行，没有并行效益。

### 关键警告记录

1. **驱动 P2P 警告**：`We've detected an older driver with an RTX 4000 series GPU. These drivers have issues with P2P.` accelerate 多卡推理在旧驱动下可能有性能损耗。当前结果仍有意义，但建议升级驱动后复测。

2. **tp2 右填充警告**：`right-padding was detected! For correct generation results, please set padding_side='left'`. TPEngine 使用默认右填充，对 decoder-only 模型的生成质量有影响（注意力掩码位置偏移），但 throughput 计数基于 output_ids token 数，不受此影响。

## 结论

| 场景 | 推荐策略 |
|------|---------|
| 模型能装单卡，追求高吞吐 | **Replica**：有效，但需要总 batch 够大（每卡 batch ≥ 8）才能显现 ~2× 收益 |
| 模型装不下单卡 | **TP / PP（device_map）**：显存折半，维持吞吐 |
| batch=8 规模 | Single 与 Replica/TP 差距极小，单卡更省资源 |

## 局限性

- 测试规模偏小（batch=8），未验证大 batch（≥16）下 replica 的真实 2× 收益
- tp2 的右填充问题未修复，生成文本质量未验证
- P2P 驱动旧版未升级，多卡通信可能有额外延迟
- TTFT 是 amortized 端到端，非严格的单请求 p50 延迟
- 未测 batch=1 的延迟场景（latency-critical 场景下各策略差异更明显）
