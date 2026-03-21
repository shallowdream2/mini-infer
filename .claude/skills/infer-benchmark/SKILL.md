---
name: infer-benchmark
description: 为 mini-infer 设计或执行 benchmark，并输出统一口径的结果说明。
argument-hint: [benchmark 任务]
---

# infer-benchmark

这个文件用于 benchmark 设计、执行和结果整理。

**执行前检查（硬性门控）：** 确认 `CLAUDE.md` 进度表中 `infer-implement` 和 `infer-review` 均为 ✓。若未满足，拒绝执行并告知用户缺少哪一步。

把附加文本视为 benchmark 任务，并遵守这些要求：

- 先说明对象：HF baseline 或 mini-infer 当前实现
- 说明 workload：模型、batch size、prompt 长度、output 长度、并发数
- 至少关注 `throughput`、`TTFT`、`TPOT`、`peak memory`
- 若某个指标在当前脚本中不可得、仅近似可得，或与其他 benchmark 的测量口径不同，必须显式标注 `N/A`、`近似` 或差异说明
- 输出环境、命令、数据口径、结果、结论和局限性
- 没有真实数据时，不输出性能结论

需要指标定义时，优先参考同目录下的 `METRICS_SPEC.md`。

## 性能优化阶段的附加要求

如果本阶段包含性能优化（如 gather 向量化、kernel 替换、调度改进等），benchmark 必须额外包含：

1. **优化前后的时间分布对比**（不只是 throughput 数字）：用 `benchmarks/profile_decode.py` 或等效工具，确认优化命中了目标瓶颈
2. **量化优化效果**：被优化的操作占总 CUDA 时间的比例（优化前 vs 优化后），防止优化了非瓶颈导致收益为零
3. **说明剩余差距的根因**：不仅报告"还差 X%"，还要说明差距主要来自哪里

**理由**：Phase 3 向量化 gather 后达到 88.4% HF，但"剩余 12% 在哪里"直到 Phase 5 才量化。每个性能优化阶段应当即时回答这个问题，而不是留给后续 phase 补测。

## 涉及 attention 替换时的附加要求（Phase 6 及类似阶段）

如果本阶段替换了 attention 实现（如从 DynamicCache 切换到 flash_attn block_tables），benchmark 还必须包含：

### 数值一致性验证（先做，再跑性能）

```python
# 对相同 prompt，对比 Phase 5 输出和 Phase 6 输出的 token 序列
# 允许存在少量不同（flash_attn fp16 精度差异），但整体 coherence 必须保持
# 验证指标：token_match_rate > 95%（前 N 个 token 中相同比例）
```

新实现的 attention 输出与原实现输出的 max_diff 必须在合理范围（< 1e-2 for fp16）。

### Profiler 消失验证

用 `profile_decode.py` 确认被删除的操作（gather_kv、write_kv、DynamicCache）不再出现在 CUDA 时间中：

```bash
python benchmarks/profile_decode.py --batch_size 8 --decode_steps 30
# 期望：gather_batch_kv CUDA time → 0，write_decode_kv CUDA time → 0
```

### 与 HF baseline 的对比维度

attention 替换后，需要和 HF baseline 在**相同 workload**下直接对比：

| 指标 | Phase 5 mini-infer | Phase 6 mini-infer | HF baseline |
|------|------------------|------------------|-------------|
| throughput (tok/s) | X | ? | X_hf |
| TTFT (ms) | X | ? | X_hf |
| model_forward CUDA (ms/step) | ~17.9 | ? | — |
| gather_kv CUDA (ms/step) | ~0.31 | 期望 ~0 | — |

## 比较性 profiling（两个实现对比时）

当需要对比 mini-infer 和 HF baseline 的内部 kernel 时：

```python
# benchmark_hf.py 目前没有 profiling 支持
# 对 HF baseline 添加 profiling 的方式：
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    out = model.generate(**inputs, max_new_tokens=30)
    torch.cuda.synchronize()

# 比较两个 prof.key_averages() 的结果
# 重点关注：aten::cat 的次数和时间（DynamicCache overhead）
# flash_sdp_forward / scaled_dot_product_attention 的时间
```

这样可以量化 HF baseline 的 DynamicCache 开销，与 mini-infer Phase 6 对比。

## benchmark 结果异常时的处理规程

如果 benchmark 结果明显不合理（相比上一阶段倒退超过 10%，或相比 HF baseline 低于 50%），**不得继续运行或得出结论**，立即执行：

1. **诊断时间分布**：运行 `profile_decode.py`，找到哪个操作的时间异常（比上一阶段慢 2× 以上）
2. **对照 review 问题列表**：检查是否有已知的性能问题未被修复（如 CPU-GPU sync、per-layer `.item()`）
3. **返回 infer-implement**：告知用户 benchmark 结果异常，建议先修复根因
4. **根因修复后重跑**：修复后重新执行 benchmark，以新结果为准

典型症状与根因对照：

| 现象 | 可能根因 |
|------|---------|
| model_forward 耗时 = N × ~18ms（N = 层数）| 每层各做一次 `.item()` 触发 CPU-GPU sync |
| throughput < 10% HF baseline，但测试通过 | 性能 bug（非正确性 bug），review 可能漏过 |
| 性能比上阶段差但无 error | 新代码路径引入了隐式 sync 或多余拷贝 |

## 完成后

以上所有要求（包括专项验证）**全部完成**后，将 benchmark 结果写入 `本地资料/实验记录/`，并在 `CLAUDE.md` 的当前阶段进度表中将 `infer-benchmark` 步骤标为 ✓。
