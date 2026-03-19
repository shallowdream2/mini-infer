# Phase 5 开发日志（2026-03-19）

本文件记录 mini-infer Phase 5（Profiling + 项目收尾）的开发过程、关键决策和遇到的问题。

---

## 任务背景

Phase 4 完成后项目核心实现已全部落地，Phase 5 定位是收尾：量化 decode_batch 内部各阶段的时间占比，并将 README 从"骨架说明"更新为实际状态。

---

## 过程记录

### 1. record_function 标签选位

在 `model_runner.py` 的 `decode_batch()` 真实路径中需要给三段加标签：gather_batch_kv、model_forward、write_decode_kv。

考虑了两种方案：
- 用 CUDA Event 手动计时（输出更清晰，不依赖 profiler）
- 用 `torch.profiler.record_function`（标准做法，profiler 未激活时 no-op，不影响正常性能）

选 record_function：它 no-op 特性更干净，且与 torch.profiler 生态集成，可以直接从 profiler 输出提取。

### 2. profile_decode.py 中的字段名拼写错误

写 EngineConfig 时把 `num_gpu_blocks` 打成了 `max_num_blocks`。由于是 `@dataclass(slots=True)`，运行时直接 TypeError。被 infer-review 发现，在 infer-implement 中修复。

教训：benchmark 脚本没有干跑验证，EngineConfig 构造错误只在真实 GPU 运行时才暴露。

### 3. Profiling 数据分析过程

运行 batch=1/4/8 三组后，发现：

**最重要的数字**：model_forward 在 batch=4 和 batch=8 时几乎相同（17.95ms vs 17.89ms）。这直接说明 GPU 在 batch=4 就已接近饱和，batch=8 的额外请求几乎不增加单步等待时间。

**gather 占比**：0.27-0.32ms/step，三种 batch 差异极小。Phase 3 向量化效果得到量化确认。

**kernel 变化**：batch=1 的 gemvx（GEMV）和 batch≥4 的 cutlass WMMA（GEMM）是本质不同的计算模式，这解释了为什么 batch 1→4 的 throughput 提升（3.6×）远大于 4→8（1.86×）。

**12% 差距溯源**：三段合计 18.4ms，但 profiler 总 CUDA 时间 22.2ms。差了 3.8ms，来自 DynamicCache 预填充（28 层 cat 调用）和其他 CPU 操作。这揭示了 paged → dense 格式转换的真实代价不只在 gather，还在后续的 DynamicCache 构造。

### 4. README 重写

旧 README 开头是"当前仓库仍处于初级骨架阶段"，正文说"当前代码不能代表已经完成的 PagedAttention..."。Phase 5 将其全部更新：加入 4 阶段性能数据表、双卡对比表、架构说明、profiling 工具使用方法。

---

## 关键决策

- `record_function` 选 no-op 方案，不影响正常推理
- `profile_decode.py` 预热后再 profiling，避免冷启动干扰
- profiler context 覆盖 prefill + decode，但只有 decode 路径有标签——docstring 中说明这个区别

---

## 残余问题

1. profile_decode.py 没有 dry_run 验证路径，字段名拼写错误在 review 前完全未发现
2. profiling 只覆盖了前 30 步 decode；更长序列（KV 增长后）gather 开销的变化未测
3. "12% 差距"的 3.8ms 分解是事后推算，没有对 DynamicCache 构造阶段独立打标签验证
