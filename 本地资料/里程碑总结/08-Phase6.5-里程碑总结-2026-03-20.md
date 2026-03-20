# Phase 6.5 里程碑总结（2026-03-20）

## 阶段名称

Phase 6.5：Triton decode attention kernel（实验性 kernel 开发，大厂核心路线主线）

---

## 一、阶段目标

在 mini-infer 中实现一个从零开始的 Triton decode attention kernel（`mini_infer/triton_attn.py`），与 flash_attn_with_kvcache 做直接对比，并通过 roofline 分析解释性能差距根因。

不要求超过 flash_attn，目标是走通 kernel 开发路径、量化性能差距、能解释清楚。

---

## 二、完成了什么

### 新增文件

| 文件 | 说明 |
|------|------|
| `mini_infer/triton_attn.py` | Triton JIT decode attention kernel + 三个 Python wrapper |
| `tests/test_triton_attn.py` | 8 个测试（2 个 dry-run + 6 个 GPU 正确性测试） |
| `benchmarks/benchmark_triton.py` | kernel 级 latency 对比 benchmark |
| `本地资料/实验记录/2026-03-20-phase6.5-benchmark.md` | 完整实验记录 |

### 核心实现

`_decode_attn_kernel`：Triton JIT kernel，设计要点：
- Grid = (batch, num_q_heads)，每个 program 处理一个 (batch, q_head) 对
- Online softmax（Milakov & Gimelshein 2018）：m_i/l_i/acc 状态跟踪，BLOCK_N=64 分块迭代
- GQA 映射：`kv_head_idx = q_head_idx * num_kv_heads // num_q_heads`
- HEAD_DIM=128 为 `tl.constexpr`（编译期固定），BLOCK_N=64 同样固定
- 内部 float32 计算，输入输出 float16

---

## 三、关键成就

### 验收标准达成情况

| 验收标准 | 状态 | 数据来源 |
|---------|------|---------|
| 1. 正确性：max_diff < 1e-2 vs flash_attn | ✅ 达成 | max_diff = 1.5e-5（实验记录） |
| 2. 覆盖场景：batch=1/8，GQA 28Q/4KV | ✅ 达成 | test_batch8、test_gqa_qwen_config 通过 |
| 3. 性能量化：triton.testing.do_bench，记录差距和 memory-bound 结论 | ✅ 达成 | 全配置延迟表（实验记录） |
| 4. 理解深度：roofline、根因分析（tile size 理由待补） | 🟡 部分达成 | roofline 和根因已记录；BLOCK_N=64 选择理由在 infer-blog 中补充 |
| 5. 不要求超越 flash_attn，能解释清楚 | ✅ 达成 | 差距已量化并有根因分析 |

### 性能关键数据（来自 `本地资料/实验记录/2026-03-20-phase6.5-benchmark.md`）

环境：RTX 4090，PyTorch 2.1.2+cu121，Triton 2.1.0，flash_attn 2.5.9.post1

| batch | seq_len | triton (μs) | flash (μs) | triton/flash |
|-------|---------|------------|-----------|-------------|
| 1 | 128 | 14.30 | 11.66 | **1.23×** |
| 1 | 2048 | 168.26 | 17.78 | **9.46×** |
| 8 | 128 | 20.83 | 12.54 | **1.66×** |
| 8 | 2048 | 182.84 | 60.01 | **3.05×** |

- 所有配置均为 memory-bound（AI ≈ 7 FLOPs/Byte << ridge point 82）
- seq_len=128 时差距仅 1.23×；seq_len 增大后差距扩大至 9.46×

### 技术亮点：解决 Triton 编译报错

遇到并解决了一个非显而易见的 Triton 编译器问题：

`tl.full([1], -1e38)` 产生 blocked encoding，`tl.max(scores, 0)` 返回 scalar encoding，两者 encoding 不匹配导致 `tl.maximum` 编译报错（`'triton_gpu.cmpf' op requires the same encoding`）。

修复：用 `scores[None, :]` 将 1D tensor 升维到 2D，再用 `tl.max(..., axis=1)` reduce 回 `[1]`，保持 blocked encoding 一致性。

---

## 四、遇到的问题

### 问题 1：`@triton.jit` 与 `python -c` 不兼容（Step 0 阶段）

`triton.jit` 使用 `inspect.getsource()` 获取 kernel 源代码，`python -c "..."` 没有源文件导致 OSError。解决方案：前置条件验证改写为临时文件运行。这是 triton 工程使用中的已知坑，不影响正式代码。

### 问题 2：Triton encoding 不匹配编译失败

如三节所述。核心教训：Triton 内部对 blocked tensor 和 scalar 有不同的 MLIR encoding，比较/算术操作要求 encoding 一致。`tl.full([1], ...)` 是 blocked，`tl.max(scores, 0)` 是 scalar，不能直接 `tl.maximum` 比较。

**验收标准中未达成的部分：**

BLOCK_N=64 tile size 的选择理由尚未写入独立技术笔记（infer-archive 阶段补充）。当前只在代码注释中提及"fits in registers"，缺少完整的 tile size 推导（register 限制 × 16KB per tile × float16 ≈ BLOCK_N=64 的依据）。

---

## 五、当前状态

| 阶段 | 状态 |
|------|------|
| Phase 1–6 | ✅ 已完成 |
| Phase 6.5 | 🔄 进行中（实现 + benchmark 完成，blog + archive 待做） |
| Phase 7 | 🔜 规划中 |

Phase 6.5 进度表：implement ✓，review ✓，benchmark ✓，summarize ✓，blog ⬜，archive ⬜。

---

## 六、下一阶段计划

**当前 Phase 6.5 剩余步骤：**
- `infer-blog`：写《用 Triton 写一个 decode attention kernel：从零到能跑》博客草稿
- `infer-archive`：归档本阶段产出，补充 BLOCK_N=64 tile size 技术笔记

**Phase 6.5 完成后，进入 Phase 7：Preemption + Priority Scheduling**（来自路线图）：
- 新增 `RequestStatus.SWAPPED` 状态
- `KVCacheManager.swap_out() / swap_in()`
- `Scheduler` 新增 `_swapped` 队列 + 优先级比较
- 验收：换出换入后 token-level 生成结果一致 + swap 实际延迟测量（CUDA event 计时）

---

## 七、可用于博客的主题

**博客 07（路线图规划）：《用 Triton 写一个 decode attention kernel：从零到能跑》**

关键素材：
- 问题背景：flash_attn 是黑盒，自己写 kernel 才能理解 memory-bound 的真实含义
- 设计过程：GQA mapping，online softmax，BLOCK_N 选择，stride 传递
- 遇到的坑：encoding 不匹配编译报错（`tl.maximum` 的 MLIR encoding 约束）
- 实验结果：seq_len=128 时仅 1.23× 差距，seq_len=2048 扩大到 9.46×
- Roofline 分析：AI=7 FLOPs/Byte，memory-bound，差距根因（无向量化、无 prefetch）
- 与 flash_attn 的本质差距：不是算法问题，是工程优化（向量化 load、prefetch pipeline、GQA 专项优化）
