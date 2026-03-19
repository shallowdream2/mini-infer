# Phase 2 开发日志（2026-03-19）

这个文件记录 mini-infer Phase 2（Paged KV Cache + Continuous Batching + Batch Decode）的开发过程，包括关键决策、卡点和解决方式。

---

## 上午：设计与规划

- 执行 `infer-plan`，明确 Phase 2 三个核心目标：Paged KV Cache、Batch Decode、Continuous Batching
- 确认模型架构参数：从 Qwen2.5-7B-Instruct 的 `config.json` 读出 `num_hidden_layers=28`、`num_key_value_heads=4`、`head_dim=128`
- 规划 KV block tensor 存储格式：`k_cache[l][num_gpu_blocks, block_size, num_kv_heads, head_dim]`
- 规划 `gather_batch_kv()` 的左填充策略

## 上午：实现 kv_cache.py

- 完整重写 `kv_cache.py`：预分配 GPU block tensor 池，BlockTable per request，FreeBlockPool deque
- 实现 `init_request()`：分配 `ceil(prompt_len/block_size)` 个物理块，设 `_seq_lens[rid] = prompt_len`
- 实现 `write_prefill_kv()`：把 HF prefill 输出的 tuple `past_key_values` 写入 block tensor，添加 layer count 和 seq_len 一致性断言
- 实现 `gather_batch_kv()`：左填充聚合所有请求 KV 到 dense tensor，返回 `k_batch, v_batch, seq_lens`
- 实现 `write_decode_kv()`：写新 token KV 到对应 slot，按需分配新块，递增 `_seq_lens`
- 实现 `free_request()`：归还所有物理块

## 下午：实现 model_runner.py 和 engine.py

- 完整重写 `model_runner.py`：
  - `prefill()` 调用 `kv_cache.write_prefill_kv()`
  - `decode_batch()` 实现 batch forward：gather → past_kv 构造 → attn_mask（左填充）→ forward → 提取 logits 和新 KV → write_decode_kv → del 大张量
  - 修复 `eos_token_id or 0` 的 falsy bug → `if eos_id is not None else -1`
  - 修复模型加载：`device_map=config.device` 替代 CPU 加载再 `.to(device)`
- 完整重写 `engine.py`：
  - Continuous batching 主循环：admit → prefill → decode_batch → cleanup
  - 新增 OOM 快速失败逻辑（避免死循环）
  - 新增 try/except 确保异常时归还 KV 块

## 卡点 1：无限循环 bug

- **问题**：`num_running==0` 时，KV 块不足导致 break，但下次迭代还是同样状态，死循环
- **发现方式**：code review 阶段静态分析发现
- **修复**：`if num_running == 0 and not newly_admitted: raise RuntimeError(...)`
- **验证**：新增测试 `test_engine_oom_raises_not_loops`，dry_run + 极小 block pool 触发

## 卡点 2：num_kv_heads 错误（8 → 4）

- **问题**：Phase 1 知识笔记记错了，以为 Qwen2.5-7B 有 8 个 KV heads，实际是 4
- **发现方式**：运行 benchmark 时 shape mismatch crash：`The expanded size of the tensor (8) must match the existing size (4)`
- **修复**：`config.py` 默认值 `num_kv_heads: int = 4`；`benchmark_mini.py` 同步修改
- **教训**：模型架构参数必须从 `config.json` 读，不能凭印象

## 卡点 3：DynamicCache 兼容性警告

- **问题**：transformers 4.43.4 已弃用 tuple 格式 `past_key_values`，会打印 warning
- **评估**：通过 `DynamicCache.from_legacy_cache` 自动转换，功能正常；下一版 transformers 可能 break
- **决策**：Phase 2 接受 warning，Phase 3 迁移 DynamicCache API

## 卡点 4：GPU 被 ollama 占用

- **问题**：GPU 0 被 ollama 进程占用约 9.7 GB，无法加载 Qwen2.5-7B（15.7 GB）
- **解决**：等待 ollama 释放，后续 GPU 0 空出，benchmark 在 GPU 0 正常运行

## 下午晚些：benchmark

- 跑 HF baseline：batch=1/4/8，各指标正常
- 跑 mini-infer Phase 2：batch=1/4/8，TTFT 与 HF 对齐，throughput 差距 12%–51%
- 主要瓶颈定位：`gather_batch_kv()` 每步从 block pool 复制 KV 到 dense tensor，O(batch × seq_len) 复制
- 数据落盘：`本地资料/实验记录/2026-03-19-phase2-benchmark.md`

## 当天完成的 skill 序列

infer-plan → infer-implement → infer-review → infer-implement（修复 review 问题）→ infer-benchmark → infer-summarize → infer-blog → infer-archive
