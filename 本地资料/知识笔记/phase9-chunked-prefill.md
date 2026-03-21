# Phase 9：Chunked Prefill 知识笔记

## 核心问题是什么

LLM 推理中 prefill 和 decode 有两种不同特征：
- **Prefill**：一次处理所有 prompt token，compute-bound，时间与 token 数成正比（1024 token 约 138ms）
- **Decode**：每步生成 1 个 token，memory-bound，时间基本固定（约 18ms/step）

当长 prompt 请求到达时，prefill 独占整步 GPU，正在 decode 的请求被完全阻塞。这就是 ITL spike。

## 解法：把大 prefill 切成小 chunk

每步只处理 `chunk_size` 个 token 的 prefill，剩余的 decode 请求依然运行。关键 trade-off：

- chunk_size 小 → ITL spike 低（平滑）→ 长请求 TTFT 增加多（需要更多步）
- chunk_size 大 → 长请求 TTFT 增加少 → ITL spike 降低效果差

经验值（1024-token prompt 场景）：chunk=256 是均衡点。

## 实现的关键数据结构

**PREFILLING 队列**：介于 WAITING 和 RUNNING 之间的独立状态。一次最多 1 个请求在此。

**DynamicCache 累积**：每个 chunk 调用 `model(input_ids=chunk_tokens, past_key_values=prev_cache)`，HF 的 DynamicCache 自动积累所有 chunk 的 KV。最后一个 chunk 完成后，把整个 DynamicCache 写入 block tensor（`write_prefill_kv`）。

**prefilled_tokens 字段**：记录已完成 prefill 的 token 数，供 engine 计算下一 chunk 的 `[t_start, t_end)`。

## 两个容易搞混的 ITL 指标

- **TTFT（Time To First Token）**：一个请求从提交到第一个 token 出来的时间。Chunked prefill 会让长请求的 TTFT 增加，这是预期行为。
- **ITL spike**：正在 decode 的短请求，因长请求 prefill 被阻塞的最大 token 间隔。这才是 chunked prefill 要降低的目标。

两个指标是 trade-off 关系：chunk 越小，ITL spike 越低，TTFT 越高。

## 和 vLLM 的差异

vLLM 的 chunked prefill：
- 允许多个请求同时 prefilling
- prefill tokens 和 decode tokens 打包进同一个 batch forward（一次前向）
- 用 `max_num_batched_tokens` 控制预算

mini-infer 的实现：
- 一次只有 1 个请求在 PREFILLING
- prefill_chunk 和 decode_batch 是分开的两次前向
- 简化了调度逻辑，但每步多一次模型前向

## 常见的测量口径陷阱

测量 max_itl_spike 时，必须全程连续记录 token 时间戳（不能只统计长请求到达后的间隔）。真正最大的间隔是"warmup 最后一个 token 到长请求 prefill 结束后第一个 token"，跨越了整个 prefill 过程。
