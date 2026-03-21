# Phase 9 开发日志：Chunked Prefill

**日期**：2026-03-21
**阶段**：Phase 9 — Chunked Prefill

---

## 做了什么

### 阶段背景

Phase 8 完成 HTTP API 后，发现服务化场景中存在一个调度层问题：长 prompt 请求（1024 token）到达时，引擎在当步独占 GPU 做完整 prefill（约 138ms），期间所有正在 decode 的短请求无法生成 token，造成 ITL spike。

Phase 9 目标：把长 prefill 拆成多个小 chunk，每步只处理 `chunk_prefill_size` 个 token，decode 请求在 chunk 之间持续运行。

### 实现步骤

1. **config.py**：新增 `chunk_prefill_size: int = 0` 字段，0 表示禁用（向后兼容），`__post_init__` 加非负校验。

2. **request.py**：新增 `prefilled_tokens: int = 0`，记录已完成 prefill 的 token 数，供 engine 计算下一 chunk 的 `[t_start, t_end)`。

3. **scheduler.py**：新增 `_prefilling: dict[str, RequestState]` 队列和 7 个操作方法。关键设计：一次最多 1 个请求在 PREFILLING，调度器不负责 DynamicCache 管理（那是 engine 的职责）。

4. **model_runner.py**：新增 `prefill_chunk()` 方法。
   - 非最后 chunk：`model(past_key_values=accumulated_cache)` → 返回更新后的 DynamicCache
   - 最后 chunk：`write_prefill_kv` 写入 block tensor → 采样第一个 token → `prefilled=True` → 返回 None
   - dry_run 路径：只更新 `prefilled_tokens`，最后 chunk 时生成 token [1]

5. **engine.py**：`generate()` 和 `step()` 均增加 `chunk_prefill_size > 0` 分支；`_prefilling_caches: dict[str, object]` 保存中间 DynamicCache；引入 `just_prefilled` 统一两条路径的"刚完成 prefill"捕获逻辑；异常路径释放 PREFILLING 的 GPU 块。

6. **tests/test_chunked_prefill.py**：7 个 dry_run 测试，全部通过。

7. **benchmarks/benchmark_chunked_prefill.py**：基于 `add_request/step()` 接口的 ITL spike 测量脚本（经三轮返工最终正确）。

---

## 卡在哪里

### 问题 1：`UnboundLocalError: newly_admitted`

`step()` 的 chunked 分支没有定义 `newly_admitted`，但后续 `pre_lens` 逻辑引用了它。修复：引入 `just_prefilled: list[RequestState] = []`，chunked 路径 append，原始路径 `just_prefilled = newly_admitted`。

### 问题 2：Benchmark 设计三次返工

**第一次**：7 个短请求也在 WAITING，长请求先占 PREFILLING，短请求反而被阻塞更久。修复：重新设计为 warmup 阶段先让短请求完成 prefill 进入 RUNNING，再提交长请求。

**第二次**：TTFT 时间戳记录在 `step()` 调用前，所有 TTFT ≈ 0ms。修复：时间戳移到 `step()` 返回后。

**第三次**：max_itl 只统计 phase 3（long 到达后）的 token 间隔，遗漏了跨 warmup 阶段的最大间隔（正是实际 spike）。修复：全程记录时间戳，计算包含 `t_long_start` 跨越间隔的最大值。

---

## 关键决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 同时允许几个 PREFILLING | 仅 1 个 | 逻辑简单，多个同时 prefill 需要同时维护多个中间 DynamicCache |
| 中间 DynamicCache 存哪里 | CPU 内存（`_prefilling_caches`） | 不需要立即用于 GPU，不占 GPU 显存 |
| PREFILLING 完成后当步 decode 还是下一步 | 当步 | 减少延迟；`move_prefilling_to_running` 后直接参与 decode_batch |
| PREFILLING 是否可以被抢占 | 否 | 重置 DynamicCache 代价高，当前实现不支持 |

---

## GPU 验证结果

chunk=0/128/256 三组对相同 prompt 输出完全一致，验证了 DynamicCache 跨 chunk 累积后 `write_prefill_kv` 的数值正确性。

Benchmark（RTX 4090，Qwen2.5-7B-Instruct）：
- chunk=256：ITL spike 138→59ms（−57%），TTFT 138→270ms（+1×），吞吐 +4%
- chunk=128：ITL spike 138→46ms（−67%），TTFT 138→399ms（+3×），吞吐 +2%

---

## 遗留技术债

- `swap_in` 检查未计入 `num_prefilling()`（preemption + chunked prefill 同时活跃时可能短暂超出 max_batch_size）
- `scheduler.remove_from_prefilling()` 是死代码，若将来被调用不会重置 `prefilled_tokens`
