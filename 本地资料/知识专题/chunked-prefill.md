# 知识专题：Chunked Prefill

## 主题

Chunked Prefill：将长 prompt 的 prefill 拆成多个小 chunk 分步执行，防止长 prefill 阻塞并发 decode 请求。

---

## 一、问题定义

LLM 推理中存在两种计算模式：

**Prefill（预填充）**：处理用户输入的所有 prompt token，一次性计算所有 KV 并得到第一个生成 token。计算量 O(n²)，受 compute 限制，延迟随 prompt 长度增长显著。

**Decode（解码）**：每步生成一个新 token，需要读取所有历史 KV。计算量 O(n)，受 memory bandwidth 限制，每步延迟基本恒定（~18ms on RTX 4090 for Qwen2.5-7B）。

**核心问题**：Continuous batching 系统中，当一个长 prompt 请求到达时，其 prefill 需要独占完整的一步（~138ms for 1024 tokens）。期间所有正在 decode 的请求无法运行，造成它们的 ITL 出现 138ms 的尖峰——这是正常步长（~18ms）的 7.6 倍。

---

## 二、核心原理

**Chunked Prefill** 的核心思想：把一次大的 prefill 拆成多步小的 prefill，每步只处理 `chunk_size` 个 token，在两次 chunk 之间继续 decode 其他请求。

```
无分块（chunk_size=0）：
  步 1: [prefill 全部 1024 token] ← 138ms，decode 请求完全停止
  步 2: [decode]
  步 3: [decode]

有分块（chunk_size=256）：
  步 1: [decode] + [prefill token 0-256]    ← 约 58ms（含 chunk prefill）
  步 2: [decode] + [prefill token 256-512]  ← 约 58ms
  步 3: [decode] + [prefill token 512-768]  ← 约 58ms
  步 4: [decode] + [prefill token 768-1024] ← 约 58ms（最后 chunk，产出第一个 token）
```

**权衡**：
- 短请求的 ITL spike 从 138ms 降至每步约 58ms（−57%）
- 长请求的 TTFT 从 138ms 增至 270ms（多了 3 步 × 约 44ms 的 decode 时间）
- 总吞吐无回归（总计算量不变，只是分散了）

---

## 三、工程实现方式

### 状态机扩展

请求生命周期从 `WAITING → RUNNING → FINISHED` 扩展为：
```
WAITING → PREFILLING → RUNNING → FINISHED
```

PREFILLING 是独立队列（`_prefilling: dict[str, RequestState]`），一次最多 1 个请求。

### 跨 chunk KV 积累：DynamicCache 传递

每个 chunk 调用模型时，传入上一个 chunk 积累的 `DynamicCache`：

```python
out = model(input_ids=chunk_tokens, past_key_values=prev_cache, use_cache=True)
if is_last_chunk:
    kv_cache.write_prefill_kv(request_id, out.past_key_values)  # 写入 block tensor
    state.prefilled = True
else:
    _prefilling_caches[rid] = out.past_key_values  # 存 CPU 内存，供下次 chunk 使用
```

- 中间 DynamicCache 存在 CPU 内存（`_prefilling_caches` dict），不占 GPU 显存
- 最后一个 chunk 完成后才写入 GPU block tensor，对 decode 路径完全透明

### 进度追踪

`RequestState.prefilled_tokens: int = 0` 记录已完成的 token 数，引擎用它计算下一 chunk 范围：
```python
t_start = state.prefilled_tokens
t_end = min(t_start + chunk_prefill_size, len(state.prompt_token_ids))
is_last = (t_end == len(state.prompt_token_ids))
```

### 关键边界条件

最后一个 chunk 完成后，请求立即进入 RUNNING，并在**当步**参与 decode_batch。这要求在 `pre_lens` 计算时把该请求的起始长度置 0，避免错过 prefill 采样产出的第一个 token：

```python
for s in just_prefilled:
    pre_lens[s.request.request_id] = 0  # 从 0 开始捕获 prefill token
```

---

## 四、设计取舍

| 维度 | mini-infer 选择 | vLLM 选择 | 理由 |
|------|---------------|----------|------|
| 同时 PREFILLING 请求数 | 1 | 多个 | mini-infer 简化逻辑，不需要同步多个中间 cache |
| Prefill + decode 是否合并为一次前向 | 否（两次） | 是 | vLLM 将 prefill tokens 和 decode tokens 打包进同一 batch，更高效但更复杂 |
| 中间 cache 存储位置 | CPU 内存 | 通常 GPU | 中间状态不需要即时计算，CPU 存储更省 GPU 显存 |
| PREFILLING 是否可被抢占 | 否 | 支持（有代价） | 重置 DynamicCache + 退回 WAITING 代价高，mini-infer 暂不支持 |
| chunk_size 参数 | 运行时可配置 | `max_num_batched_tokens` | 效果等价 |

**chunk_size 选择建议**：
- 256：均衡，ITL spike −57%，TTFT 增加约 1 倍，推荐默认值
- 128：激进，ITL spike −67%，TTFT 增加约 2.9 倍，用于对 ITL 极度敏感场景
- 0（禁用）：Phase 8 行为，不引入任何延迟开销

---

## 五、常见误区

**误区 1：Chunked prefill 会降低长请求的 TTFT**
实际相反，TTFT 必然增加。Chunked prefill 优化的是并发 decode 请求的 ITL，不是自身的 TTFT。

**误区 2：ITL spike 只看长请求到达后的间隔**
最大 ITL 通常是"warmup 最后一个 token 到长请求 prefill 结束后第一个 token"的间隔，横跨整个 prefill 过程。只统计到达后的间隔会低估实际 spike。

**误区 3：Chunk overhead 会降低吞吐**
实测吞吐无明显回归（+2-4% 属测量波动）。总计算量不变，只是分散了计算时机。

**误区 4：中间 DynamicCache 会消耗大量内存**
一个请求的中间 DynamicCache 约 = `num_layers × 2 × seq_len × num_kv_heads × head_dim × dtype_bytes`，对 Qwen2.5-7B（28层，4头，128维，fp16）和 1024 token prompt 约 ~28 × 2 × 1024 × 4 × 128 × 2 ≈ 58 MB CPU 内存，可接受。

---

## 六、和 mini-infer 的关系

Phase 9 在 Phase 8（HTTP API + AsyncEngine）基础上扩展了调度层。`chunk_prefill_size=0` 时行为与 Phase 8 完全一致，不破坏已有接口。

涉及文件：
- `config.py`：新增 `chunk_prefill_size` 配置项
- `request.py`：新增 `prefilled_tokens` 追踪字段
- `scheduler.py`：新增 PREFILLING 队列和相关方法
- `model_runner.py`：新增 `prefill_chunk()` 方法
- `engine.py`：`generate()` 和 `step()` 增加 chunked 分支

实测结果（RTX 4090，Qwen2.5-7B-Instruct，7 并发 decode + 1 个 1024-token 长请求）：
- chunk=256：ITL spike 138ms → 59ms（−57.5%），TTFT 138ms → 270ms，吞吐无回归
- chunk=128：ITL spike 138ms → 46ms（−66.8%），TTFT 138ms → 399ms，吞吐无回归

---

## 七、进一步阅读

- vLLM 论文（Kwon et al., 2023）：Section 3.3 介绍了 continuous batching 下的 preemption；chunked prefill 是后续工程演进
- Sarathi（2023）：专门讨论 chunked prefill + decode 合并调度的论文，量化了吞吐和延迟的权衡
- vLLM 源码 `vllm/core/scheduler.py`：`_schedule_prefills()` 的 `chunked_prefill_enabled` 分支，实现了多请求并发 prefilling
