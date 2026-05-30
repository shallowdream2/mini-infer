# Life cycle for the Mini-infer system
# 请求生命周期图

> 基于 mini-infer 源码精确标注，每步注明关键文件和函数。

---

## 全链路纵览

```
用户 / OpenAI Client
        │  POST /v1/chat/completions
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI HTTP Server          serving/server.py:chat_completions()  │
│  • 解析 ChatCompletionRequest（Pydantic）                            │
│  • 调用 engine.generate_stream(prompt, max_new_tokens)              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  await AsyncEngine.generate_stream()
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AsyncEngine              runtime/async_engine.py                   │
│  • add_request()：生成 request_id，创建 asyncio.Queue               │
│  • 后台线程持续调用 engine.step()（1 ms sleep 空转保护）              │
│  • 跨线程投递：loop.call_soon_threadsafe(queue.put_nowait, token)    │
│  • 前台 async for：await queue.get() → yield token                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  engine.add_request() / engine.step()
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LLMEngine                runtime/engine.py                         │
│                                                                     │
│  ┌── step() 每轮迭代 ──────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  1. 准入（Admission）                                        │   │
│  │     Scheduler.peek_next_waiting()                            │   │
│  │     KVCacheManager.num_free_blocks() >= 需求？               │   │
│  │       ├─ 是 → kv_cache.init_request(state)                  │   │
│  │       │        scheduler.add_to_running(state)              │   │
│  │       └─ 否 → 尝试抢占 lowest_priority_running              │   │
│  │                kv_cache.swap_out() → scheduler.mark_swapped()│   │
│  │                                                              │   │
│  │  2. Prefill                                                  │   │
│  │     model_runner.prefill(state)                              │   │
│  │     kv_cache.write_prefill_kv(request_id, past_key_values)  │   │
│  │     state.prefilled = True                                   │   │
│  │                                                              │   │
│  │  3. Decode Batch                                             │   │
│  │     kv_cache.ensure_next_slot()   ← 预分配下一 token 的块   │   │
│  │     block_tables, cache_seqlens = kv_cache.build_block_tables│   │
│  │     model_runner.decode_batch(states, block_tables, ...)     │   │
│  │     kv_cache.advance_seq_lens()   ← seq_len += 1            │   │
│  │                                                              │   │
│  │  4. Token 采样 + 完成检测                                    │   │
│  │     sampling_params.temperature / top_p                      │   │
│  │     EOS 或 max_new_tokens → scheduler.finish_request()       │   │
│  │                             kv_cache.free_request()          │   │
│  │                                                              │   │
│  │  5. Swap-in（如有）                                          │   │
│  │     kv_cache.num_free_blocks() 足够 →                        │   │
│  │     kv_cache.swap_in() + scheduler.move_swapped_to_running() │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  model_runner.prefill / decode_batch
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ModelRunner              modeling/model_runner.py                  │
│                                                                     │
│  prefill(state)                                                     │
│    • 调用 HF model.forward(input_ids)                               │
│    • 返回 past_key_values（DynamicCache）                            │
│    • ※ chunk_prefill_size > 0 时分多步执行，中间 cache 存            │
│       engine._prefilling_caches[rid]                                │
│                                                                     │
│  decode_batch(states, block_tables, cache_seqlens, kv_cache)        │
│    • patch_model_for_paged_decode() 已将 attention 层替换            │
│    • 底层调用 flash_attn_with_kvcache(                               │
│          q, k_cache, v_cache,                                       │
│          block_table=block_tables,    ← PagedAttention 核心         │
│          cache_seqlens=cache_seqlens                                │
│      )                                                              │
│    • CUDA Graph 模式：graph.replay() 替代逐层 forward               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  flash_attn_with_kvcache
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  KV Cache（GPU Block Pool）   cache/kv_cache.py                     │
│                                                                     │
│  物理布局：                                                          │
│    k_cache[layer]  shape: [num_gpu_blocks, block_size, num_kv_heads, head_dim]  │
│    v_cache[layer]  同上                                             │
│                                                                     │
│  逻辑→物理映射（BlockTable）：                                        │
│    请求 A：逻辑块 0 → 物理块 42                                      │
│    请求 B：逻辑块 0 → 物理块  7   ← 独立物理块，无冲突              │
│    Prefix：逻辑块 0 → 物理块 42   ← ref_count=2，共享同一物理块     │
│                                                                     │
│  FreeBlockPool（deque）：已回收的物理块 ID                           │
│  PrefixCacheManager（LRU OrderedDict）：                            │
│    key  = block-level SHA-256 链式 hash                             │
│    value = 物理块 ID                                                 │
│    淘汰条件：ref_count == 1（仅 cache 持有，无活跃请求）             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prefill vs Decode：为什么要区分？

| 维度 | Prefill | Decode |
|------|---------|--------|
| 输入形状 | `[1, prompt_len]`，每次不同 | `[batch_size, 1]`，shape 固定 |
| 计算特征 | compute-bound（大矩阵乘） | memory-bound（小矩阵 + KV 读取） |
| KV 写入 | 一次写入 `prompt_len` 个位置 | 每步写入 1 个位置 |
| CUDA Graph | 不适用（shape 动态） | 适用（per batch_size 捕获一张图） |
| Chunked Prefill | 将长 prompt 切成 chunks，每 chunk 一步 | 不影响 |

---

## Chunked Prefill 状态机（Phase 9）

```
WAITING
   │  有空闲块 + chunk_prefill_size > 0
   ▼
PREFILLING ──── chunk 未完成 ────→ 下一步继续 PREFILLING
   │
   │  最后一个 chunk 完成
   ▼
RUNNING（进入 decode_batch）
   │
   │  EOS / max_new_tokens
   ▼
FINISHED
```

调度器中对应的队列：`_waiting → _prefilling → _running`

---

## Preemption 路径（Phase 7）

```
新请求到来，空闲块不足
        │
        ▼
scheduler.get_lowest_priority_running()
        │  找到优先级最低的 running 请求 V
        ▼
kv_cache.swap_out(V)   ← KV blocks 序列化到 CPU
scheduler.mark_swapped(V)
        │
        ▼
重新为高优先级请求分配块并 prefill
        │  后续步：空闲块恢复
        ▼
kv_cache.swap_in(V)
scheduler.move_swapped_to_running(V)
```

---

## 关键文件索引

| 关注点 | 文件 | 核心函数 |
|--------|------|---------|
| HTTP 入口 | `serving/server.py` | `chat_completions()` |
| 异步调度桥 | `runtime/async_engine.py` | `generate_stream()`, `_step_loop()` |
| 主调度循环 | `runtime/engine.py` | `step()`, `generate()` |
| 请求队列 | `runtime/scheduler.py` | `add_request()`, `get_lowest_priority_running()` |
| KV 块管理 | `cache/kv_cache.py` | `init_request()`, `ensure_next_slot()`, `build_block_tables()` |
| 前缀缓存 | `cache/kv_cache.py` | `find_prefix_cache()`, `register_prefix_blocks()` |
| Prefill 执行 | `modeling/model_runner.py` | `prefill()` |
| Decode 执行 | `modeling/model_runner.py` | `decode_batch()` |
| Paged Attention | `kernels/attention.py` | `patch_model_for_paged_decode()` |