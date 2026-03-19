# 自己实现 Paged KV Cache 和 Batch Decode：从串行推理到 mini-vLLM

这篇文章是 mini-infer 项目的第二篇技术复盘。Phase 1 跑通了最小串行推理链路；Phase 2 的目标是实现工业级推理引擎的三个核心机制：Paged KV Cache、Batch Decode 和 Continuous Batching。

---

## 一、Phase 1 的根本问题

Phase 1 的 decode 循环长这样：

```python
while unfinished:
    for state in batch:
        out = model(input_ids=[[last_token]], past_key_values=state.kv, ...)
        state.kv = out.past_key_values
        state.append_generated(...)
```

这是一个串行 `for` 循环。每条请求独立做一次 forward，GPU 的利用率和批次大小呈倒数关系——批次越大，等待越长，GPU 等着你 Python 循环。

更深的问题是 `past_key_values` 的存储方式：每个请求的 KV 是一个 Python dict 里的 tuple，随着序列增长按请求粒度分散在 GPU 内存各处，没有上限，没有统一管理。

Phase 2 要解决的就是这两个问题：**显存怎么管、batch 怎么跑**。

---

## 二、设计：三个机制，一个主循环

### 2.1 Paged KV Cache

思路来自 vLLM 的 PagedAttention 论文：把 KV cache 切成固定大小的 block，预先分配好一个 GPU tensor 池，用 BlockTable 记录每个请求占了哪些物理块。

存储格式：
```
k_cache[layer_idx]  shape: [num_gpu_blocks, block_size, num_kv_heads, head_dim]
v_cache[layer_idx]  shape: 同上
```

每个 block 存 `block_size` 个 token 的 KV。所有 block 归属一个 `deque`（FreeBlockPool），分配 O(1)，回收也 O(1)。

BlockTable 是 `dict[request_id, list[int]]`，列表里是物理块号。逻辑 token 位置 `pos` 对应的物理位置：

```python
block_idx = pos // block_size
slot_idx  = pos % block_size
phys_blk  = block_table[request_id][block_idx]
k_cache[l][phys_blk, slot_idx]  # 这就是对应的 KV slot
```

与 Phase 1 相比：
- 显存有上限（`num_gpu_blocks` 固定）
- 请求结束立即归还所有块，无碎片
- 多请求共享同一个 pool，跨请求调度成为可能

### 2.2 Batch Decode

Phase 1 的核心问题是串行 forward。Batch decode 的思路是：把 N 个请求的最后一个 token 拼成 `[N, 1]` 的 input_ids，同时把它们各自的 KV 聚合到 `[N, num_kv_heads, max_seq_len, head_dim]`，做一次 batch forward。

但问题来了：每个请求的序列长度不同，如何对齐？

用左填充（left-padding）：短请求在左边补 0，attention_mask 对应位置也设为 0。为什么是左填充而不是右填充？

因为 decode 阶段每个请求的 input_ids 都只有 1 个新 token，它的位置编码必须对应正确的绝对位置。Qwen2 用的是 RoPE，位置 ID 由 attention_mask 的 cumsum 自动推导：

```python
# transformers 内部，position_ids 大致是：
position_ids = attention_mask.cumsum(-1) - 1
# 左填充下，真实 token 的 position_ids 从 0 开始连续，不会错位
```

右填充会把新 token 挤到序列开头，position_ids 算错，导致 RoPE 错乱。

### 2.3 Continuous Batching

Continuous batching 的核心不是一次把所有请求塞进去，而是**每个 decode 步骤结束后都检查能不能加入新请求**。

主循环结构：

```python
while has_waiting or running:
    # 1. 准入：尽量接入等待队列里的请求
    while has_waiting and num_running < max_batch_size:
        if free_blocks < blocks_needed:
            break  # 显存不足，等其他请求跑完释放块
        admit(next_state)

    # 2. Prefill：新接入的请求独立 prefill
    if newly_admitted:
        prefill(newly_admitted)

    # 3. Batch decode：所有 running 请求合并一次 forward
    if running:
        decode_batch(running)

    # 4. 清理完成的请求，归还 KV 块
    for state in running:
        if state.finished:
            free_request(state)
```

与静态 batching 的区别：静态 batching 等一批请求全部完成才拉新的；continuous batching 允许先完成的请求立即空出位置给新请求，GPU 不会因为等"最慢的那条"而空转。

---

## 三、实现细节和踩坑

### 坑 1：无限循环

第一版 engine.py 的准入逻辑有一个隐患：

```python
while has_waiting and num_running < max_batch_size:
    if free_blocks < blocks_needed:
        break  # 等其他请求释放块
    admit(next_state)
```

如果此时 `num_running == 0`，也就是没有任何请求在跑，没有任何块会被释放，但循环还是会一直跑下去。下一次迭代依然 `has_waiting=True`，依然 `free_blocks < blocks_needed`，依然 `break`，死循环。

修复：

```python
if free_blocks < blocks_needed:
    if num_running == 0 and not newly_admitted:
        raise RuntimeError(
            f"请求需要 {blocks_needed} 个 KV 块，"
            f"但只有 {free_blocks} 个空闲块。"
        )
    break
```

加了一个测试专门覆盖这个路径：`test_engine_oom_raises_not_loops`，用 dry_run + 极小的 block pool 验证不死循环。

### 坑 2：KV heads 数量写错了

Phase 1 的知识笔记里记的是 Qwen2.5-7B 有 8 个 KV heads（GQA），但实际查 `config.json`：

```
num_hidden_layers: 28
num_attention_heads: 28
num_key_value_heads: 4   ← 不是 8
```

28 个 Q heads，4 个 KV heads，7:1 的 GQA 比例。

这个错误在 `k_cache` 分配时触发了 shape mismatch：

```
RuntimeError: The expanded size of the tensor (8) must match the existing size (4)
```

改 `config.py` 默认值：`num_kv_heads: int = 4`。顺带教训：模型架构参数**必须从 `config.json` 读**，不能凭印象。

验证命令：

```bash
python3 -c "
import json
with open('config.json') as f:
    c = json.load(f)
print(c['num_hidden_layers'], c['num_key_value_heads'], c['hidden_size']//c['num_attention_heads'])
"
# → 28 4 128
```

### 坑 3：DynamicCache 兼容性

transformers 4.43.4 已经弃用了 tuple 格式的 `past_key_values`：

```
UserWarning: We detected that you are passing `past_key_values` as a tuple and this is deprecated...
Please use an appropriate `Cache` class.
```

但它还没有强制报错，内部通过 `DynamicCache.from_legacy_cache(past_kv)` 把我们传的 tuple 转换了。所以功能上没问题，只是有个 warning。

这意味着目前的实现在 transformers 4.43 之后的版本可能直接 break，Phase 3 需要迁移到 `DynamicCache` API。暂时接受这个 warning。

### 坑 4：显存管理细节

batch decode 时 `gather_batch_kv()` 创建的 `k_layer`、`v_layer` 是新的 dense tensor，forward 完成后必须立即释放：

```python
k_new = [out.past_key_values[l][0][:, :, -1, :] for l in range(num_layers)]
v_new = [out.past_key_values[l][1][:, :, -1, :] for l in range(num_layers)]
logits_batch = out.logits[:, 0, :].clone()

self.kv_cache.write_decode_kv(request_ids, k_new, v_new)
del k_batch, v_batch, past_kv, out, k_new, v_new  # 必须主动 del
```

注意 `logits_batch = out.logits[:, 0, :].clone()`——必须 clone，否则 del out 之后 logits_batch 也失效。

另一个细节：异常时必须归还 KV 块，否则下次运行 block pool 会被耗尽：

```python
except Exception:
    for state in self.scheduler.get_running_states():
        try:
            self.kv_cache.free_request(state)
        except Exception:
            pass
    raise
```

Phase 1 没有这个 try/finally，每次异常都会泄漏显存。

---

## 四、数据

环境：Ubuntu 24.04，RTX 4090（24 GB），Qwen2.5-7B-Instruct，float16，max_new_tokens=128。

### HuggingFace Baseline（静态 batching）

| batch_size | Throughput (tok/s) | TTFT (ms) | TPOT (ms/tok) | Peak Mem (GB) |
|-----------|-------------------|-----------|---------------|---------------|
| 1 | 56.2 | 19.0 | 17.78 | 15.78 |
| 4 | 210.5 | 20.8 | 18.99 | 15.81 |
| 8 | 408.9 | 23.6 | 19.53 | 15.88 |

HF 的 TPOT 几乎不随 batch size 变化（17.78 → 18.99 → 19.53ms），说明它在所有 batch size 下都是 compute-bound，每步 forward 的延迟几乎只由模型大小决定。

### mini-infer Phase 2（Paged KV Cache + Continuous Batching）

| batch_size | Throughput (tok/s) | TTFT (ms) | TPOT 均摊 (ms/tok) | Peak Mem (GB) |
|-----------|-------------------|-----------|---------------------|---------------|
| 1 | 49.4 | 18.7 | 20.24 | 16.26 |
| 4 | 135.2 | 19.0 | 7.42 | 16.31 |
| 8 | 201.0 | 18.8 | 5.00 | 16.42 |

> TPOT 口径说明：mini-infer 报的是 amortized 均摊值 `decode_time / total_decode_tokens`，HF 报的是单请求均摊值，两者不能直接比较。

### 分析

**TTFT 与 HF 对齐**：mini-infer 的 TTFT 为 18.7–19.0ms，与 HF 的 19.0–23.6ms 基本一致，说明 prefill 路径没有引入额外开销。

**Throughput 差距随 batch 扩大**：

| batch | HF | mini-infer | 比率 |
|-------|-----|-----------|------|
| 1 | 56.2 | 49.4 | 87.9% |
| 4 | 210.5 | 135.2 | 64.2% |
| 8 | 408.9 | 201.0 | 49.1% |

batch=1 时 mini-infer 达到 HF 的 87.9%，差距尚可；batch=8 时只剩 49.1%。差距的主因是 `gather_batch_kv()`。

**`gather_batch_kv()` 的代价**：这个函数每个 decode step 都把所有请求的 KV 从 block pool 复制到一个新的 dense tensor。复制量是 `O(batch × max_seq_len × num_layers)`，随 batch 线性增长。而真正的 PagedAttention 用自定义 CUDA kernel，直接在 block pool 上做 attention，完全不需要复制。

**Batch decode 的效果**：mini-infer 均摊 TPOT 从 batch=1 的 20.24ms 下降到 batch=8 的 5.00ms，说明 batch decode 确实在均摊 GPU compute，方向是对的，主要问题在 KV 复制的 memory bandwidth 开销。

**显存开销**：mini-infer 比 HF 多约 0.5 GB，符合预期：

```
512 blocks × 16 slots × 28 layers × 2(K+V) × 4 heads × 128 dim × 2 bytes(fp16)
= 512 × 16 × 28 × 2 × 4 × 128 × 2 = 471M ≈ 0.48 GB
```

---

## 五、还差什么

当前 mini-infer Phase 2 的吞吐量低于 HF baseline，**原因不是 batch decode 的思路错了，而是 `gather_batch_kv()` 的实现方式带来了额外的 memory bandwidth 开销**。

真正的 PagedAttention（vLLM 的做法）是：
1. 把 block table 作为 GPU tensor 传给 attention kernel
2. kernel 直接通过 block table 索引访问 block pool，不复制
3. 整个 attention 在 block pool 上原地完成

这需要写自定义 CUDA 或 Triton kernel，是 Phase 3 的方向。

另一个悬而未决的问题是 continuous batching 的优势没有被测出来。本次 benchmark 用的是 8 条同质 prompt（都是问 LLM 推理优化的），长度接近，完成时间也接近，continuous batching 的"先完成先补位"效果几乎没有体现。真实的优势需要在混合长度、混合请求速率的场景下测量。

---

## 六、结论

Phase 2 实现了工业推理引擎的三个核心机制：Paged KV Cache、Batch Decode 和 Continuous Batching。架构上是对的，测试通过，benchmark 可复现。

但不要对数字抱有幻想：batch=8 时吞吐只有 HF 的 49%，主要瓶颈是 Python 层的 KV 复制，不是模型本身。消除这个瓶颈需要自定义 attention kernel，下一阶段再处理。

这一阶段最大的收获不是性能数字，而是把 BlockTable、FreeBlockPool、gather_batch_kv、batch forward 这一套机制亲自写了一遍，知道每个环节在做什么，以及哪里是真正的瓶颈。
