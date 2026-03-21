# 从串行 Decode 到 HTTP 服务：手写 LLM 推理系统的全程复盘

> mini-infer 项目完整回顾，2026-03-18 至 2026-03-21，历经 8 个主阶段与 1 个 Phase 6.5 专项阶段。

---

## 前言

这篇文章记录了我从零实现一个 LLM 推理系统的完整过程。项目从最朴素的串行 decode 开始，逐步实现了 Paged KV Cache、Batch Decode、Continuous Batching、True PagedAttention（flash_attn block_table）、Triton decode kernel、Preemption + Priority Scheduling，最终包装成一个 OpenAI Chat Completions 子集兼容的 HTTP 服务。

当前 HTTP 层支持 `/v1/models` 与 `/v1/chat/completions` 的 streaming / non-streaming 基础路径，固定 `model="mini-infer"`，`n` 仅支持 `1`，`stop`、`presence_penalty`、`frequency_penalty` 等字段会直接报错。它足够支撑本项目的 benchmark 与基础 SDK 对接；在清理本机代理环境变量后，OpenAI Python SDK 已能正常调用这条基础路径。但它仍然不是完整的 OpenAI API 复刻。

选择 Qwen2.5-7B-Instruct 作为目标模型，RTX 4090 作为硬件平台。整个项目的性能基准线是 HuggingFace Transformers 的 batch=8 吞吐（~408 tok/s）。

这篇文章**不是教程**，不会一行一行讲解代码。它是一份从工程实践角度的复盘，重点记录这些内容：每个阶段做了什么关键决策、卡在哪里、怎么突破的，以及那些只有跑真实 GPU 才能暴露的 bug。

---

## 整体性能曲线

先把结果放在最前面。以下是 batch=8、max_new_tokens=128 的吞吐演进（相对 HF baseline）：

| 阶段 | 实现 | Throughput (tok/s) | vs HF |
|------|------|-------------------|-------|
| HF baseline | HuggingFace Transformers | 408.9 | 100% |
| Phase 1 | 串行 decode（逐请求 forward） | 56.3 | 13.8% |
| Phase 2 | Paged KV Cache + Batch Decode + Continuous Batching | 201.0 | 49.1% |
| Phase 3 | 向量化 gather_batch_kv（PyTorch advanced indexing） | 361.3 | 88.4% |
| Phase 6 | True PagedAttention（flash_attn block_table） | 406.3 | 100.0% |

Phase 4 是双卡扩展（数据并行 +4.1%，Pipeline Parallel 吞吐不变但显存减半），Phase 5 是 profiling 分析，Phase 6.5 是 Triton kernel（对比实验），Phase 7 是 Preemption，Phase 8 是 HTTP API。这几个阶段的性能数字在对应章节单独讨论。

---

## Phase 1：串行推理链路，建立基准

### 做了什么

Phase 1 的目标很简单：在 RTX 4090 上跑通 Qwen2.5-7B，输出真实文字，建立一个可以对比的 HuggingFace baseline。

实现方式是最直接的：每个请求单独调用 `model(input_ids, past_key_values=past_kv)`，逐 token 生成，串行执行。代码结构参考 HuggingFace 的 `generate()` 内部逻辑，但剥离了 beam search 等复杂功能，只保留 greedy sampling。

```python
# Phase 1 核心 decode 循环（简化）
for state in running_states:
    out = model(input_ids=state.next_token.unsqueeze(0),
                past_key_values=state.past_kv,
                use_cache=True)
    next_token = out.logits[0, -1].argmax()
    state.update(next_token, out.past_key_values)
```

### 数据

| batch | mini-infer Phase 1 | HF baseline | 比率 |
|-------|-------------------|-------------|------|
| 1 | 56.4 tok/s | 56.2 tok/s | 100% |
| 4 | 56.4 tok/s | 210.4 tok/s | 26.8% |
| 8 | 56.3 tok/s | 408.9 tok/s | 13.8% |

batch=1 时两者几乎一致，说明单请求推理链路没有问题。batch=4 时差距出现，batch=8 时差距扩大到 7.3×。差距随 batch 线性扩大，这正是串行 decode 的特征：8 个请求轮流占用 GPU，每个 token 要做 8 次独立的 forward，GPU 利用率极低。

### 关键发现

**批量推理不等于串行执行**。HF 的 `generate()` 在接受多条 prompt 时，会将它们 pad 到相同长度后做一次 batch forward，后续 decode 也是 batch forward。mini-infer Phase 1 的错误在于"每个请求独立 forward"——这与 batch=1 运行 8 次没有区别。

这就确定了 Phase 2 的核心目标：实现 Batch Decode。

---

## Phase 2：Paged KV Cache + Batch Decode + Continuous Batching

### 三件事，一个阶段

Phase 2 要做的事情比 Phase 1 多得多，但这三件事耦合在一起，不容易拆开：

**Paged KV Cache**：不能用 HuggingFace 的 `DynamicCache`，因为它是每个请求独立的 Python 列表，无法在请求间共享块内存。需要一个全局的预分配 block tensor pool，每个请求通过 BlockTable 映射到物理块。

**Batch Decode**：多个请求的 KV 需要"聚合"成一个 dense tensor，才能送入一次 batch forward。这个聚合操作就是 `gather_batch_kv`。

**Continuous Batching**：不等所有请求同时到达，新请求随时加入运行队列。这需要一个调度主循环（engine.py），每 step 决定 prefill 哪些新请求、decode 哪些已有请求。

```python
# engine.py 主循环骨架
while has_unfinished():
    # 准入：尽量多接新请求，但不超 KV 空间
    admit_new_requests()
    # Prefill 新准入的请求
    for state in newly_admitted:
        model_runner.prefill(state)
    # Batch decode 所有 running 请求
    model_runner.decode_batch(running_states)
    # 清理已完成的请求
    cleanup_finished()
```

### 关键设计：KV cache 布局

KV cache 的物理布局：

```python
# k_cache[l]: (num_blocks, block_size, num_kv_heads, head_dim)
# 例：Qwen2.5-7B，block_size=4，num_kv_heads=4，head_dim=128
# 预分配一次，整个生命周期不 realloc
k_cache = [
    torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                device='cuda', dtype=torch.float16)
    for _ in range(num_layers)
]
```

每个请求有一个 BlockTable（逻辑块号 → 物理块号的映射）。prefill 时按 token 顺序写入 block，decode 时通过 BlockTable gather 出完整的 KV 序列。

### Phase 2 gather_batch_kv：三层 Python 循环

Phase 2 的 gather 实现是最直接的 Python 循环：

```python
# Phase 2 实现（慢）
for l in range(num_layers):
    for b, state in enumerate(running_states):
        for block_idx, phys_block in enumerate(state.block_table):
            k_gathered[l][b, block_idx*block_size:(block_idx+1)*block_size] = \
                k_cache[l][phys_block]
```

3 层嵌套循环，每个循环都在 Python 层执行，GPU 只在最内层的 tensor 赋值时才介入。这就是 Phase 2 相比 HF 仍有 2× 差距的根源。

### 数据

batch=8 从 Phase 1 的 56.3 tok/s 提升到 201.0 tok/s，是 HF 的 49.1%。TTFT 与 HF 基本一致（18.7 ms vs 19.0 ms），说明 prefill 路径没有问题，overhead 全在 decode。

---

## Phase 3：向量化 gather，一次 CUDA kernel 解决三层循环

### 问题定位

Phase 2 的瓶颈已经很清楚：`gather_batch_kv` 的三层 Python 循环。Phase 3 的目标是用 PyTorch advanced indexing 消除它。

关键思路：把"按 BlockTable 寻址"转换成一次 advanced indexing 操作。

```python
# Phase 3 实现（向量化）
# 1. 构建物理块索引 [batch, max_num_blocks]
block_table_tensor = build_block_table_tensor(running_states)  # [B, max_blocks]

# 2. 计算每个 token 对应的物理块号和槽位
# token_positions: [B, max_seq_len]，含左填充偏移
block_indices = token_positions // block_size              # [B, max_seq_len]
slot_indices = token_positions % block_size                # [B, max_seq_len]
phys_blocks = block_table_tensor[batch_arange, block_indices]  # [B, max_seq_len]

# 3. 一次 advanced indexing gather
# k_cache[l]: [num_blocks, block_size, num_kv_heads, head_dim]
k_gathered = k_cache[l][phys_blocks, slot_indices]        # [B, max_seq_len, num_kv_heads, head_dim]
k_gathered = k_gathered * valid_mask_f                    # 置零填充位
```

Python 层只有常数次操作，CUDA 层一次 gather kernel 完成所有工作，batch 大小不影响 Python 调度开销。

### 数据

| batch | Phase 2 | Phase 3 | 提升 | vs HF |
|-------|---------|---------|------|-------|
| 1 | 49.4 | 53.7 | +8.7% | 95.5% |
| 4 | 135.2 | 194.2 | +43.6% | 92.3% |
| 8 | 201.0 | 361.3 | **+79.8%** | 88.4% |

batch=8 提升 79.8%，从 49.1% HF 跳到 88.4%。仍差 12%，但差距已经很小了。

### 为什么还差 12%

Phase 5 的 profiling 给出了答案（见下）。简短版：gather 虽然从 3 层 Python 循环变成了 1 次 CUDA kernel，但**仍然是物理拷贝**——它把 block tensor 里分散的 KV 数据复制成一个 dense tensor，再传给 Transformer attention。这个拷贝本身就是开销，而且 HF 的 `DynamicCache` 是 in-place 写入，没有这一步。

真正的解法需要 attention kernel 直接从 block tensor 寻址，这就是 Phase 6 要做的事情。

---

## Phase 4：双卡扩展——收益的边界在哪里

### 两种策略，两种结论

Phase 4 实现并测量了两种双卡策略：

**Replica（数据并行）**：两块 GPU 各跑一个完整的 LLMEngine，round-robin 分发请求。

**Pipeline Parallel（HF device_map="balanced"）**：模型的前 14 层在 GPU0，后 14 层在 GPU1，单条序列串行通过两卡。

结果：

| 模式 | Throughput | vs single | GPU0 显存 | GPU1 显存 |
|------|-----------|-----------|----------|----------|
| single | 361.4 tok/s | 1× | 16.42 GB | — |
| replica | 376.1 tok/s | +4.1% | 16.31 GB | 16.31 GB |
| Pipeline Parallel | 361.5 tok/s | +0.0% | 7.00 GB | 8.97 GB |

Replica +4.1%，Pipeline Parallel 0% 吞吐提升但显存减半。

### 为什么 Replica 只有 +4.1%

这是一个很好的问题。直觉上两块 GPU 应该接近 2× 速度，实际只有 +4%，原因在数学上是精确的：

- Phase 3 单卡 batch=4 吞吐 = 194.2 tok/s
- Replica 双卡各跑 batch=4，理论上限 = 194.2 × 2 = 388.4 tok/s
- 实测 376.1 tok/s（97% 效率，有 ThreadPoolExecutor 调度开销）

但单卡 batch=8 = 361.3 tok/s，已经接近 "两卡各跑 batch=4" 的上限了。原因是 batch=4→8 在单卡上只增加了 86% 的吞吐（194→361），而两卡并行 batch=4 的总量是单卡 batch=8 的 107%。所以 replica 相比单卡 batch=8 只能多出 7%，刨去调度开销剩 4%。

**真正发挥 Replica 价值的场景**：总并发明显高于单卡最优 batch（例如 16+ 这类更高总并发），或者单卡 KV 显存装不下更多请求。对于 Qwen2.5-7B 在 24 GB 显卡上，它更像是"面向更高总请求量"而不是"让 batch=8 立刻翻倍"的方案。

**Pipeline Parallel 的正确用场**：不是提速，是让装不进单卡的大模型（70B+）能跑起来，每卡显存减半。

---

## Phase 5：Profiling——97% 在 model_forward，说明了什么

### 为什么做 profiling

Phase 3 之后性能差距只剩 12%，但不知道这 12% 具体在哪里。Phase 5 在 `decode_batch()` 的三个关键操作上加了 `torch.profiler.record_function` 标签，然后在真实 GPU 上测量。

```python
# model_runner.py decode_batch() 中
with record_function("gather_batch_kv"):
    k_batch, v_batch = kv_cache.gather_batch_kv(running_request_ids)

with record_function("model_forward"):
    out = model(input_ids=decode_input_ids, past_key_values=dynamic_cache)

with record_function("write_decode_kv"):
    kv_cache.write_decode_kv(running_request_ids, out.past_key_values)
```

### 结果

| batch | gather_kv | model_forward | write_kv | model_forward 占比 |
|-------|-----------|--------------|----------|--------------------|
| 1 | 0.27 ms | 17.27 ms | 0.02 ms | 98.3% |
| 4 | 0.32 ms | 17.95 ms | 0.18 ms | 97.3% |
| 8 | 0.31 ms | 17.89 ms | 0.19 ms | 97.2% |

model_forward 占 97-98%，而且 batch=1 到 batch=8 的 model_forward 时间几乎不变（17.27 ms vs 17.89 ms）。这说明 RTX 4090 在 batch=4~8 的 decode 场景下已经接近算力上限。

### 剩余 12% 差距的根因

| 来源 | 量化 |
|------|------|
| gather_batch_kv | 0.31 ms（确定消除，Phase 6 做） |
| write_decode_kv | 0.19 ms（确定消除，Phase 6 做） |
| 左填充 attention_mask | 难以单独量化 |
| Python 调度开销 | < 0.5 ms |

gather + write_kv = 0.5ms / 18ms ≈ 2.8%。这解释了约 3 个百分点的差距，剩余的来自 attention_mask 开销（flash_attn 用 cache_seqlens 处理变长序列，不需要 attention_mask）。

**关键结论**：残余 12% 差距不是因为代码写得差，而是 gather→DynamicCache→write_kv 这个三段路径本身有 0.5ms/step 的不可消除开销，需要改变架构（Phase 6）才能解决。

---

## Phase 6：True PagedAttention——flash_attn 如何消灭 gather

### 问题的根本

Phase 1-5 的 decode 数据流：

```
block tensor → gather_batch_kv → dense KV tensor
                                       ↓
                              DynamicCache 预填充（28层）
                                       ↓
                              model.forward() → flash_sdp（从 DynamicCache 读取）
                                       ↓
                              write_decode_kv → block tensor
```

每一步都有独立的内存操作。True PagedAttention 应该是：

```
flash_attn_with_kvcache(q, k_cache[l], v_cache[l],
                         k=k_new, v=v_new,
                         block_table=block_table,
                         cache_seqlens=cache_seqlens)
```

一个 kernel 完成：从 block tensor 寻址读取历史 KV，写入新 KV，计算 attention。没有 gather，没有 DynamicCache，没有 write_kv。

### 实现方式：patch 而非重写

不重写 Transformer 模型，而是用 Python monkey patch 替换每层的 attention forward：

```python
def patch_model_for_paged_decode(model, kv_cache_manager):
    for layer_idx, layer in enumerate(model.model.layers):
        original_forward = layer.self_attn.forward
        # 替换为 paged_decode_attention 的封装
        layer.self_attn.forward = make_paged_attn_forward(
            layer_idx, original_forward, kv_cache_manager)
```

Prefill 路径不变，decode 路径替换。通过一个 `PagedDecodeContext` 单例注入 decode 时需要的共享状态（block_table、cache_seqlens、max_kv_len）。

### 关键 bug：`.item()` 在 28 层各调一次

Phase 6 第一次跑 benchmark 时结果是 3.7% HF（原来 88.4%），严重倒退。

诊断：profiler 显示 decode_batch 耗时 527 ms/step，而 Phase 5 只有 17.9 ms/step。

根因：`patched_forward` 内部计算 `max_kv_len`：

```python
# ❌ 这行代码被 28 层各执行一次
max_kv_len = int(ctx.cache_seqlens.max().item()) + 1
```

`.item()` 会触发一次 CPU-GPU sync（等待 GPU 完成 `max()` 计算再把结果传回 CPU）。被 28 层各调用一次 = 28 次 sync/step。每次 sync 约 18 ms，合计 504 ms，主导了所有 decode 时间。

修复很简单：在 `decode_batch()` 里算一次，通过 `PagedDecodeContext.max_kv_len: int` 传给所有层：

```python
# ✅ decode_batch() 里算一次，存为普通 Python int
paged_ctx.max_kv_len = int(cache_seqlens.max().item()) + 1
# patched_forward 里直接用 ctx.max_kv_len，不再触发 sync
```

修复后 benchmark：406.3 tok/s，100.0% HF。

这个 bug 在三轮 infer-review 中都没被发现——它是**性能问题而非正确性问题**，静态代码分析无法量化 28 次 sync 的实际开销。只有跑了 benchmark 看到异常数字，再用 profiler 诊断才能定位。

### 数据

| 指标 | Phase 5/旧路径 | Phase 6/新路径 | HF baseline |
|------|---------------|---------------|-------------|
| Throughput | 361.3 tok/s | **406.3 tok/s** | 406.4 tok/s |
| vs HF | 88.4% | **100.0%** | 100% |
| TTFT（近似） | 43.8 ms | **18.8 ms** | 19.9 ms |
| model_forward | 17.89 ms/step | **17.06 ms/step** | — |
| gather_batch_kv | 0.31 ms/step | **0** | — |
| write_decode_kv | 0.19 ms/step | **0** | — |

这里最硬的结论是 throughput：Phase 6 达到 100.0% HF。TTFT 的下降方向也是对的，但它来自不同 benchmark 脚本下的近似测量，不能把 43.8 → 18.8ms 全部机械归因到单一优化。更稳妥的说法是：Phase 6 去掉了 gather / write_kv 路径，并把 decode 侧的 attention 输入准备收紧到 `cache_seqlens` 语义后，端到端延迟指标也随之改善。

---

## Phase 6.5：用 Triton 写一个 decode attention kernel

### 为什么做这件事

flash_attn 是高度优化的黑盒。理解 attention kernel 的真实性能瓶颈（memory-bound vs compute-bound、tile size 对性能的影响）需要自己写一个。

Phase 6.5 的目标不是超越 flash_attn，而是：走通 Triton kernel 开发路径，量化差距，能解释清楚差距在哪里。

### 实现要点

一个 decode attention kernel 的关键参数：
- Q shape：`(batch, 1, num_q_heads, head_dim)` — decode 每步只有 1 个新 token
- KV cache shape：`(num_blocks, block_size, num_kv_heads, head_dim)`
- block_table：`(batch, max_blocks_per_seq) int32`

每个 Triton program 处理一个 `(batch_id, q_head_id)` 对，沿 KV 序列长度方向分块迭代，维护 online softmax 状态（`m_i`, `l_i`, `acc`）。

遇到的最有意思的编译报错：

```
'triton_gpu.cmpf' op requires the same encoding
```

根因：`tl.full([1], -1e38)` 产生 blocked encoding，`tl.max(scores, 0)` 产生 scalar encoding，两种 encoding 不能直接参与 `tl.maximum`。修复：用 `scores[None, :]` 升维到 2D，再 `tl.max(..., axis=1)` reduce 到 `[1]`，保持 blocked encoding 一致。

### 数据

| batch | seq_len | Triton (μs) | flash_attn (μs) | 差距 |
|-------|---------|------------|----------------|------|
| 1 | 128 | 14.30 | 11.66 | 1.23× |
| 1 | 2048 | 168.26 | 17.78 | **9.46×** |
| 8 | 128 | 20.83 | 12.54 | 1.66× |
| 8 | 2048 | 182.84 | 60.01 | 3.05× |

seq_len=128 时差距只有 1.23×，但 seq_len=2048 时扩大到 9.46×。

Roofline 分析：decode attention 的算术强度约 7 FLOPs/Byte，远低于 RTX 4090 的 ridge point（82 FLOPs/Byte），是典型的 memory-bound 操作。Triton 实现与 flash_attn 的差距不是算法问题，而是工程优化：
- flash_attn：向量化 load（128 bit 对齐），prefetch pipeline，GQA K/V 共享，warp-level 优化
- 自实现：标量 load，无 prefetch，无专项 GQA 优化

seq_len 越大，从显存反复读取 KV 的次数越多，工程优化的绝对差值也就越大，所以差距随 seq_len 扩大。

---

## Phase 7：Preemption + Priority Scheduling

### 问题背景

Phase 1-6 的调度器在 KV 块不足时直接 RuntimeError。真实推理服务需要优雅降级：将低优先级请求的 KV 换出到 CPU，腾出空间给新来的高优先级请求。

### 状态机

```
WAITING → RUNNING → FINISHED
              ↓           ↑
           SWAPPED ────────
```

`swap_out`：将请求的 KV blocks 逐块 `.cpu()` 拷贝存储，释放 GPU block，标记为 SWAPPED。
`swap_in`：重新分配 GPU block，将 CPU tensor `.cuda()` 写回，恢复 BlockTable，加入 running。

### 关键 bug：never-prefilled 请求的 crash

Preemption 逻辑在准入循环里：准入新请求 → 如果 KV 块不足，换出 running 中优先级最低的请求。

第一版实现没有区分"刚准入但还没有 prefill 的请求"和"已经 prefill 的请求"。如果一个请求刚被 `add_to_running`（还没执行 prefill），就被选为 victim，原逻辑会调用 `swap_out`，后者会 `free_request` 清空它的 `_block_tables`。但此时这个请求还在 `newly_admitted` 列表里，等一会儿 `write_prefill_kv` 会去访问 `_block_tables[request_id]` → KeyError crash。

修复：加 `prefilled: bool` 标志，eviction 时分支处理：

```python
if victim.prefilled:
    kv_cache.swap_out(victim)
    scheduler.mark_swapped(victim)
else:
    # 还没 prefill，没有有效 KV，直接撤销准入
    kv_cache.free_request(victim.request_id)
    scheduler.un_admit(victim)
    newly_admitted.remove(victim)
```

这个 bug 被 dry_run 完全掩盖。dry_run 模式下 `write_prefill_kv` 有 `if self._dry_run: return` 早返回，所以 `_block_tables` 为空也不 crash。所有 dry_run 测试通过，但真实 GPU 路径随时会炸。

### 数据

| 指标 | 值 |
|------|-----|
| swap_out 延迟（元数据操作） | 0.36 µs |
| swap_in 延迟（元数据操作） | 0.60 µs |
| GPU↔CPU KV 拷贝实测带宽 | ~1100 MB/s |
| seq_len=256 请求的 swap 耗时 | ~13.5 ms |
| 吞吐回归（有/无优先级调度） | 198.3 vs 198.2 tok/s（-0.0%）|

PCIe 实测带宽 ~1100 MB/s 远低于理论峰值 32 GB/s，根因是 Python 层逐块循环（每块一次 `.cuda()`/`.cpu()` 调用触发独立 PCIe 传输 + CUDA sync）。vLLM 的实现会将同一请求所有层的 KV 合并成一次大块传输，带宽利用率高一个数量级。

---

## Phase 8：把推理引擎包装成 HTTP 服务

### 核心挑战

LLMEngine 是同步的。HTTP 请求是异步到来的。要在 HTTP 层保留 continuous batching，不能让每个 HTTP 请求独占推理循环。

错误的直觉：

```python
# ❌ 每个 HTTP 请求独立调用 generate()
@app.post("/v1/chat/completions")
async def chat(request):
    result = await loop.run_in_executor(None, engine.generate, [prompt])
    return result
```

这样两个并发请求会串行执行，没有任何 batching 收益。

正确的架构：

```
HTTP 请求 A ─┐
HTTP 请求 B ─┤── add_request() ──→ [后台 step loop] ──→ step() 处理所有 running 请求
HTTP 请求 C ─┘                                               ↓
                                                    每个请求的 token 投递到各自的 asyncio.Queue
```

单后台线程持续运行 `engine.step()`，HTTP 请求通过 `add_request()` 加入调度队列，每个请求通过 `asyncio.Queue` 等待 token，后台线程通过 `loop.call_soon_threadsafe(queue.put_nowait, token)` 跨线程投递。

### 接口边界

这里实现的是 OpenAI Chat Completions 的**受限子集**，不是完整复刻：

- 支持 `GET /v1/models`
- 支持 `POST /v1/chat/completions` 的 streaming / non-streaming
- `model` 固定为 `mini-infer`
- `n` 仅支持 `1`
- `stop`、`presence_penalty`、`frequency_penalty` 当前不支持

这个边界足够支撑 benchmark、curl 调用和基础 SDK 适配，但如果把它表述成"完整 OpenAI-compatible 服务"，就会比实际能力更强。

### 竞态条件

第一版实现犯了一个微妙的错误：先调用 `add_request()`，再注册 queue。

```python
# ❌ 竞态：queue 注册在 add_request() 之后
rid = engine.add_request(prompt, max_new_tokens)
self._token_queues[rid] = asyncio.Queue()  # 后台线程可能已经在 _put()
```

后台线程在 `add_request()` 之后、queue 注册之前执行了 `step()`，调用 `_put(rid, token)` 时 `_token_queues.get(rid)` 返回 None，token 丢弃，HTTP 请求永远等不到这个 token。

修复：先注册 queue，再调用 `add_request()`：

```python
# ✅ 正确顺序
rid = str(uuid4())
self._token_queues[rid] = asyncio.Queue()  # 先注册
engine.add_request(prompt, max_new_tokens, request_id=rid)  # 后提交
```

### 流式解码的空字符串陷阱

benchmark 首次运行时 throughput = 0 tok/s。诊断发现 `engine.step()` 返回的全是空字符串：

```
step 0: {'rid': ['', '']}
step 1: {'rid': ['']}
step 2: {'rid': ['']}
```

但 `engine.generate()` 完全正常。

追溯：`model_runner.prefill()` 和 `decode_batch()` 调用 `state.append_generated(token_id, "")` 时，token 文本始终是空字符串。`generate()` 不受影响，因为它在末尾做整个序列的批量 decode。`step()` 依赖 `generated_text_parts` 这个空数组，自然一无所获。

根因：单 token ID decode 对中文不安全。Qwen 用 BPE，一个汉字往往对应多个 token，每个 token 单独 decode 会返回空字符串或字节片段。例如"大"可能编码为字节级别的 2-3 个 token，只有合并 decode 才能得到完整字符。

正确的流式 decode 方式：**增量全序列 decode**。

```python
# step() 中收集新 token 文本
pre = pre_lens.get(rid, 0)
curr = len(state.generated_token_ids)
if curr > pre:
    old_text = tokenizer.decode(ids[:pre], skip_special_tokens=True) if pre > 0 else ""
    new_text = tokenizer.decode(ids[:curr], skip_special_tokens=True)
    delta = new_text[len(old_text):]
    if delta:
        new_tokens[rid] = [delta]
```

每步 decode 从头到当前位置的完整序列，再与上一步的结果取差值。这个做法把 tokenizer 的 CPU 端开销放回了流式路径里，但相对真实 GPU forward 仍然是可接受的保守修复。

这个 bug 被所有 dry_run 测试完全掩盖。`_StubTokenizer.decode([1])` 返回 `" [1]"`，decode `[1, 2]` 返回 `" [1] [2]"`，永远非空，逐 token decode 看起来完全正确。

### 数据

| 并发数 | Throughput (tok/s) | 相对单并发 |
|--------|-------------------|-----------|
| 1 | 55.7 | 1× |
| 2 | 105.0 | 1.88× |
| 4 | 193.6 | 3.47× |
| 8 | **351.4** | **6.3×** |

8 并发 351.4 tok/s，是单并发的 6.3 倍。峰值显存 23.34 GB，与 Phase 6/7 完全相同，说明 HTTP 层本身没有引入额外的 GPU 内存占用。

结合 benchmark 结果，可以判断 continuous batching 已通过 HTTP 层工作：并发请求被同一个后台 step loop 合并进 `decode_batch()`，而不是逐请求串行排队。不过当前自动化回归仍主要覆盖 dry-run 路径，真实模型 HTTP 路径更多依赖 benchmark 和手工验证。

---

## 整个项目的五条关键教训

### 1. 测试 stub 和真实路径是两个世界

这个项目里三个最严重的 bug，都被 dry_run 测试完全掩盖：

- **Phase 7 never-prefilled crash**：干跑 `write_prefill_kv` 有早返回，KeyError 不出现
- **Phase 8 step() 空字符串**：stub tokenizer 的 decode 行为与真实 tokenizer 差异极大
- **Phase 8 竞态条件**：单线程测试环境没有后台线程，race window 不存在

Stub / dry_run 测试验证的是逻辑流程，不能验证真实路径的数据质量。对于任何涉及"真实路径行为"的功能（tokenizer decode 的字符边界、GPU-CPU 内存传输、多线程竞态），干跑测试不够，需要真实 GPU 集成测试。

### 2. 被 N 层各调用一次的函数里，任何 CPU-GPU sync 都是性能杀手

Phase 6 的 `.item()` bug：一行看起来无害的 `int(tensor.max().item())` 被 28 层各调用一次，导致 28 次 CPU-GPU sync，decode 时间从 18ms 膨胀到 527ms。

规律：LLM 模型有 N 层 transformer block，每次 decode 会调用 `patched_forward` N 次。任何在 `patched_forward` 里的 CPU-GPU sync 操作（`.item()`、`.numpy()`、Python 条件判断 tensor 值）都会 ×N 放大。

预防方法：在 `decode_batch()` 层面预计算所有需要 Python 访问的标量值（用普通 `int`/`float`），通过 context 对象传给所有 attention 层。

### 3. 性能问题不能只靠 review 发现

Phase 6 的 `.item()` 经过三轮 review 都没被标注为阻塞性问题。它在代码逻辑上完全正确，只有在真实 GPU 上跑了 benchmark 看到 3.7% HF 这个异常数字，才触发了诊断。

review 擅长发现：接口错误、状态机错误、竞态条件（逻辑层面）、边界条件。
review 不擅长发现：只有在 GPU 运行时才能量化的性能问题、被 N 放大的隐性开销、跨线程时序问题。

### 4. 架构决策比优化更重要

从 Phase 2 到 Phase 6，throughput 从 201 到 406 tok/s，翻了一倍。但两次最大的跳跃都来自架构变更：

- Phase 2→3：从 Python 循环变成 PyTorch advanced indexing，+79.8%（改的是 gather 方式，不是模型）
- Phase 3→6（跳过4/5）：从 gather→DynamicCache→write_kv 变成直接 block_table，+12.4%（改的是数据流路径）

Phase 3 的向量化是在同一个数据流框架里的优化。Phase 6 是从根本上改变了数据流框架。类比到真实推理系统的演进：vLLM 对 PagedAttention 的实现也是"attention kernel 直接从 block tensor 寻址"，不是"gather + 标准 attention"。

### 5. 推理引擎和 HTTP 服务的解耦设计

Phase 8 最重要的设计决策：不是"每个 HTTP 请求独占一次 generate()"，而是"单后台线程持续 step()，HTTP 请求只是注册 queue + add_request"。

这个设计让 continuous batching 对 HTTP 层透明：无论有多少并发 HTTP 请求，它们都被同一个 `decode_batch()` 一起处理。8 并发 6.3× 的吞吐提升，是这个设计正确性的直接验证。

---

## 项目总结

mini-infer 在 4 天内从串行 decode 走到了 100% HF 吞吐 + OpenAI Chat Completions 子集兼容 HTTP API。核心路径：

```
串行 decode（13.8% HF）
→ Paged KV + Batch Decode + Continuous Batching（49.1%）
→ 向量化 gather（88.4%）
→ True PagedAttention（100.0%）
→ Preemption + Priority Scheduling
→ HTTP API（8 并发 6.3× 单并发）
```

最有价值的不是最终的性能数字，而是每个阶段踩到的坑：gather 的物理拷贝开销、`.item()` 在 N 层里的放大效应、dry_run 掩盖的 bug、HTTP 层的竞态条件、流式 decode 的多字节字符问题。这些在任何 LLM 推理框架的源码里都有对应的痕迹。

这些机制不抽象，每一个都在真实的实验数据里留下了印记。

---

## 还没做完的事情

项目主线已经收束，但有几类空白我不想假装不存在：

- **P99 / 端到端分位数延迟**：目前有 TTFT / TPOT / throughput，但还没有独立 uvicorn 进程 + 真实 streaming client 下的统一分位数口径
- **Chunked Prefill**：当前还没有 token budget 机制，长 prefill 仍可能阻塞 decode
- **Prefix Caching**：还没有 block 级前缀复用与命中统计
- **真正的 Tensor Parallel**：当前双卡只有 Replica 和 Pipeline Parallel，没有 Megatron/vLLM 风格的 all-reduce TP
- **生产级监控与故障恢复**：HTTP 服务已经能跑，但还没有监控、熔断、重试、健康恢复这一层

所以更准确的说法不是"我已经做完了一个生产级推理系统"，而是"我把一个教学级 / 研究级 mini 推理系统推进到了可以认真对照真实框架原理的程度"。

---

## 附：完整性能数据汇总

**单卡吞吐（batch=8，max_new_tokens=128，Qwen2.5-7B-Instruct，RTX 4090）**

| 阶段 | Throughput | vs HF | 关键改动 |
|------|-----------|-------|---------|
| HF baseline | 408.9 tok/s | 100% | — |
| Phase 1 | 56.3 tok/s | 13.8% | 串行 decode |
| Phase 2 | 201.0 tok/s | 49.1% | Paged KV + Batch Decode |
| Phase 3 | 361.3 tok/s | 88.4% | 向量化 gather |
| Phase 6 | 406.3 tok/s | **100.0%** | flash_attn block_table |

**Phase 5 Profiling（decode_batch 时间分布，batch=8）**

| 操作 | 时间 | 占比 |
|------|------|------|
| gather_batch_kv | 0.31 ms/step | 1.7% |
| model_forward | 17.89 ms/step | 97.2% |
| write_decode_kv | 0.19 ms/step | 1.0% |

**Phase 6 `.item()` bug 修复前后**

| 状态 | decode 时间 | Throughput |
|------|------------|-----------|
| 修复前（28× sync） | 527 ms/step | ~15 tok/s（3.7% HF）|
| 修复后 | 17.06 ms/step | 406.3 tok/s（100.0% HF）|

**Phase 6.5 Triton vs flash_attn（RTX 4090）**

| seq_len | Triton | flash_attn | 差距 |
|---------|--------|-----------|------|
| 128 | 14.30 µs | 11.66 µs | 1.23× |
| 512 | 44.22 µs | 13.79 µs | 3.21× |
| 2048 | 168.26 µs | 17.78 µs | 9.46× |

**Phase 8 HTTP API 并发吞吐（max_tokens=64，ASGI transport）**

| 并发数 | Throughput | 相对单并发 |
|--------|-----------|-----------|
| 1 | 55.7 tok/s | 1× |
| 2 | 105.0 tok/s | 1.88× |
| 4 | 193.6 tok/s | 3.47× |
| 8 | 351.4 tok/s | 6.3× |
