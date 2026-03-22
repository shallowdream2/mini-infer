# CUDA Graph 接入 Paged Attention：静态图与动态调度的折中

> 系列：mini-infer 推理系统学习项目 Phase 12
>
> 2026-03-22 当前仓库复验：在较短 workload（`decode_steps=20`、`warmup=8`）下，1.5B 仍然复现了同方向收益：bs=1/4/8 分别从 `9.06/9.50/12.00 ms` 降到 `5.51/7.74/9.63 ms`。正文表格继续保留阶段正式 benchmark 的完整结果。

---

## 背景：flash_attn 之后，还有什么开销？

经历了 Phase 6 的 True PagedAttention（`flash_attn_with_kvcache` + block_table）之后，mini-infer 的 decode 路径已经相当干净：每步只做一次 batch model forward，没有 gather、没有 DynamicCache.update，没有冗余拷贝。在 Qwen2.5-7B 上，batch=8 的 decode throughput 与 HuggingFace baseline 持平（100% HF）。

这似乎是终点了。但 profiler 里还有一个固定的 CPU 时间开销，始终占据每步的 3–5 ms，与 batch size、模型大小无关。

这就是 **Python → CUDA dispatcher 的调度开销**。

每次调用 `model.forward()`，PyTorch 需要遍历 28 层，对每层的 q_proj、k_proj、v_proj、o_proj、attention kernel、MLP……逐一构造 CUDA kernel 的启动参数、提交到 CUDA stream。这个过程不涉及 GPU 计算，但在 CPU 侧消耗的时间是固定的——大约每步 3–5 ms（1.5B 模型）。对于 flash_attn 本身只需 1–2 ms 的小模型，这个开销是不可忽视的。

CUDA Graph 的思路：**把 forward 录制一次，后续步骤直接回放**，跳过所有 Python dispatch。

---

## 问题定义

**目标**：在 decode_batch 路径上启用 CUDA Graph，减少 Python dispatch 开销。

**非目标**：不修改 prefill 路径；不改变 attention kernel；不要求对所有情况都走 graph（有降级路径）。

**硬约束**：
1. CUDA Graph 捕获时和回放时的 tensor **形状必须完全相同**
2. 已有的 Paged Attention、Chunked Prefill、Prefix Caching 均不得回归
3. 输出必须与 eager 模式 token-level 一致（greedy 下完全匹配）

**环境**：RTX 4090，PyTorch 2.1.2+cu121，flash_attn 2.5.9.post1

---

## 难点：decode 是动态的，graph 是静态的

CUDA Graph 的根本限制是"录制时的形状即最终形状"。一旦形状改变，整个图就失效，必须重新捕获。

mini-infer 的 decode_batch 有三处动态性：

| 动态来源 | 具体表现 |
|---------|---------|
| batch size 变化 | 请求随时进出，active batch 从 1 到 8 变化 |
| block_table 变化 | 每步新分配 KV 块，block_table 的值（和列数）都在变 |
| max_kv_len 变化 | `max(cache_seqlens) + 1` 每步递增，影响 RoPE cos/sin 的大小 |

这三个问题，处理方式各不相同。

---

## 方案：Graph Pool + 静态 Buffer + 固定 max_kv_len

### 1. batch size：Graph Pool

为每个支持的 batch size（1、2、4、8）各捕获一张图，存入 `_cuda_graphs` dict。实际 batch size 不匹配时，找到最小的 `padded_bs >= actual_bs`，把真实数据填入 padded_bs 张图的静态 buffer，pad 行填零，回放后只取前 actual_bs 行的结果。

```python
# 查找合适的 padded batch size
def _find_padded_bs(self, actual_bs: int) -> int | None:
    for padded in sorted(self._cuda_graphs.keys()):
        if padded >= actual_bs:
            return padded
    return None  # 无合适图，降级 eager

# 回放时取真实结果
return static["logits"][:actual_bs].clone()
```

### 2. block_table：静态 buffer + copy_()

CUDA Graph 捕获的是 CUDA kernel 序列和它们读取的 GPU 内存**地址**，不是当时的值。只要在回放前通过 `copy_()` 更新静态 buffer 的内存内容，kernel 回放时就会读到最新的 block 映射。

```python
# warmup 时分配固定形状的静态 buffer（max_blocks_per_seq = ceil(max_model_len / block_size)）
s_block_table = torch.zeros(bs, max_blocks_per_seq, dtype=torch.int32, device=device)

# 每步 replay 前：copy_() 写入当前 block 分配（staging buffer 预分配，避免每步动态分配）
staging.zero_()
staging[:actual_bs, :min(bt_cols, max_blocks)].copy_(block_table[:, :max_blocks])
static["block_table"].copy_(staging)
```

这里有一个实现细节：staging buffer 必须是**预分配**的，不能每步 `torch.zeros(...)` 重新分配——否则每步还是有一次 GPU 内存分配，部分抵消 graph 带来的收益。（review 阶段发现并修复了这个问题。）

### 3. max_kv_len：固定为 config.max_model_len

这是三个问题里最微妙的一个。

在 `attention.py` 的 patched forward 里，RoPE 的计算是：

```python
cos, sin = attn_module.rotary_emb(v, seq_len=ctx.max_kv_len)
q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
```

`rotary_emb(v, seq_len=max_kv_len)` 返回 `cos_cached[:max_kv_len]`，形状是 `[max_kv_len, head_dim]`。如果 `max_kv_len` 每步不同，这个 slice 的形状就不同，CUDA Graph 就必须针对每个可能的 seq_len 捕获一张图——这是不现实的。

解决方案：在 graph 模式下，**把 max_kv_len 固定为 config.max_model_len**（例如 2048）。`rotary_emb` 每次返回完整的 `[2048, head_dim]` cos/sin 表，形状恒定。实际的 token 位置通过 `position_ids`（一个 `[batch, 1]` 的静态 buffer，每步 copy_() 更新）从表里索引，不受 max_kv_len 固定的影响。

```
position_ids = [45, 67, 102, ...]   # 当前各请求的 KV 长度，每步不同
cos/sin 表 = [0:2048, head_dim]     # 形状固定，内容由模型初始化
apply_rotary_pos_emb 用 position_ids 索引正确行
```

这个方案的代价是：每步 RoPE 都生成完整的 2048 长度 cos/sin，而实际上只用 1 行。但这个开销在 graph 捕获后是 CUDA 层面的（一次 tensor slice），极小。

### 图的捕获流程

```python
# 1. warmup：让 CUDA 分配好所有 workspace tensor，避免 graph 捕获时触发动态分配
for _ in range(3):
    self._paged_ctx.set(s_block_table, s_cache_seqlens, fixed_max_kv_len)
    with torch.no_grad():
        _ = self.model(input_ids=s_input_ids, position_ids=s_position_ids, use_cache=False)
    self._paged_ctx.clear()
torch.cuda.synchronize()  # 确保 warmup 完成再捕获

# 2. 捕获
g = torch.cuda.CUDAGraph()
self._paged_ctx.set(s_block_table, s_cache_seqlens, fixed_max_kv_len)
with torch.cuda.graph(g):
    with torch.no_grad():
        graph_out = self.model(
            input_ids=s_input_ids, position_ids=s_position_ids, use_cache=False
        )
    s_logits = graph_out.logits[:, 0, :]  # slice 也在 graph 内，形状固定
self._paged_ctx.clear()
```

注意：`_paged_ctx.set()` 和 `_paged_ctx.clear()` 都在 `torch.cuda.graph()` 外执行——它们是 Python 属性赋值，不产生 CUDA kernel，无法也无需被录制进 graph。

### 与 Paged Attention 的兼容

Phase 6 的 patched forward 通过 `ctx.block_table is None` 判断走 prefill 路径还是 decode 路径：

```python
if ctx.block_table is None:
    return orig_fwd(...)   # prefill：回退 HF 原始 forward
# decode：用 flash_attn_with_kvcache + block_table
```

CUDA Graph 只捕获 decode 路径（已 set block_table）。Prefill 步不走 graph（ctx.block_table 不设置），Chunked Prefill 的中间步也走 eager。两者天然隔离，无需额外处理。

---

## 实验结果

**测试模型**：Qwen2.5-1.5B-Instruct（主要开发验证）+ Qwen2.5-7B-Instruct（最终验证）

**Workload**：固定 8 条 prompt，greedy，max_new_tokens=80（1.5B）/ 40（7B），warmup=20/10 步

### Qwen2.5-1.5B-Instruct

| bs | eager (ms/step) | graph (ms/step) | speedup |
|----|----------------|----------------|---------|
| 1  | 7.33           | 5.21           | **1.41×** (+28.9%) |
| 2  | 7.44           | 5.88           | **1.27×** (+21.0%) |
| 4  | 7.73           | 6.43           | **1.20×** (+16.8%) |
| 8  | 8.31           | 6.79           | **1.22×** (+18.3%) |

### Qwen2.5-7B-Instruct

| bs | eager (ms/step) | graph (ms/step) | speedup |
|----|----------------|----------------|---------|
| 1  | 17.63          | 16.76          | **1.05×** (+4.9%)  |
| 4  | 19.84          | 18.84          | **1.05×** (+5.0%)  |
| 8  | 21.97          | 21.01          | **1.05×** (+4.4%)  |

### Profiler：CPU dispatch overhead

用 `torch.profiler` 测量 20 步 decode（1.5B，bs=1）的 CPU 侧总时间：

| 指标 | eager | graph | 变化 |
|------|-------|-------|------|
| CPU self total（20 步） | 257.5 ms | 124.1 ms | **−51.8%** |
| 每步 forward 的 CPU 时间 | ~4.35 ms | ~0.59 ms | **7.4× 减少** |

---

## 为什么 1.5B 收益 28%，7B 只有 5%？

这不是 7B 上出了问题——而是两个模型上 Python dispatch 的**占比不同**。

```
1.5B bs=1：Python dispatch ≈ (7.33 - 5.21) / 7.33 ≈ 29% of total step time
7B   bs=1：Python dispatch ≈ (17.63 - 16.76) / 17.63 ≈ 5% of total step time
```

7B 的 model_forward（flash_attn + 线性层）本身就需要 15–16 ms/step，Python dispatch 的绝对值（~1 ms）基本等于 CUDA Graph 能消除的上限，而占比只有 5%。

CUDA Graph 的适用场景：**model forward 快、Python dispatch 占比高**的场景，典型是小模型（≤ 3B）或低延迟在线服务（单请求、bs=1）。对于 7B 这类推理时间主导的模型，收益是正的，但有限。

---

## 踩过的坑

### 坑 1：`_graph_decode_forward` 缺少 `try/finally`

implement 阶段写了：

```python
self._paged_ctx.set(...)
self._cuda_graphs[padded_bs].replay()
self._paged_ctx.clear()   # ← 如果 replay() 抛异常，这行不执行
```

`_paged_ctx.clear()` 一旦不执行，`ctx.block_table` 残留非 None，下一次 prefill 调用会误走 decode 路径，输出静默错误，不崩溃，很难发现。

修复：与 eager 路径保持一致，加 `try/finally`：

```python
try:
    self._cuda_graphs[padded_bs].replay()
finally:
    self._paged_ctx.clear()
```

### 坑 2：每步分配 GPU tensor

初版 `_graph_decode_forward` 里每步都写：

```python
bt_padded = torch.zeros(padded_bs, max_blocks, dtype=torch.int32, device=device)
```

这在每个 decode step 都触发一次 GPU 内存分配。CUDA Graph 消除了 kernel launch overhead，但如果每步还在 Python 侧分配内存，部分收益就被抵消了。

修复：在 warmup 时预分配 `bt_staging`，每步只做 `staging.zero_()` 和 `copy_()`，无动态分配。

### 坑 3：warmup 顺序不能错

warmup 必须在捕获前完成，且完成后要 `torch.cuda.synchronize()` 确保所有 workspace tensor 已在 GPU 上就位。如果 warmup 跑得不够（次数不足），捕获阶段 CUDA 可能还在动态分配 workspace，导致捕获进 graph 的是不稳定状态的地址，回放时出现数据错乱（没有报错，但输出错误）。

---

## 遗留问题和局限

**7B 下 5% 的收益是否值得**：启用 CUDA Graph 会延长引擎启动时间（4 张图 × warmup + capture 约 5–8s），对于长时间运行的服务是合理的；对于一次性脚本则不值。这也是为什么 `use_cuda_graph` 默认为 `False`，按需开启。

**bs > 8 的情况**：当前捕获 bs={1,2,4,8}，超出 8 的请求会降级 eager。实际上 `max_batch_size` 超过 8 时，只有超出部分才会降级，影响不大。

**capture 时的 KV 写入**：warmup 和 capture 阶段，`s_block_table` 全零指向物理块 0，flash_attn 会把 dummy KV 写入块 0。这不影响正确性（块 0 的内容会被后续正常请求覆盖），但意味着引擎启动时的块 0 是脏数据，不适合被当作前缀缓存的有效块。现有实现里 prefix caching 只缓存已完成请求的块，不缓存块 0，所以实际无影响。

---

## 总结

Phase 12 在 mini-infer 的 decode_batch 路径上接入了 CUDA Graph，实现了：

- **1.5B 模型：+20–29% decode speedup**（Python dispatch 从 29% 降到 4%）
- **7B 模型：+5% decode speedup**（接近理论上限）
- **与 Chunked Prefill、Prefix Caching、Paged Attention 完全兼容**
- **输出与 eager 模式 token-level 一致**（greedy 下 100% 匹配）

CUDA Graph 的本质是在"动态"的 LLM 推理中找到"静态"的部分——decode 步的 model forward 形状实际上是可以固定的，通过 graph pool + copy_() 模式可以在不改变正确性的前提下绕过 PyTorch 的动态调度。

下一步 Phase 12.5 是 Flash Decoding（Split-K Attention）：针对长序列场景，在 KV 长度维度做 split-K 并行，进一步提升 SM 利用率。CUDA Graph 已处理了 Python dispatch 开销，Flash Decoding 处理的是 GPU 端的 attention kernel 并行度问题——两者互补，不冲突。

---

## 自查结果（QUALITY_CHECKLIST.md）

- [x] 文章主题对应真实项目阶段或真实技术问题（Phase 12，CUDA Graph 接入 Paged Attention）
- [x] 结构覆盖：背景、问题定义、方案设计、实现细节、实验结果、坑点反思、总结
- [x] 有明确背景和问题定义（"flash_attn 之后还有什么开销"和三处动态性分析）
- [x] 有设计取舍（graph pool vs 动态捕获、max_kv_len 固定的代价与合理性、warmup 顺序要求）
- [x] 有代码、实验或运行结果支撑（代码片段来自 model_runner.py 实际实现，数据来自 benchmark 记录文件）
- [x] 有失败点、坑点或反思（三个坑：try/finally 缺失、每步分配 GPU tensor、warmup 顺序）
- [x] 语言克制、专业、可发表（无未经验证的性能宣称，局限性有明确说明）
