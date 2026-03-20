# 知识专题：flash_attn_with_kvcache 与 Paged Decode Attention

这个文件深度解析 flash_attn 2.5+ 的 block_table 接口，以及在多层 Transformer 中使用它时的核心工程陷阱。来自 mini-infer Phase 6 的真实实现经验。

---

## 主题

flash_attn_with_kvcache 的 block_table 接口如何实现 True PagedAttention，以及 CPU-GPU 同步问题的定位与修复。

---

## 一、问题定义

传统的 Paged KV Cache 实现（mini-infer Phase 1-5）需要三段独立操作：

```
block tensor → gather_batch_kv → dense KV tensor
                                        ↓
                         model.forward(past_key_values=DynamicCache)
                                        ↓
                         write_decode_kv → block tensor
```

**问题**：
1. `gather_batch_kv` 是独立的 GPU kernel（高级索引），每步都会启动一次 Python→CUDA 调度
2. `write_decode_kv` 同理
3. `DynamicCache` 在 model.forward 内部还有 `aten::cat`（HF 的 `DynamicCache.update()`）

Phase 5 profiling 量化结果（batch=8）：
- gather_batch_kv: 0.31ms/step（1.7%）
- write_decode_kv: 0.19ms/step（1.1%）
- model_forward: 17.89ms/step（97.2%）

Phase 6 目标：用 flash_attn_with_kvcache 的 block_table 参数，让 attention kernel 直接从 block tensor 寻址，一次调用完成 KV 写入 + attention 计算。

---

## 二、核心原理

### flash_attn_with_kvcache API

```python
from flash_attn import flash_attn_with_kvcache

output = flash_attn_with_kvcache(
    q,                # (batch, 1, num_heads, head_dim)  — 当前 decode token 的 Q
    k_cache,          # (num_blocks, block_size, num_kv_heads, head_dim)  — 全局 block tensor
    v_cache,          # 同上
    k=k_new,          # (batch, 1, num_kv_heads, head_dim)  — 当前 token 的新 K
    v=v_new,          # 同上
    cache_seqlens=cache_seqlens,  # (batch,) int32 — 每请求的当前 cache 长度（写入位置）
    block_table=block_table,      # (batch, max_blocks_per_seq) int32 — 物理块映射表
    causal=True,
    softmax_scale=1.0 / math.sqrt(head_dim),
)
# output: (batch, 1, num_heads, head_dim)
# 副作用：k_cache / v_cache 被 in-place 修改（k_new/v_new 已写入正确位置）
```

**一次调用完成三件事：**
1. 根据 `block_table` 和 `cache_seqlens`，将 `k_new`/`v_new` 写入 `k_cache`/`v_cache` 的正确 block 位置
2. 用 `block_table` 寻址读取历史 KV，无需 gather 成 dense tensor
3. 计算 causal attention，返回输出

### block_table 语义

`block_table[b, i]` = 请求 b 的第 i 个逻辑块对应的物理块编号。

```
请求 0: 逻辑块 [0, 1, 2]  → 物理块 [5, 12, 3]
请求 1: 逻辑块 [0, 1]     → 物理块 [7, 2]
```

flash_attn kernel 用这个映射直接从全局 block tensor 读取正确的 KV，无需任何 gather。

### block_size 约束

flash_attn_with_kvcache 内核要求 `k_cache.shape[1] % 256 == 0`，即 block_size 必须是 256 的倍数。

实践中用 block_size=256：每块可存 256 个 token。200 块 × 256 = 51200 token 总容量。

运行时错误提示：
```
RuntimeError: Paged KV cache block size must be divisible by 256
```

---

## 三、工程实现方式

### 状态共享：PagedDecodeContext

28 层 attention 的 `patched_forward` 各自独立执行，但需要共享同一个 `block_table`、`cache_seqlens` 和 `max_kv_len`。

解决方案：可变的共享对象，在 `decode_batch()` 里 `set()`，`model.forward()` 期间各层读取，结束后 `clear()`。

```python
class PagedDecodeContext:
    def __init__(self) -> None:
        self.block_table: torch.Tensor | None = None
        self.cache_seqlens: torch.Tensor | None = None
        self.max_kv_len: int = 0  # 预计算 int，不是 GPU tensor

    def set(self, block_table, cache_seqlens, max_kv_len: int) -> None:
        self.block_table = block_table
        self.cache_seqlens = cache_seqlens
        self.max_kv_len = max_kv_len

    def clear(self) -> None:
        self.block_table = None
        self.cache_seqlens = None
        self.max_kv_len = 0
```

`ctx.block_table is None` 同时作为 prefill/decode 的路由信号：`None` = prefill（走原始 HF forward），非 `None` = decode（走 paged attention）。

### RoPE 处理

flash_attn_with_kvcache 有内置的 `rotary_cos/sin` 参数，但 Qwen2 的 RoPE 格式（非 interleaved）与之不兼容。

做法：手动 apply RoPE。

```python
# position_ids = 每个请求的新 token 位置 = cache_seqlens.long().unsqueeze(1)
cos, sin = attn_module.rotary_emb(v, seq_len=ctx.max_kv_len)
q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
# 再传入 flash_attn
```

`seq_len=ctx.max_kv_len` 确保 cos/sin 覆盖到最大位置（`max(cache_seqlens) + 1`）。

### decode_batch 的完整调用序列

```python
# 1. 确保下一个位置有物理块
self.kv_cache.ensure_next_slot(request_ids)
# 2. 构建 block_table 和 cache_seqlens（CPU int tensor）
block_table, cache_seqlens = self.kv_cache.build_block_tables(request_ids)
# 3. 预计算 max_kv_len（唯一的 .item() 调用）
max_kv_len = int(cache_seqlens.max().item()) + 1
# 4. 注入共享状态
self._paged_ctx.set(block_table, cache_seqlens, max_kv_len)

try:
    out = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    self.kv_cache.advance_seq_lens(request_ids)  # flash_attn in-place 写入后递增
finally:
    self._paged_ctx.clear()  # 异常时也必须清除，防止下次 prefill 路由错误
```

---

## 四、设计取舍

### 永久 patch vs 逐层手写 Transformer

| 方案 | 优点 | 缺点 |
|------|------|------|
| 永久 patch 各层 self_attn.forward | 只动 attention，模型其余部分（MLP、LayerNorm）不碰；prefill 路径零改动 | 引入全局可变状态（PagedDecodeContext）；patch 逻辑需理解 Qwen2 内部结构 |
| 逐层手写 Transformer 步骤 | 代码直观，无隐式状态 | 维护成本高，每次模型升级（Qwen2→Qwen3）都要重写 |

mini-infer 选择永久 patch，理由：prefill 路径完全不受影响，只有 decode 路径改变。

### 预分配 vs 动态分配 KV Cache

mini-infer 预分配全量 block tensor：
```
200 blocks × 256 tokens × 28 layers × 2(K+V) × 4 kv_heads × 128 head_dim × fp16
= 2.93 GB（固定开销）
```

HF DynamicCache 按需分配，峰值显存更低（当前 batch 用多少分多少）。

Paged KV Cache 的取舍：以固定显存换取 O(1) block 分配、无碎片、支持 preemption。生产场景中，这个固定开销是用来接收更多并发请求的容量。

---

## 五、常见误区

### 误区 1：`.item()` 在多层函数里"只是一个小操作"

`.item()` 把 GPU tensor 标量读到 CPU，强制等待 GPU 上所有已提交 kernel 完成——这是一次完整的 CPU-GPU 同步。

在被 N 层各调用一次的 `patched_forward` 里，每个 `.item()` = N 次 sync。

Qwen2.5-7B：28 层 × 18ms/sync = 504ms/step。
这足以把本来正确的实现变成 3.7% HF baseline（15 tok/s vs 405 tok/s）。

**修复**：只在 `decode_batch()` 里做一次 `.item()`，结果存为 Python int，通过 `PagedDecodeContext.max_kv_len` 传入所有层。

同类陷阱：`.numpy()`、`print(gpu_tensor)`、`bool(gpu_tensor)` 都会触发同步。

### 误区 2：block_size 只是一个配置参数，随便改

flash_attn_with_kvcache 内核有硬约束：`block_size % 256 == 0`。

如果原来 block_size=16，改成 256 后每块大了 16×，相同 num_gpu_blocks 的显存开销也增加 16×。需要同步降低 num_gpu_blocks，否则 OOM。

### 误区 3：flash_attn 内置 RoPE 参数对所有模型通用

flash_attn 内置 `rotary_cos/sin` 要求特定的 interleaved 格式。Qwen2 的 RoPE 实现不是这个格式，用内置参数会导致数值错误（生成结果异常，但不报错，难以察觉）。

安全做法：用模型自己的 `rotary_emb` 模块手动 apply，然后把已旋转的 Q/K 传给 flash_attn。

### 误区 4：性能问题在 code review 中能被发现

`.item()` 的问题在三轮 code review 中均未被发现。原因：它是性能问题，不是正确性问题。代码功能正确，测试通过，只有运行 profiler 才能量化 28 次 sync 的实际开销。

benchmark 的意义不只是得到数字，更是发现代码里隐藏的性能陷阱。

---

## 六、和 mini-infer 的关系

Phase 6 的实现文件：
- `mini_infer/attention.py` — PagedDecodeContext + paged_decode_attention + patch_model_for_paged_decode
- `mini_infer/kv_cache.py` — ensure_next_slot / build_block_tables / advance_seq_lens
- `mini_infer/model_runner.py` — decode_batch 替换为 paged 路径
- `mini_infer/config.py` — block_size 默认值 16 → 256
- `benchmarks/benchmark_flash.py` — Phase 6 专用 benchmark

Phase 6 结果（batch=8，Qwen2.5-7B，float16，RTX 4090）：

| 指标 | Phase 3 | Phase 6 | HF baseline |
|------|---------|---------|-------------|
| Throughput | 361.3 tok/s（88.4%）| **406.3 tok/s（100.0%）** | 406.4 tok/s |
| gather_batch_kv | 0.31ms/step | 0（消除）| — |
| write_decode_kv | 0.19ms/step | 0（消除）| — |
| model_forward | 17.89ms/step | 17.06ms/step | — |
| Peak Mem | 16.08 GB | 18.71 GB（+2.93 GB KV cache）| 15.87 GB |

---

## 七、进一步阅读

- flash_attn 源码：`flash_attn/flash_attn_interface.py` — `flash_attn_with_kvcache` 的参数文档
- vLLM 的 PagedAttention 实现：`vllm/attention/backends/flash_attn.py`（对比生产级实现）
- FlashAttention-2 论文（Tri Dao, 2023）：IO-aware attention 的核心思想
- mini-infer Phase 5 profiling 笔记：`本地资料/知识笔记/phase5-profiling-decode.md`（Phase 6 的出发点）
