# True PagedAttention：一次 `.item()` 把性能打回 3.7% 的教训

这是 mini-infer 项目的第六篇复盘。

Phase 5 做完 profiling 之后，decode 路径的全貌是这样的：

```
block tensor → gather_batch_kv(0.31ms) → k_batch[28层]
                                               ↓
                           DynamicCache 预填充（28层 list.append，纯 Python，无 GPU 时间）
                                               ↓
   model_forward(17.89ms) = Q/K/V proj × 28 + flash_sdp × 28 + 28× DynamicCache.update(k, v)
                                               ↓
                           write_decode_kv(0.19ms) → 写回 block tensor
```

gather 只占 1.7%，model_forward 占 97%。理论上消掉 gather 和 write_kv，能提升 2-3 个点，从 88.4% 涨到 90-93%。

Phase 6 的目标是：把 flash_attn 2.5+ 的 `block_table` 参数用起来，让 attention kernel 直接在 block tensor 上寻址，彻底去掉 gather → DynamicCache → write_kv 这三段。

最终结果是 100.0% HF baseline。但中间首次 benchmark 只跑出了 3.7%。

---

## 一、为什么要做 True PagedAttention

Phase 1-5 里的 KV cache 管理是"外挂式"的：用 PyTorch 的 advanced indexing 把分散在 block tensor 里的 KV 拼成连续的 dense tensor，喂给模型，再把结果写回 block tensor。flash_attn 在里面做了高效的 attention 计算，但 KV 的搬运还是在外面完成的。

`flash_attn_with_kvcache` 的 `block_table` 参数改变了这一点：

```python
output = flash_attn_with_kvcache(
    q,           # (batch, 1, num_heads, head_dim)
    k_cache,     # (num_blocks, block_size, num_kv_heads, head_dim) — 全量 block tensor
    v_cache,     # 同上
    k=k_new,     # (batch, 1, num_kv_heads, head_dim) — 当前 token 的新 K
    v=v_new,
    cache_seqlens=cache_seqlens,  # (batch,) int32 — 各请求当前 cache 长度
    block_table=block_table,      # (batch, max_blocks) int32 — 每请求的 block 映射表
    causal=True,
    softmax_scale=1.0 / math.sqrt(head_dim),
)
```

这一个调用完成三件事：
1. 把 `k_new`/`v_new` in-place 写入 `k_cache`/`v_cache` 的正确 block 位置
2. 根据 `block_table` 和 `cache_seqlens` 计算 attention（无需 gather 成 dense tensor）
3. 返回 attention 输出

gather、DynamicCache、write_kv 三段都消失了。

---

## 二、设计取舍：永久 patch 还是条件分支

替换 attention 有两条路：

**方案 A：改 model_runner.decode_batch，绕过模型 forward，逐层手写 Transformer 步骤。**

直接，但维护成本高。每次模型升级（Qwen2.5 → Qwen3）都要重写。

**方案 B：永久 patch 模型各层的 `self_attn.forward`，在函数内部区分 prefill/decode 路径。**

侵入性低，模型的其余部分（MLP、LayerNorm、embedding）不动。代价是引入一个可变的全局共享状态（`PagedDecodeContext`）来传递 block_table 和 cache_seqlens。

选了方案 B。理由是：prefill 路径完全不动（`ctx.block_table is None` 时直接调用原始 HF forward），只有 decode 路径走新分支。

**PagedDecodeContext** 是核心的状态载体：

```python
class PagedDecodeContext:
    def __init__(self) -> None:
        self.block_table: torch.Tensor | None = None
        self.cache_seqlens: torch.Tensor | None = None
        self.max_kv_len: int = 0  # 预计算，避免每层各调用一次 .item()

    def set(self, block_table, cache_seqlens, max_kv_len: int) -> None:
        self.block_table = block_table
        self.cache_seqlens = cache_seqlens
        self.max_kv_len = max_kv_len

    def clear(self) -> None:
        self.block_table = None
        self.cache_seqlens = None
        self.max_kv_len = 0
```

`block_table is None` 是 prefill/decode 的路由信号。每次 decode forward 前 `set()`，结束后 `clear()`，用 `try/finally` 保证异常时也不污染状态。

---

## 三、patched_forward 的实现

28 层 attention 各自有一个 `patched_forward`。核心逻辑：

```python
def patched_forward(hidden_states, position_ids=None, ...):
    # prefill：block_table 未设置，走原始 HF forward
    if ctx.block_table is None:
        return orig_fwd(hidden_states, ...)

    # decode：paged attention
    bsz, q_len, _ = hidden_states.shape

    # Q/K/V 投影
    q = attn_module.q_proj(hidden_states)
    k = attn_module.k_proj(hidden_states)
    v = attn_module.v_proj(hidden_states)

    # reshape: [batch, num_heads, 1, head_dim]
    q = q.view(bsz, 1, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)

    # RoPE（手动 apply，不用 flash_attn 内置参数——Qwen2 格式兼容性风险）
    cos, sin = attn_module.rotary_emb(v, seq_len=ctx.max_kv_len)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    # flash_attn 格式: [batch, 1, num_heads, head_dim]
    q_fa = q.transpose(1, 2)
    k_fa = k.transpose(1, 2)
    v_fa = v.transpose(1, 2)

    # paged decode attention（同时 in-place 写入 KV 到 block cache）
    attn_out = flash_attn_with_kvcache(
        q_fa, k_cache[layer], v_cache[layer],
        k=k_fa, v=v_fa,
        cache_seqlens=ctx.cache_seqlens,
        block_table=ctx.block_table,
        causal=True,
        softmax_scale=1.0 / math.sqrt(head_dim),
    )

    # 输出投影
    attn_out = attn_out.reshape(bsz, 1, -1)
    return attn_module.o_proj(attn_out), None, None
```

RoPE 没有用 flash_attn 内置的 `rotary_cos/sin` 参数，而是手动调用 Qwen2 自己的 `rotary_emb` 模块。原因：flash_attn 内置 RoPE 要求特定的 cos/sin 格式（interleaved），Qwen2 的格式不同，贸然使用会导致数值错误。

---

## 四、首次 benchmark：3.7%

实现完成，GPU 测试通过（paged 输出与 HF greedy 逐 token 一致），跑 benchmark：

```
mini-infer throughput = 3.7% of HF baseline
15.0 tok/s  vs  405.5 tok/s
```

这个数字明显不对。Phase 3 是 88.4%，Phase 6 理应更好而不是更差。

用 profiler 看了 batch=4, 20 decode steps 的数据：

```
model_forward    19 calls    10013.66ms    527.035ms/step
```

Phase 5 同等配置是 17.9ms/step。Phase 6 是 527ms/step——慢了 **29 倍**。

---

## 五、根因：`.item()` 在 28 层内各调用一次

定位到 `patched_forward` 里的这行：

```python
max_kv_len = int(ctx.cache_seqlens.max().item()) + 1
cos, sin = attn_module.rotary_emb(v, seq_len=max_kv_len)
```

`max_kv_len` 是 RoPE 所需的序列长度上限——需要覆盖到当前批次中已缓存的最大位置。

问题在 `.item()`。

`.item()` 把 GPU tensor 的标量值读到 CPU，这需要等待 GPU 上所有已提交的 kernel 执行完毕，才能安全地把数据传到 CPU。这是一次**完整的 CPU-GPU 同步**。

`patched_forward` 被每层各调用一次，Qwen2.5-7B 共 28 层。每个 decode step = 28 次 `.item()` = 28 次 CPU-GPU sync。

每次 sync 约 18ms（RTX 4090 上的典型值），28 次 × 18ms ≈ **504ms**——完全覆盖了 profiler 测到的 527ms/step，model_forward 真正的计算时间（17ms）淹没在同步等待中。

**修复**：把计算移到 `decode_batch()` 里，在所有层的 forward 开始前做一次：

```python
# decode_batch() 中，在 model.forward() 之前
max_kv_len = int(cache_seqlens.max().item()) + 1  # 一次 sync
self._paged_ctx.set(block_table, cache_seqlens, max_kv_len)
```

`patched_forward` 改成直接读 `ctx.max_kv_len`：

```python
cos, sin = attn_module.rotary_emb(v, seq_len=ctx.max_kv_len)  # 无 sync
```

28 次 sync → 1 次。

---

## 六、修复后的结果

```
=================================================================
  Phase 6 True PagedAttention vs HF Transformers 对比
=================================================================
  指标                     mini-infer Phase 6          HF baseline
  --------------------------------------------------------------
  Throughput (tok/s)                  406.3                406.4
  TTFT (ms)                            18.8                 19.9
  TPOT (ms/tok)                        2.46                 2.46
  Peak Mem (GB)                       18.71                15.87
  --------------------------------------------------------------
  mini-infer throughput = 100.0% of HF baseline
=================================================================
```

（环境：Ubuntu 24.04，RTX 4090 × 2，Qwen2.5-7B-Instruct float16，batch=8，max_new_tokens=128。mini-infer 跑 cuda:0，HF baseline 跑 cuda:1。flash_attn 2.5.9.post1。）

Profiler 确认了优化路径：

```
model_forward    29 calls    17.06ms/step
```

以及 top-20 中新出现的 kernel：

```
flash_fwd_splitkv_kernel    812 calls    8.72μs/call
```

`812 = 29步 × 28层`，每层每步都在走 flash_attn 路径。`gather_batch_kv` 和 `write_decode_kv` 两个标签完全消失。

---

## 七、为什么能达到 100%

Phase 5 预测 Phase 6 能提升 2-5%（从 88.4% 到 90-93%），实际达到了 100%。

对比 Phase 5 的 profiling 数据（batch=8）：

| 操作 | Phase 5 | Phase 6 |
|------|---------|---------|
| gather_batch_kv | 0.31ms/step | 0 |
| write_decode_kv | 0.19ms/step | 0 |
| model_forward | 17.89ms/step | 17.06ms/step |

gather + write 合计 0.5ms，只占 18.4ms 总时间的 2.7%，不足以解释从 88.4% 到 100% 的 11.6 个百分点。

差距来自两处：

**1. model_forward 本身也快了一点**：17.89ms → 17.06ms（4.6%）。flash_attn_with_kvcache 比 SDPA + DynamicCache.update() 的组合略快。SDPA 使用的是 `_efficient_attention_forward`（prefill 时适用），decode 时 token 序列长度为 1，flash_attn 对单 token decode 的 paged 访问做了特殊优化（`flash_fwd_splitkv_kernel`）。

**2. Phase 3 benchmark 和 Phase 6 benchmark 跑在不同环境**：Phase 3 用的 PyTorch 2.3.1，Phase 6 用的 PyTorch 2.1.2+cu121；HF baseline 也因此不同（Phase 3 测到 408.9 tok/s，Phase 6 测到 406.4 tok/s）。两次测量的"HF baseline"本身就有轻微波动，这会影响相对比率。

不能确定 11.6pp 的提升全部来自 flash_attn 的优势——环境差异、测量噪声都有贡献。结论只能说：在当前 workload（batch=8，max_new_tokens=128）下，Phase 6 的 throughput 与 HF baseline 在同一数量级，差距在测量误差范围内。

---

## 八、剩余代价：+2.84 GB 显存

mini-infer 峰值显存 18.71 GB，HF 15.87 GB，差了 2.84 GB。

计算下来，预分配的 KV cache 需要：

```
200 blocks × 256 tokens/block × 28 layers × 2(K+V) × 4 kv_heads × 128 head_dim × 2 bytes(fp16)
≈ 2.93 GB
```

HF 的 DynamicCache 按需分配，不预留，所以低。这是 Paged KV Cache 的固有代价——用固定显存换取 O(1) 分配和无碎片。生产场景中，这个内存可以用于接收更多请求，而不是"浪费"——只是测量时 peak memory 会比 HF 高。

---

## 九、这个 bug 为什么在 review 里没被发现

`.item()` 在 `patched_forward` 里，三轮 code review 都没有把它标记为阻塞性问题。

原因是它不是正确性 bug——每次调用的结果是对的，只是慢。静态分析代码时，"一个函数里的 `.item()` 被调用 28 次" 和 "这 28 次的同步开销达到 504ms" 之间的距离，需要运行 profiler 才能量化。review 能发现接口错误、状态污染、逻辑分支，但性能的量级通常只能靠实测。

这也是 infer-benchmark 阶段存在的意义——不只是为了得到一个好看的数字，而是为了发现代码里藏着的性能陷阱。

---

## 总结

Phase 6 用 `flash_attn_with_kvcache` 的 `block_table` 接口实现了 True PagedAttention：

- gather_batch_kv、write_decode_kv 从 decode 路径完全消失
- batch=8 throughput 从 88.4% HF → **100.0% HF**
- prefill 路径零改动，patch 通过 `ctx.block_table is None` 自动路由

最重要的教训：**在被 N 层各调用一次的函数里，任何涉及 CPU-GPU sync 的操作都是 O(N) 的开销**。`.item()`、`.numpy()`、Python 打印 GPU tensor，都会触发同步。28 层 × 18ms = 504ms，足以把一个本来正确的优化变成 3.7% 的灾难。

