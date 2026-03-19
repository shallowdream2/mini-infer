# 知识专题：PyTorch Advanced Indexing 与 Paged KV Cache Gather

本文深度解析 mini-infer Phase 3 的核心优化：用向量化 advanced indexing 替代嵌套 Python 循环完成 Paged KV Cache 的 gather 操作，以及与 PagedAttention（flash_attn 2.5+）的本质区别。

---

## 主题

**Paged KV Cache 的 gather 操作：从 Python 循环到 CUDA kernel，以及真正的 PagedAttention 是什么**

---

## 一、问题定义

Paged KV Cache 把每个请求的 KV 分散存储在固定大小的 block 中，解决了显存管理问题。但在 batch decode 阶段，需要把多个请求的 KV 聚合成连续的 dense tensor，才能传给 HF 模型做一次 batch forward。

这个聚合操作（gather）面临的挑战：
- 不同请求 seq_len 不同（需要左填充对齐）
- 每个请求的 block 分散在 block pool 各处（物理地址不连续）
- 每步 decode 都要重新 gather（动态序列，不能提前缓存）
- 层数 × batch × seq_len 的规模下，Python 循环会成为 CPU 瓶颈

---

## 二、核心原理

### 2.1 Block 存储结构

```
k_cache[layer_idx]  shape: [num_gpu_blocks, block_size, num_kv_heads, head_dim]
```

每个请求的 BlockTable 是 `list[int]`，记录占用的物理块号。逻辑 token 位置 `pos` 对应：

```python
block_idx = pos // block_size   # 第几个 block
slot_idx  = pos % block_size    # block 内第几个 slot
phys_blk  = block_table[block_idx]  # 物理块号
kv = k_cache[layer][phys_blk, slot_idx]  # 物理地址
```

### 2.2 Advanced Indexing 原理

PyTorch advanced indexing 允许用 tensor 作为索引：`x[idx_tensor]` 会一次性完成所有位置的 gather，底层对应一次 CUDA gather kernel，而不是 Python 层面的逐个取值。

关键：两个同 shape 的索引 tensor `[phys_blocks, slot_indices]` 可以同时索引 `k_cache[l]` 的前两维，结果 shape 与索引 shape 一致。

### 2.3 左填充数学

batch 中每个请求 seq_len 不同，输出对齐到 max_seq_len（左填充）：

```
output_position:   0  1  2  3  4  5   (max_seq_len=6)
req_a (len=6):     0  1  2  3  4  5   (无填充)
req_b (len=3):     P  P  P  0  1  2   (左填充 3 位)

token_position[b, i] = i - (max_seq_len - seq_len[b])
```

负值是填充区，clamp 到 0 后 gather 出"无效值"，用 `valid_mask_f=0` 乘掉。

---

## 三、工程实现方式

### Phase 2：Python 嵌套循环

```python
for l in range(num_layers):          # 28
    for b, rid in enumerate(request_ids):   # 8
        for i in range(seq_len):            # 128
            phys = block_table[rid][i // bs]
            slot = i % bs
            k_dense[l, b, i] = k_cache[l][phys, slot]  # 每次一次小索引
```

总迭代：28 × 8 × 128 = 28,672 次。每次小索引都有 Python 调度开销。

### Phase 3：向量化

```python
# 预计算所有索引（一次性，两个 [batch, max_seq_len] tensor）
block_table_tensor = ...  # [batch, max_num_blocks]
token_positions = out_positions - (max_seq_len - seq_lens_t)
token_pos_clamped = token_positions.clamp(min=0)
block_indices = token_pos_clamped // block_size
slot_indices  = token_pos_clamped % block_size
phys_blocks   = block_table_tensor[batch_range, block_indices]  # [batch, max_seq_len]

valid_mask_f = (token_positions >= 0).unsqueeze(-1).unsqueeze(-1).to(dtype=cache_dtype)

for l in range(num_layers):   # 28（固定，不随 batch/seq_len 增长）
    k_tokens = k_cache[l][phys_blocks, slot_indices] * valid_mask_f  # 1次 CUDA kernel
    k_batch.append(k_tokens.permute(0, 2, 1, 3))
```

Python 循环仅 28 次（层数），每层内部是 1 次 CUDA advanced indexing。

### 关键细节

**valid_mask 的 dtype**：`token_positions >= 0` 得到 bool tensor，必须 `.to(dtype=cache_dtype)` 转换为 float，否则乘法触发隐式类型转换，可能有精度问题。

**新 KV 提取**：DynamicCache forward 后 `out.past_key_values.key_cache[l]` shape 为 `[batch, num_kv_heads, max_seq_len+1, head_dim]`，最后位置 `[:, :, -1, :]` 是新 token 的 KV。

**及时 del**：gathered KV tensor 是 `[layers, batch, num_kv_heads, max_seq_len, head_dim]` 量级，写回 block tensor 后立即 del 释放显存，避免峰值叠加。

---

## 四、设计取舍

### 向量化 gather vs 真正的 PagedAttention

| | Phase 3 向量化 gather | flash_attn 2.5+ PagedAttention |
|--|----------------------|-------------------------------|
| 做什么 | 把分散 block 复制到 dense tensor，再做标准 attention | 在 attention kernel 内部直接按 block_table 寻址，不产生 dense tensor |
| KV 复制 | 每步 decode 都复制，O(batch × seq_len × layers × head_dim) | 无复制 |
| 显存带宽 | 复制消耗带宽 | 更节省 |
| 实现复杂度 | PyTorch 原生，无外部依赖 | 需要 flash_attn 2.5+ 或自写 CUDA/Triton kernel |
| 当前环境支持 | ✅（任意 PyTorch 版本） | ❌（flash_attn 2.3.6，2.5 才有 `block_tables`） |

Phase 3 的向量化消除了 **Python 循环开销**，但没有消除 **gather 本身的内存拷贝开销**。这是 batch=8 还有 ~12% 差距的根本原因。

### 左填充 vs 右填充

decode 阶段用左填充，因为 HF `attention_mask` 标记哪些位置有效，左填充保证真实 token 靠右对齐，新 token（input_ids 的那个 token）自然接在最后，position_ids 不需要特殊处理。右填充需要额外处理 position_ids，容易出错。

### DynamicCache vs tuple past_key_values

transformers 4.40+ 推荐 DynamicCache，tuple 格式已弃用。但 transformers 4.43.4 的弃用警告在 `model(use_cache=True)` 不传 `past_key_values` 时从模型内部触发，而不是从调用方触发。调用方必须显式传 `DynamicCache()` 实例（哪怕是空的）来告知模型期望格式，否则模型内部走旧路径并发出警告。

---

## 五、常见误区

**1. "向量化 gather = 消除了 KV 复制"**

错误。向量化把"28k 次 Python 小索引"变成"28 次 CUDA kernel gather"，依然是复制。消除复制需要 PagedAttention（注意力计算内直接寻址 block）。

**2. "DynamicCache warning 是因为 decode_batch 传了 tuple"**

错误（至少在 transformers 4.43.4）。warning 出在 prefill，是模型内部在 `past_key_values=None` 时自行构造 tuple 然后 "警告自己"。decode_batch 先改好了，warning 依然存在。

**3. "left-padding 位置可以不 clamp，直接用负 index"**

Python/PyTorch 支持负索引（`-1` 表示最后一个），不 clamp 的话负数 token_position 会索引到 block pool 的最后几个 block，造成数据污染。必须 clamp 到 0 再用 mask 置零。

**4. "valid_mask 是 bool，直接乘 float16 没问题"**

有隐患。bool tensor 乘 float16 会先提升类型，但不同 PyTorch 版本和设备上的行为可能有差异。显式 `.to(dtype=float16)` 更安全，也更明确。

---

## 六、和 mini-infer 的关系

Phase 3 实现了向量化 gather（`kv_cache.py`），消除了 Python 循环，是 batch decode 性能从 HF 49% 提升到 88% 的关键。

剩余 ~12% 差距的消除路径：
1. **flash_attn 2.5+ block_tables**：完整的 PagedAttention，attention kernel 内直接寻址
2. **自写 Triton gather kernel**：保持现有流程但用 Triton 优化 gather kernel

两个路径都需要额外工作量，Phase 3 在当前环境约束下选择了 PyTorch 原生向量化，达到了合理的性能提升。

---

## 七、进一步阅读

- vLLM PagedAttention 论文（Kwon et al., 2023）：真正的 PagedAttention 设计
- flash_attn 2.5 release notes：`block_tables` 参数的引入
- PyTorch advanced indexing 文档：`gather`、advanced indexing 的语义和性能特征
- transformers `DynamicCache` 源码：`update()` 和 `key_cache[l]` 的内存布局
