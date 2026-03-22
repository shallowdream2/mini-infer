# 知识专题：Prefix Caching 与 RadixAttention

## 主题

LLM 推理系统中的前缀 KV 缓存：block 粒度缓存、hash 设计、引用计数生命周期管理，以及从简单 hashmap 到 vLLM radix tree 的演进路径。

---

## 一、问题定义

**场景**：Agent/RAG 服务中，大量请求共享相同的前缀（system prompt、检索文档、few-shot 示例）。

**成本**：每个请求从头跑 prefill，相同前缀的 KV 矩阵被反复计算。设 system prompt = 512 tokens，用户消息 = 32 tokens：
- 无 prefix cache：每次 TTFT 含 prefill(544 tokens)
- 有 prefix cache 命中：TTFT ≈ prefill(32 tokens)，节省 94% prefill 时间

**核心问题**：KV 结果对相同的 token 序列是确定性的（greedy/temperature=0 时精确相同；采样时近似可复用）。复用 KV 不影响生成质量，只影响延迟。

---

## 二、核心原理

### 为什么是 block 粒度而非 token 粒度？

Paged KV Cache 的物理组织已经是 block 粒度。以 block 为单位缓存的收益：

1. **零拷贝命中**：命中时直接把物理块 ID 放入新请求的 block_table，无需移动 KV 数据
2. **与现有分配器对齐**：FreeBlockPool 管理的单位是 block，引用计数和生命周期管理更自然

代价：缓存粒度只到 block 边界，不足一块的 tail 部分不缓存，每次 miss 时需重新计算。

### hash 的链式设计

每个 block 的 hash 嵌入前驱历史：

```
hash[0] = H(0       ‖ tokens[0 : bs])
hash[1] = H(hash[0] ‖ tokens[bs: 2bs])
hash[i] = H(hash[i-1]‖ tokens[i*bs : (i+1)*bs])
```

若不链式：位置 k 和位置 m 的相同 token 块（不同上下文）得到相同 hash，触发错误命中，生成结果不对。链式 hash 等价于把从 root 到节点的完整路径信息压缩进单个 hash 值——这正是 radix tree 的前缀内嵌思想，只是用 hash 压缩了 tree 的路径。

### 为什么不用 Python hash()？

Python 的 `hash()` 受 `PYTHONHASHSEED` 控制：3.3+ 默认开启随机化，每次进程启动 hash 值不同。结果：
- 缓存内容跨进程无法共享
- 多进程测试（pytest）hash 偶发冲突
- 进程重启后缓存完全失效

正确做法：`hashlib.sha256`（确定性，低碰撞，标准库不需要额外依赖）取前 8 字节作为 64-bit int。

---

## 三、工程实现方式

### 数据结构

```python
_prefix_cache: dict[int, int]             # block_hash → phys_block_id
_lru: OrderedDict[int, None]             # LRU 顺序（move_to_end 更新）
_ref_count: dict[int, int]               # phys_block_id → 引用计数
```

`OrderedDict` 天然支持 LRU：插入有序，`move_to_end` O(1)，从头迭代找最旧。

### 引用计数语义

```
ref_count[block] = (cache 持有) + (所有当前 running/prefilling 请求持有)
```

- 注册进 cache：`ref_count += 1`（cache 持有）
- 请求 admit 并复用：`ref_count += 1`（请求持有）
- 请求 free_request：`ref_count -= 1`（请求释放）
- eviction：只淘汰 `ref_count == 1`（只剩 cache 持有，无运行请求依赖）

### 请求状态机中的 prefix 字段

```python
# request.py
prefix_cached_len: int = 0           # 命中的前缀长度
prefix_cached_blocks: list[int] = [] # 命中的物理块 ID 列表
```

生命周期：
- admit（命中）：字段置非零，调用 `init_request_with_prefix`
- free_request（完成）：`free_request` 递减 ref_count
- swap_out（preemption）：`free_request` 递减 ref_count 后，**必须清零这两个字段**，防止 swap-in 后走错路径

### admission check 修正

```python
# 命中 prefix 后，blocks_needed 只算 suffix 和 decode
cached_len, _ = kv_cache.find_prefix_cache(prompt_token_ids)
suffix_len = prompt_len - cached_len
blocks_needed = ceil((suffix_len + max_out) / block_size)
```

不修正时，高命中率 workload 下 engine 错误地把已缓存的 prefix 也算进所需块数，拒绝完全可以服务的请求。

---

## 四、设计取舍

### hashmap vs. radix tree

| 维度 | hashmap（mini-infer）| radix tree（vLLM）|
|------|---------------------|------------------|
| 最长公共前缀查找 | 逐 block hash 匹配，遇到 miss 停止 | 沿树路径走，自动找最长 |
| 不同长度前缀共享 | 只能精确 hash 匹配 | 子树复用（A 的前 3 块 = B 的前 3 块）|
| Copy-on-Write | 无 | 有（多请求 decode 到不同位置时，shared prefix block 不拷贝）|
| 代码复杂度 | ~150 行 | ~1000 行 |
| 适用场景 | 固定 system prompt（hash 完全匹配）| 多样化前缀（few-shot 长度变化）|

对于"所有请求都用同一 system prompt"的生产场景，两者效果等价。

### 缓存一致性 vs. 缓存效率

当 prefix 命中时，suffix forward 用的 `past_key_values` 是从 block tensor 重建的 DynamicCache（不是原始 prefill 的 past_kv）。重建精度和原始 prefill 完全一致（按位相等），因为 block tensor 存的就是完整的 KV 浮点数据。

### 是否对 chunked prefill 请求启用 prefix cache？

Phase 10 的决策：**不启用**。原因：
1. chunk 状态机的 prompt_token_ids 在 chunk 过程中是局部的，hash 计算语义不清晰
2. chunk 场景通常是超长**新** prompt，不是重复 prefix 场景
3. 避免两个复杂状态机互相干扰

---

## 五、常见误区

### 误区 1：prefix cache 命中时可以跳过整个 prompt 的计算

不对。命中 N 个完整 blocks（N × block_size tokens）后，**suffix（prompt[N×block_size:]）仍需 prefill**。如果跳过 suffix，模型没有看到完整 prompt，第一个生成 token 会出错。

### 误区 2：命中 prefix 后不需要分配新块

不对。suffix 需要新块存放 suffix KV；decode 阶段也需要继续分配块。只是 prefix 对应的块不需要重新分配（直接复用缓存块的 ID）。

### 误区 3：ref_count 不用管也行，反正 eviction 有 check

不对。如果 eviction 的 check 是"块是否在某个请求的 block_table 里"，每次 eviction 都要遍历所有 running 请求的 block_table——O(num_requests × num_blocks)，低效且复杂。引用计数是 O(1) check。

### 误区 4：block-aligned prompt 可以全部缓存

不对。如果 prompt 恰好 block 对齐（len % block_size == 0），缓存所有 block 会导致 suffix 为空，模型 forward 收到空 input_ids，行为未定义（HF 会报错或给出无意义输出）。必须 capping：留最后一块不缓存。

---

## 六、和 mini-infer 的关系

Phase 10 在 `KVCacheManager` 里增加了 `_prefix_cache / _lru / _ref_count` 三个结构和对应的接口。`LLMEngine` 的 admit 路径分叉为 `_admit_with_prefix` 和原有的 `init_request`，prefill 路径分叉为 `model_runner.prefill_with_prefix`（suffix-only）和原有的 `prefill`（full prompt）。

Chunked prefill 路径（Phase 9）不修改，保持隔离。

---

## 七、进一步阅读

- **vLLM APC 设计**：vLLM blog "Automatic Prefix Caching"；`vllm/core/block_manager.py` AllocatorV2
- **SGLang RadixAttention**：论文 "Efficient LLM Serving with Radix Attention"（SGLang，2024）
- **KV cache 共享的正确性边界**：Flash attention 的 KV 是确定性的（同 prompt 同精度完全相同），prefix cache 复用是精确正确的，不是近似
- **Copy-on-Write 扩展**：多请求共享 prefix block 后各自 decode 时，需要 CoW 保证各自 decode 步的 KV 不互相污染——mini-infer 未实现，每个请求的 suffix block 是独立分配的，不存在 CoW 问题
