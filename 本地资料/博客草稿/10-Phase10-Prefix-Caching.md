# LLM 推理系统实战（十）：从零实现前缀缓存（Prefix Caching）

> 本文是 mini-infer 系列的第十篇，基于 Phase 10 的真实实现和 RTX 4090 实测数据。
> 实验环境：Ubuntu 24.04 + RTX 4090，Qwen2.5-7B-Instruct，PyTorch 2.1.2+cu121，flash_attn 2.5.9.post1。

---

## 问题背景：每次都从头 prefill，真的有必要吗？

考虑一个典型的 Agent 服务场景：

- 一个 RAG 问答服务，每个请求都带着 512 token 的系统提示（检索出的文档）
- 一天内同一文档被查询了 10000 次
- 每次请求都从头跑一遍 512 token 的 prefill

这 10000 次中，有 9999 次 prefill 是重复计算——同样的 token 序列，同样的 KV 矩阵，结果完全一样。

**Prefix Caching** 解决的就是这个问题：把 prefill 计算出来的 KV 缓存起来，下次遇到相同前缀时直接复用，跳过这部分计算。vLLM 称之为 Automatic Prefix Caching（APC），SGLang 的论文里叫 RadixAttention。

mini-infer 在 Phase 10 实现了一个基于 hashmap 的 block 粒度前缀缓存，本文记录实现思路、设计决策和踩过的坑。

---

## 为什么是"block 粒度"而不是 token 粒度？

KV cache 的物理组织已经是 block 粒度了（Phase 2 以来的 Paged KV Cache）。每个物理 block 存 `block_size` 个 token 的 KV，以 block 为单位分配和回收。

如果以 token 为粒度缓存，需要在 prefix cache 中存储每个 token 的 KV，访问时需要逐 token 查找并拼接——开销大且破坏了 block 的物理对齐假设。

**以 block 为粒度**则自然地与现有物理块对齐：

- 命中时，直接把缓存的物理块 ID 放进新请求的 block_table
- 无需任何 KV 数据移动
- `blocks_needed` 的计算也自然减少

代价是：**缓存长度必须是 block_size 的整数倍**。一个 260-token 的 prompt，`block_size=256` 时只能缓存前 256 token（1 block），最后 4 token 不缓存（作为 suffix 需要重新 prefill）。

---

## 核心数据结构

```python
# kv_cache.py
self._prefix_cache: dict[int, int] = {}          # block_hash → phys_block_id
self._lru: OrderedDict[int, None] = OrderedDict() # 维护 LRU 顺序
self._ref_count: dict[int, int] = {}             # phys_block_id → 引用计数
```

三个字典配合工作：

- `_prefix_cache`：hash 到物理块的映射，O(1) 查找
- `_lru`：插入顺序即 LRU 顺序（`move_to_end` 更新），内存压力时从头淘汰
- `_ref_count`：**引用计数**，这是保证正确性的关键

### 为什么需要引用计数？

考虑这个场景：两个请求 A 和 B 共享同一个缓存 block（物理块 #5）。此时内存紧张，LRU eviction 想淘汰块 #5。如果直接释放，请求 A 的 block_table 里还有 #5——使用已释放的块，等同于 use-after-free。

引用计数的语义：

```
ref_count[block] = (cache 持有) + (每个正在运行的请求持有)
```

- cache 注册时：`ref_count += 1`（cache 持有）
- 请求 admit 并复用时：`ref_count += 1`（请求持有）
- 请求完成 free_request 时：`ref_count -= 1`（请求释放）
- LRU evict 时：**只淘汰 ref_count == 1 的块**（只剩 cache 持有，没有请求在用）

---

## Block Hash 的设计：为什么不用 Python 的 hash()？

最直觉的做法：`hash(tuple(token_ids[start:end]))`。问题：Python 的 `hash()` 受 `PYTHONHASHSEED` 影响，每次进程启动 hash 值不同。这意味着：

- 进程重启后所有缓存失效（hash 对不上）
- 单元测试里 hash 值随机变化，测试不稳定

正确做法是用确定性 hash。minimax 的选择是 SHA-256 截断：

```python
def _compute_block_hashes(self, token_ids: list[int]) -> list[int]:
    hashes = []
    prev_hash = 0
    for i in range(max_cacheable):
        start, end = i * self.block_size, (i + 1) * self.block_size
        buf = struct.pack(
            f">Q{end - start}i",
            prev_hash & 0xFFFFFFFFFFFFFFFF,  # 8 字节前驱 hash
            *token_ids[start:end],             # 本块 token ids
        )
        block_hash = int.from_bytes(hashlib.sha256(buf).digest()[:8], "big")
        hashes.append(block_hash)
        prev_hash = block_hash  # 链式传递
    return hashes
```

**链式设计的价值**：`hash[i]` 包含了 `hash[0]...hash[i-1]` 的历史。两个在不同位置出现的相同 token 序列（比如两段不同文档里都出现了 "Hello, I am a helpful assistant"），它们的 block hash 不同——避免了跨上下文的错误命中。

这和 radix tree 的前缀内嵌原理是一样的：树的每条从根到节点的路径，天然嵌入了前缀信息。

---

## 关键边界条件：capping 防止空 suffix

实现时最容易踩的坑之一：**prompt 完全 block 对齐时，不能缓存所有 block**。

假设 `block_size=4`，prompt 长度恰好是 8（2 个完整 block）：

```
token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
            |-- block 0 --|-- block 1 --|
```

如果把 2 个 block 都缓存，`cached_len = 8 = prompt_len`，那么 `suffix = prompt[8:] = []`——空输入无法 forward。

因此 `_compute_block_hashes` 的 capping 逻辑：

```python
num_full_blocks = len(token_ids) // self.block_size
# 若 prompt 恰好 block 对齐，留最后一块不缓存（保证至少 1 block suffix）
max_cacheable = (num_full_blocks - 1) if (len(token_ids) % self.block_size == 0) else num_full_blocks
```

非 block 对齐时（如 9 token），partial tail（1 token）自然成为 suffix，无需特殊处理。

---

## 准入检查的修正：别让高命中率成为诅咒

这是一个不那么明显但影响正确性的 bug。原始准入检查（来自 Phase 2）：

```python
blocks_needed = math.ceil((prompt_len + max_out) / block_size)
if self.kv_cache.num_free_blocks() < blocks_needed:
    # 拒绝准入...
```

加入前缀缓存后，如果命中了前 256 token，实际只需要为 suffix 和 decode 分配新块：

```python
# Phase 10 修正：先 peek prefix cache，调整 blocks_needed
cached_len_peek, _ = self.kv_cache.find_prefix_cache(next_state.prompt_token_ids)
suffix_len = prompt_len - cached_len_peek
blocks_needed = math.ceil((suffix_len + max_out) / block_size)
```

不修正的后果：高命中率 workload 下，每个请求都以为自己需要 `ceil((512+64)/256) = 3` 个块，但实际只需要 1 个（256 token 的 suffix + 64 decode = 320 token = 2 块，减去已缓存的 1 块）。引擎会错误地拒绝完全可以服务的请求。

---

## 请求生命周期中的 prefix 状态管理

prefix cache 的引用计数需要在请求的完整生命周期中正确维护：

### admit 阶段
```python
def _admit_with_prefix(self, state):
    cached_len, cached_blocks = self.kv_cache.find_prefix_cache(state.prompt_token_ids)
    if cached_len > 0:
        self.kv_cache.init_request_with_prefix(state, cached_len, cached_blocks)
        state.prefix_cached_len = cached_len
        state.prefix_cached_blocks = list(cached_blocks)
    else:
        self.kv_cache.init_request(state)
```

`init_request_with_prefix` 内部对每个 cached block 做 `ref_count += 1`，然后只为 suffix 分配新块。

### prefill 阶段
```python
def _prefill_and_register(self, newly_admitted):
    miss_states = [s for s in newly_admitted if s.prefix_cached_len == 0]
    hit_states  = [s for s in newly_admitted if s.prefix_cached_len > 0]

    if miss_states:
        self.model_runner.prefill(miss_states)         # 整个 prompt
    for state in hit_states:
        self.model_runner.prefill_with_prefix(         # 只做 suffix
            state, state.prefix_cached_len, state.prefix_cached_blocks)

    for state in newly_admitted:
        self.kv_cache.register_prefix_blocks_for_request(  # miss 路径新计算的块注册进 cache
            state.request.request_id, state.prompt_token_ids)
```

miss 路径 prefill 完成后，新计算的 KV blocks 注册进 cache。下一个有相同前缀的请求就能命中了。

### free/preemption 阶段

请求完成后 `free_request` 自动递减引用计数。这里有个容易遗漏的角落：**swap_out 时**。

如果一个有 prefix 命中的请求被换出（preemption），swap_out 会调用 `free_request` 释放 GPU 块、递减引用计数。此时 `state.prefix_cached_len` 和 `state.prefix_cached_blocks` 还保留着旧值。swap_in 后，引擎重新 admit 该请求，如果不清除这两个字段，会错误地调用 `init_request_with_prefix`——但引用计数已经减过了，再减就可能出现负数或提前释放。

修复：

```python
def swap_out(self, state):
    # ... 现有的 swap 逻辑 ...
    self.free_request(state)
    state.prefix_cached_len = 0
    state.prefix_cached_blocks = []
```

swap_in 后请求回到 waiting 队列，下次 admit 时会重新 `find_prefix_cache`——如果 cache 还在，再次命中；如果已被 evict，走 miss 路径。语义清晰，无 stale state。

---

## hit 路径的 prefill：suffix-only forward

命中 prefix 后，只需要对 suffix 部分做模型 forward：

```python
def prefill_with_prefix(self, state, cached_len, cached_blocks):
    # 1. 从 block tensor 重建 prefix KV（DynamicCache 格式，供 HF 模型消费）
    prefix_cache_obj = self.kv_cache.get_prefix_kv(cached_blocks, cached_len)

    # 2. 只对 suffix tokens 做 forward
    suffix_ids = state.prompt_token_ids[cached_len:]
    input_ids = torch.tensor([suffix_ids], device=self.config.device)

    with torch.no_grad():
        out = self.model(
            input_ids=input_ids,
            past_key_values=prefix_cache_obj,  # prefix KV 作为 past_key_values
            use_cache=True,
        )

    # 3. 把 suffix 的 KV 写入 block tensor（新分配的块）
    self.kv_cache.write_prefill_kv_suffix(state.request.request_id, out.past_key_values, cached_len)

    # 4. 采样第一个 token
    next_token = _sample_token(out.logits[0, -1], state.request.sampling_params)
    state.append_generated(next_token, "")
    state.prefilled = True
```

`get_prefix_kv` 从 block tensor 重建出 HF 格式的 `DynamicCache`，然后传给模型的 `past_key_values`。模型 forward 时就像已经处理了前缀一样，只对 suffix 计算 attention。

---

## 实测结果

**环境**：RTX 4090，Qwen2.5-7B-Instruct，block_size=256，num_gpu_blocks=256

**实验 1：单请求 TTFT**

| 路径 | TTFT | cache_size |
|------|------|-----------|
| miss（建立 cache） | 1468.0 ms | 1 block |
| hit（复用 prefix） | 1139.3 ms | 1 block |
| **speedup** | **1.29× (−22%)** | — |

shared prefix = 257 tokens（恰好 1 个完整 cacheable block）。

**实验 2：batch=4 吞吐**

| 路径 | 耗时 | 近似吞吐 |
|------|------|---------|
| miss batch | 1235.8 ms | ~207 tok/s |
| hit batch | 1242.9 ms | ~206 tok/s |
| speedup | 0.99× | — |

batch 吞吐几乎无提升。原因很直接：`max_new_tokens=64` 意味着 64 步 decode，而 prefix 只有 1 block（256 token）。节省的 256 token prefill 时间相比 64 步 decode 的总时间占比很低。

**这不是 bug**，而是当前测试 workload 不是 prefix caching 的适用场景。真正有收益的场景：

| 场景 | prefix | output | 预期收益 |
|------|--------|--------|---------|
| 当前测试 | 257 token（1 block）| 64 token | 低（1.29×）|
| RAG 问答 | 1024 token（4 blocks）| 32 token | 高（理论 ~5×）|
| Few-shot 评估 | 2048 token（8 blocks）| 16 token | 很高（理论 ~10×）|

---

## 踩过的坑

### 坑 1：Python hash() 随机失效

实现初版用 `hash(tuple(token_ids))`，在同一进程内是稳定的，但单元测试里偶发失败——测试跑 100 遍有几次 hash 撞了不该撞的位置。根因是 PYTHONHASHSEED：虽然 CPython 3.6+ 在同一进程内 hash 稳定，但不同进程（pytest 并发）或重启后不同。换成 SHA-256 后完全消除。

### 坑 2：capping 导致 6 个测试失败

初始测试设计用 4-token prompt + block_size=4。4 % 4 == 0，触发 capping：max_cacheable = 1 - 1 = 0，`find_prefix_cache` 返回 `(0, [])`。测试断言 `cached_len > 0` 失败。

这不是实现 bug，是测试设计错误——block-aligned prompt 的 capping 是正确行为。修复方式：改用 5-token prompt（5 % 4 != 0，partial tail 自然成为 suffix）。

### 坑 3：GPU 上 prefix_cache_size == 0

第一次 GPU benchmark 输出 `prefix_cache_size=0 after miss`。原因：benchmark 脚本的 shared_prefix 只有 39 tokens（一段硬编码的字符串），远低于 block_size=256。39 token 里没有一个完整的 cacheable block，自然没有任何东西注册进 cache。

修复：用模型的 tokenizer 精确截断到 `block_size + 1`（257）tokens：

```python
tids = tokenizer.encode(long_string, add_special_tokens=True)[:257]
shared_prefix = tokenizer.decode(tids)
```

这样保证了至少 1 个完整可缓存 block，同时保留 1 token suffix。

### 坑 4：GPU OOM

第一次 benchmark 设 `num_gpu_blocks=1024`。Qwen2.5-7B 加载后 block tensor pool 也要 `1024 × 256 × num_layers × num_kv_heads × head_dim × sizeof(fp16)` 字节，显存不足 OOM。降到 256 后正常。生产环境需要根据 `nvidia-smi` 的可用显存动态计算合理的 `num_gpu_blocks`。

---

## 与 vLLM 的实现对比

vLLM 的 Automatic Prefix Caching 在 block_manager v2 中使用了更复杂的 radix tree，支持：
- 跨请求的任意前缀共享（不要求完全相同，只要前缀匹配即可）
- copy-on-write（多请求 decode 到不同位置时，共享 prefix block 不拷贝）
- 跨多机的分布式 prefix cache

mini-infer 的实现是 hashmap 版本，核心差异：

| 特性 | mini-infer (Phase 10) | vLLM APC |
|------|----------------------|---------|
| 数据结构 | hashmap + OrderedDict LRU | radix tree + eviction |
| 共享粒度 | block 级（整块对齐）| 同上 |
| 跨请求共享 | 相同 hash 可以共享 | 自动识别最长公共前缀 |
| Copy-on-Write | 未实现 | 有（多请求 decode 分叉时） |
| 分布式 | 无 | 有（实验性）|
| 代码量 | ~150 行 | ~1000 行 |

对于单机推理，hashmap 版本在常见场景（固定 system prompt）下与 radix tree 等价——因为 system prompt 是公共前缀，hash 链自然会命中。radix tree 的优势在于可以高效处理不同长度的共享前缀（比如 few-shot 中用 3 条 example 还是 5 条，前 3 条的 KV 可以共享）。

---

## 总结

Phase 10 在现有的 Paged KV Cache 基础上，以约 150 行核心代码实现了 block 粒度的前缀缓存：

- **确定性 hash**：SHA-256 链式 hash，跨进程、跨重启稳定
- **ref_count 安全**：cache + 运行请求双持有，eviction 只淘汰无请求的 block
- **capping 边界**：block-aligned prompt 保留最后一块为 suffix，避免空 input_ids
- **准入修正**：命中 prefix 后 blocks_needed 仅算 suffix，防止过度拒绝
- **swap_out 清零**：preemption 后 prefix state 清空，re-admit 时重新命中

实测在 1-block（257 token）prefix 下单请求 TTFT **−22%**。batch 吞吐收益取决于 prefix 长度与 output 长度之比；RAG 类 workload（长 prefix + 短 output）将有更显著的提升。

系列下一篇：**Speculative Decoding**——用小模型（Qwen2.5-0.5B）批量 draft，大模型（Qwen2.5-7B）并行验证，目标在高接受率 workload 下将 decode 吞吐提升 2× 以上。

---

## 自查结果（QUALITY_CHECKLIST.md）

- [x] 文章主题对应真实项目阶段或真实技术问题
  — Phase 10 Prefix Caching，有完整的实现代码和 GPU 实测数据

- [x] 结构覆盖：背景、问题定义、方案设计、实现细节、实验结果、坑点反思、总结
  — 覆盖：背景（重复 prefill 问题）→ 方案（block 粒度 + hash）→ 实现（hash 设计、ref_count、capping、准入修正、swap_out 修复、hit 路径 prefill）→ 实验（1.29× TTFT）→ 坑点（4 个真实问题）→ 总结 + vLLM 对比

- [x] 有明确背景和问题定义
  — "每次都从头 prefill，10000 次查询有 9999 次是重复计算"

- [x] 有设计取舍，而不是只罗列概念
  — block 粒度 vs. token 粒度的取舍；Python hash() vs. SHA-256 的选择；capping 的必要性；准入 check 的修正逻辑

- [x] 有代码、实验或运行结果支撑
  — 关键函数代码片段均来自真实实现；实验数据来自 `本地资料/实验记录/2026-03-22-phase10-benchmark.md`（miss=1468ms，hit=1139ms，speedup=1.29×）

- [x] 有失败点、坑点或反思
  — 4 个真实坑点：hash 随机性、capping 测试失败、GPU prefix_cache_size=0、OOM；batch 吞吐无明显提升的诚实分析

- [x] 语言克制、专业、可发表
  — 无夸大结论；batch 吞吐 0.99× 如实报告并解释原因；与 vLLM 的差距明确说明
