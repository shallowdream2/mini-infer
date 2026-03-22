# Phase 10 开发日志 — Prefix Caching

**日期**：2026-03-22
**阶段**：Phase 10 — Prefix Caching（RadixAttention 风格 block-level prefix cache + LRU eviction）

---

## 开发过程

### 阶段起点

Phase 9 完成了 Chunked Prefill，调度器结构更规整（prefilling 队列 + chunk 状态机），Phase 10 在此基础上添加 prefix 复用能力。计划文档已在上一次会话确认，主要改动在 `kv_cache.py`、`engine.py`、`model_runner.py` 和 `request.py`。

---

### 第一轮实现：核心数据结构（kv_cache.py）

新增三个字典：
- `_prefix_cache: dict[int, int]`：block_hash → phys_block_id
- `_lru: OrderedDict`：LRU 顺序维护
- `_ref_count: dict[int, int]`：引用计数（cache + 运行请求各持有一份）

**第一个坑：Python hash() 非确定性**

实现 `_compute_block_hashes` 时最初用了 `hash(tuple(token_ids))`。本地测试通过，但意识到 PYTHONHASHSEED 会导致进程间 hash 不同、进程重启后缓存失效。立即换成 `hashlib.sha256 + struct.pack`，取 digest 前 8 字节作为 64-bit int。

链式设计：每个 block 的 hash 包含上一个 block 的 hash（`prev_hash` 作为前缀），避免同一 token 序列在不同位置错误命中。

**capping 逻辑**：prompt 恰好 block 对齐时，末尾一块不缓存，保证 suffix 非空：
```python
max_cacheable = (num_full_blocks - 1) if (len(token_ids) % block_size == 0) else num_full_blocks
```

实现了 `find_prefix_cache`、`init_request_with_prefix`、`register_prefix_blocks`、`evict_lru_prefix_block` 等接口。

---

### 第二轮：model_runner.py — prefill_with_prefix

hit 路径需要只对 suffix 做 HF forward，把 prefix KV 作为 `past_key_values` 传入。

`get_prefix_kv` 从 block tensor 重建 `DynamicCache`：按 block 遍历，把物理块数据拼装成 `[batch=1, heads, seq, head_dim]` 的 tensor，用 HF DynamicCache 的 `update()` 接口填充。

然后 `write_prefill_kv_suffix` 只把 suffix 的新 KV（`out.past_key_values` 中 `cached_len:` 之后的部分）写入 block tensor。

---

### 第三轮：engine.py + request.py

`request.py` 在 `RequestState` 加两个字段：`prefix_cached_len: int = 0`，`prefix_cached_blocks: list[int]`。

`engine.py` 加两个辅助方法：
- `_admit_with_prefix`：admit 时查 prefix cache，命中则走 `init_request_with_prefix`
- `_prefill_and_register`：分 miss/hit 两路 prefill，prefill 完成后统一注册

关键修正：**admission check 要用 suffix_len 而非 prompt_len**。否则高命中率时，引擎认为需要更多块（把已缓存的 prefix 也算进去），拒绝完全可以服务的请求。

---

### 第四轮：测试

写了 `tests/test_prefix_cache.py`，15 个测试。

**第二个坑：6 个测试 cached_len == 0**

测试 prompt 用了 4-token 字符串 + block_size=4。4 % 4 == 0，触发 capping，max_cacheable=0。这是正确行为，但测试设计错了。改用 5-token prompt（5 % 4 != 0，partial tail 自然成 suffix）。

修改了 `tests/test_engine.py` 和 `tests/test_preemption.py` 中 4+1 处断言：
```python
# 旧
assert engine.kv_cache.num_free_blocks() == initial_free
# 新（prefix cache 合法持有 block）
assert engine.kv_cache.num_free_blocks() + engine.kv_cache.prefix_cache_size() == initial_free
```

---

### 第五轮：swap_out 修复

infer-review 发现：swap_out 后 `state.prefix_cached_len` 和 `state.prefix_cached_blocks` 保留旧值。swap-in 后 re-admit 时会走 `init_request_with_prefix`，但引用计数已经被 `free_request` 递减过了——逻辑混乱，可能导致负计数或提前释放。

修复：在 `swap_out` 里 `free_request` 之后清零这两个字段：
```python
state.prefix_cached_len = 0
state.prefix_cached_blocks = []
```

---

### GPU Benchmark

**第三个坑：prefix_cache_size=0**

GPU benchmark 的 shared_prefix 只有 39 token（一段硬编码英文），block_size=256。39 token 没有完整 block，什么都没注册进 cache。

修复：用引擎加载的 tokenizer 精确 encode 后截断到 `block_size+1=257` tokens，保证 1 个完整 cacheable block。

**第四个坑：OOM**

首次 benchmark `num_gpu_blocks=1024`，block tensor pool 太大，Qwen2.5-7B 显存不足。降到 256 正常。

**最终结果**：miss TTFT=1468ms，hit TTFT=1139ms，**speedup=1.29×（−22%）**。
batch=4 吞吐 speedup=0.99×（decode 占主导，1 block prefix 节省比例低）。

---

## 关键决策

1. **用 SHA-256 而不是 MurmurHash/CRC32**：标准库可用，碰撞率足够低，无需额外依赖。
2. **不实现 radix tree**：hashmap + OrderedDict LRU 对单机固定 system prompt 场景足够，代码量少 5×。
3. **chunked prefill 路径不启用 prefix cache**：Phase 9 的 chunk 状态机与 prefix 的 suffix 对齐逻辑冲突，且 chunk 场景通常是超长新 prompt，不是重复 prefix 场景，保持隔离是正确决策。

---

## 测试最终状态

```
conda run -n ai-infra python -m pytest tests/ -v
# 100 passed
```
