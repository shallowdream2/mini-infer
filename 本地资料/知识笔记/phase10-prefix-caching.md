# Phase 10 知识笔记 — Prefix Caching

## 核心问题

Agent/RAG 场景中，大量请求共享相同的 system prompt 或文档前缀。每次从头 prefill 是重复计算，KV 完全相同。

---

## Block 粒度缓存

以 block 为粒度（而非 token）：

- 与 Paged KV Cache 的物理 block 天然对齐
- 命中时直接把物理块 ID 插入 block_table，零拷贝
- 代价：缓存长度必须是 block_size 的整数倍

**capping**：prompt 完全 block 对齐（len % block_size == 0）时，末尾一块不缓存，保证 suffix 非空（不能给模型传空 input_ids）。

---

## Block Hash 设计

### 为什么要链式 hash？

```
hash[0] = sha256(0 ‖ tokens[0:bs])
hash[1] = sha256(hash[0] ‖ tokens[bs:2bs])
hash[2] = sha256(hash[1] ‖ tokens[2bs:3bs])
```

如果不链式，位置 `i*bs:(i+1)*bs` 相同的 token 序列（不管前面是什么）都会得到相同 hash。链式设计把"历史上下文"嵌入进每个 block 的 hash，不同位置的相同 token 序列 hash 不同。

### 为什么用 SHA-256 不用 hash()?

Python `hash()` 受 `PYTHONHASHSEED` 影响，进程重启后结果不同。多进程测试也不稳定。SHA-256 是确定性的。

---

## 引用计数

```
ref_count[block] = cache持有(1) + sum(running请求持有)
```

- cache 注册时 +1
- 请求复用时 +1
- 请求 free 时 -1
- LRU evict 只淘汰 ref_count == 1 的（只剩 cache 持有）

**不做引用计数的后果**：两个请求共享 block，eviction 释放了 block，第一个请求还在用——use-after-free。

---

## 请求生命周期中的注意点

### admission check 要用 suffix_len

```python
cached_len, _ = find_prefix_cache(prompt_token_ids)
suffix_len = prompt_len - cached_len
blocks_needed = ceil((suffix_len + max_out) / block_size)  # 不是 prompt_len
```

否则高命中率时过度拒绝准入。

### swap_out 要清零 prefix state

```python
state.prefix_cached_len = 0
state.prefix_cached_blocks = []
```

否则 swap-in 后 re-admit 时走错路径（引用计数已被 free_request 递减，再走 with_prefix 会混乱）。

---

## 与 vLLM 的差距

| 特性 | mini-infer | vLLM |
|------|-----------|------|
| 数据结构 | hashmap + OrderedDict | radix tree |
| 任意前缀共享 | 只支持完全相同 hash | 自动最长公共前缀 |
| Copy-on-Write | 无 | 有（多请求 decode 分叉） |
| 代码量 | ~150 行 | ~1000 行 |

hashmap 版本在"固定 system prompt"场景等价于 radix tree；多样化前缀场景（不同长度的 few-shot）radix tree 优势才显现。

---

## 实测（RTX 4090 + Qwen2.5-7B，block_size=256）

- 单请求 TTFT：miss 1468ms → hit 1139ms，**speedup 1.29×（−22%）**
- batch=4 吞吐 speedup=0.99×（decode 占主导，1-block prefix 节省比例低）
- 真实收益场景：长 prefix（1000+）+ 短 output（RAG）
