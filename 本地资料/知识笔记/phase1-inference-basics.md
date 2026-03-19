# Phase 1 推理基础知识笔记

这个文件记录 Phase 1 过程中涉及的核心概念，是自己的理解，不是复制文档。

---

## Prefill 和 Decode 的区别

**Prefill**：把整个 prompt（可能几百个 token）一次性送入模型，得到所有位置的 logits 和 past_key_values。计算量 O(n²)，但只做一次。

**Decode**：每次只送入上一步生成的 1 个 token，利用 past_key_values 缓存避免重复计算，得到下一个 token 的 logits。计算量 O(n)，但要重复 max_new_tokens 次。

关键点：decode 是 memory-bound 的，每步 GPU 的计算量极小，主要时间花在加载权重（KV cache + model weights）上。这就是为什么 batch decode 很重要——把多个请求的 decode 合并成一次 forward，权重只加载一次，摊销了显存带宽的代价。

---

## past_key_values 是什么

HuggingFace 模型 forward 时，`use_cache=True` 会在输出里附带 `past_key_values`：一个 tuple，每层一个 `(key, value)` 对，shape 是 `[batch, num_heads, seq_len, head_dim]`。

下一次 decode 时把这个 tuple 传回去，模型就不需要重新计算之前所有 token 的 K/V，直接拼接新 token 的 K/V 进行 attention。

Phase 1 把每个请求的 `past_key_values` 存在一个 dict 里（key 是 request_id）。问题：
1. 不同请求的 seq_len 不同，无法直接 stack 成 batch
2. 显存随 seq_len 增长，没有上限控制

这就是 Paged KV Cache 要解决的问题。

---

## 为什么串行 decode 吞吐不随 batch 增长

Phase 1 的 `decode_step`：
```python
for state in states:
    out = model(input_ids=[[last_token]], past_key_values=state_kv)
```

batch=4 → 4 次独立 GPU forward，每次 GPU 利用率极低（7B 模型的 compute 需求远小于显存带宽需求）。

HF baseline 的 `generate`：
```python
model(input_ids=[[tok1], [tok2], [tok3], [tok4]], past_key_values=batched_kv)
```

4 条请求合并成一次 forward，权重读一次，throughput 近线性扩展。

**结论**：decode 阶段的性能瓶颈不是计算，而是显存带宽。batch 越大，带宽摊销越好，吞吐越高。

---

## Greedy vs Sampling

Greedy（temperature=0）：直接取 logits argmax，deterministic，每次输出相同。

Temperature sampling（temperature>0）：先把 logits 除以 temperature，再 softmax，然后按概率采样。temperature 越高，分布越平，输出越随机。

Top-p（nucleus sampling）：在按概率排序后的 token 里，只保留累积概率 ≤ top_p 的那些，然后在这个子集里采样。避免采到极低概率的 token。

Phase 1 实现了这三种，测 benchmark 时用 greedy（do_sample=False）保证可复现。

---

## Qwen2.5 的 EOS token

Qwen2.5 的 EOS token 不是普通的 `</s>`，而是 `<|im_end|>`，token_id 是 151645。

用 chat template 时，每轮对话结束后模型会生成这个 token。如果 engine 不正确处理 EOS，要么提前停止（误判），要么一直跑到 max_new_tokens（不停）。

判断是否到 EOS：
```python
if next_token_id == self.eos_token_id:
    state.mark_finished("eos")
```

坑：`self.eos_token_id = self.tokenizer.eos_token_id or 0` 在 eos_token_id=0 的模型上有歧义，正确写法是 `if ... is not None else -1`。

---

## GQA（Grouped Query Attention）

Qwen2.5-7B 用 GQA：28 个 Q heads，但只有 8 个 KV heads（每 3.5 个 Q head 共享 1 个 KV head）。

这意味着 past_key_values 的 shape 是 `[batch, 8, seq_len, head_dim]`，而不是 `[batch, 28, seq_len, head_dim]`。

影响：Phase 2 的 KV block tensor 分配时，num_kv_heads=8，不是 num_heads=28。搞错了会 OOM 或维度不匹配。
