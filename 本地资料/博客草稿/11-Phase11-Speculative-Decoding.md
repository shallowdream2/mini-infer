# 从零实现投机解码：rejection sampling 正确性与 v1 双 forward 的代价

> 系列：mini-infer 推理系统学习项目 Phase 11

---

## 背景：大模型解码的瓶颈在哪

大语言模型自回归解码的根本瓶颈是内存带宽。每生成一个 token，需要加载一次模型全部权重（Qwen2.5-7B 约 14GB），而 GPU 的 FLOPS 远大于其能达到的 memory bandwidth 上限。换句话说，decode 是 memory-bound 的，算力是空闲的。

Speculative Decoding 的思路正是利用这部分空闲算力：用一个小模型（0.5B）快速预测多个候选 token，再让大模型（7B）一次 forward 同时验证 K 个位置——批量验证的代价与验证 1 个相差不多，但如果 K 个里有多个被接受，等效减少了大模型的 forward 次数。

本文记录在 mini-infer 中实现 speculative decoding 的完整过程：算法选择、KV cache 设计决策、一个隐蔽的正确性 bug，以及最终为什么 v1 实现反而比 target-only 更慢。

---

## 算法：Modified Rejection Sampling

核心算法来自 Leviathan et al. 2023，关键性质是**无偏**：输出序列的分布等价于只用 target model 生成。

设 draft model 在某位置的分布为 $q$，target model 的分布为 $p$。对 draft 采样的 token $x$：

- 以概率 $\min(1, p(x)/q(x))$ 接受
- 拒绝时，从修正分布 $\text{norm}(\max(0, p - q))$ 重采样一个新 token，然后停止这一轮

如果 K 个 draft token 全部被接受，额外从 target 在位置 $n+K$ 的分布里再采一个（bonus token），保证每轮至少产出 1 个 token。

这个算法有两个值得注意的地方：

**第一，拒绝时的修正分布是关键。** 不是简单地"拒绝就丢弃"，而是从 $p - q$ 的正部分重采样，这才使得最终输出分布恰好等于 $p$。

**第二，greedy 下 acceptance_rate 有自然的上界。** Temperature=0 时，draft 和 target 各自会选 argmax token。如果两个模型的 argmax 不同，当前 token 一定被拒绝。acceptance_rate 反映的是两个模型在多大比例的位置"意见一致"。

代码实现：

```python
def _rejection_sample(
    draft_tokens: list[int],
    draft_probs: list[torch.Tensor],   # K × [vocab]
    target_prev_logit: torch.Tensor,   # [vocab]，spec 开始前 target 的分布
    target_verify_logits: torch.Tensor, # [K, vocab]，K-token verify 输出
) -> tuple[list[int], torch.Tensor]:
    accepted: list[int] = []
    for i, token in enumerate(draft_tokens):
        p_target = target_probs_check[i][token].item()
        p_draft  = draft_probs[i][token].item()
        accept_prob = min(1.0, p_target / (p_draft + 1e-10))

        if torch.rand(1).item() < accept_prob:
            accepted.append(token)
        else:
            # 从修正分布重采样
            residual = (target_probs_check[i] - draft_probs[i]).clamp(min=0.0)
            s = residual.sum().item()
            new_token = _sample_from(residual / s) if s > 1e-10 else _sample_from(target_probs_check[i])
            accepted.append(new_token)
            return accepted, target_probs_check[i]  # 终止，不再看后续 draft token

    # 全部接受：额外采 bonus token
    bonus_token = _sample_from(_softmax(target_verify_logits[-1]))
    accepted.append(bonus_token)
    return accepted, target_verify_logits[-1]
```

---

## 实现架构：为什么需要两次 target forward

Speculative decoding 的实现难点不在算法，而在 **KV cache 的一致性管理**。

每轮 spec 迭代涉及三个动作，对应 target model 的两次 forward：

**① `spec_verify_target`（K-token 验证，不写 KV）**

从 block tensor 读取当前完整 KV，以 draft_tokens 为输入做 HF forward：

```python
current_kv = self.kv_cache.get_prefix_kv(block_table, seq_len)
out = self.model(input_ids=draft_tokens, past_key_values=current_kv, use_cache=False)
return out.logits[0].clone()  # [K, vocab]
```

`use_cache=False` 意味着这次 forward 不产生新 KV 输出，只取 logits。用于 rejection sampling 的判断。

**② Rejection sampling 决定接受哪些 token**

**③ `spec_advance_target_kv`（提交 KV）**

只把被接受的 token 写入 block tensor：

```python
current_kv = self.kv_cache.get_prefix_kv(block_table, seq_len)
out = self.model(input_ids=accepted_tokens, past_key_values=current_kv, use_cache=True)
self.kv_cache.write_prefill_kv_suffix(request_id, out.past_key_values, seq_len)
```

这是 v1 的设计取舍：**先验证再提交**，两步分离，逻辑清晰，但代价是每轮要跑两次 target forward。

---

## 一个隐蔽的 KV 位置 bug

在 review 阶段发现了一个正确性问题：`_get_prefill_last_logit`。

prefill 完成后，target model 的 KV block tensor 只有 prompt tokens 的 KV（seq_len = prompt_len）。第一个生成 token（first_token）的 KV 从未被写入。为了获取后续 spec 循环需要的"上一个位置的 target logit"，原实现调用了：

```python
# 原来的写法（有 bug）
logits = engine.model_runner.spec_verify_target(state, [first_token])
return logits[-1]
```

`spec_verify_target` 使用 `use_cache=False`，这意味着 first_token 的 KV 仍然没有被写入 block tensor。

后果：后续每轮 `spec_verify_target` 调用加载的 KV 都是"只有 prompt，没有 first_token"的，attention 相当于在错误的上下文里运行，偏移了整整 1 个位置。

修复方案是将这次调用替换为 `spec_advance_target_kv`：

```python
# 修复后：commit first_token 的 KV，同时获取 logit
target_prev_logit = self.target.model_runner.spec_advance_target_kv(
    t_state, t_state.generated_token_ids[:1]
)
```

这样 first_token 的 KV 被正确写入，seq_len 更新为 prompt_len + 1，后续所有 attention 都在正确上下文里运行。

这个 bug 在 dry_run 测试中不会被发现（stub 不做真实 attention），在 GPU 上的表现是输出略微不一致，acceptance_rate 偏低，但不会崩溃。只有仔细 review 代码流才能找到。

---

## 跨设备张量对齐

draft（0.5B）在 cuda:0，target（7B）在 cuda:1。rejection sampling 需要比较两个模型的概率分布，此时有两个对齐问题：

**设备对齐**：draft_probs 在 cuda:0，target_probs 在 cuda:1，不能直接做张量运算。

**vocab 对齐**：Qwen2.5-0.5B vocab_size = 151936，Qwen2.5-7B vocab_size = 152064，差了 128 个 token。

解决方案：在 rejection sampling 入口统一处理：

```python
def _align_draft_prob(p: torch.Tensor) -> torch.Tensor:
    p = p.to(device)  # 移到 target 设备
    if p.shape[0] < target_vocab:
        # padding 到 target vocab 大小
        return torch.cat([p, torch.zeros(target_vocab - p.shape[0], device=device, dtype=p.dtype)])
    return p[:target_vocab]
```

---

## 实验结果

**环境**：RTX 4090 × 2，Qwen2.5-0.5B @ cuda:0，Qwen2.5-7B @ cuda:1

**Workload**：4 条英文 prompt，max_new_tokens=64，temperature=0.0（greedy），K=4

| 指标 | SpecEngine (K=4) | Target-only |
|------|-----------------|-------------|
| total_time | 5.69s | 4.51s |
| throughput | ~29.0 tok/s | ~44.4 tok/s |
| **speedup** | **0.65×** | baseline |
| **acceptance_rate** | **55.85%** | N/A |
| memory (draft) | 1.9 GB (cuda:0) | — |
| memory (target) | 18.2 GB (cuda:1) | 18.2 GB (cuda:1) |

*throughput 为 word count 近似，非精确 token count。*

acceptance_rate = 55.85%（229/410）：K=4 greedy 下 0.5B 和 7B 在超过一半的位置"意见一致"，rejection sampling 工作正常。

**但 v1 spec 比 target-only 慢了 35%。**

---

## 为什么更慢：v1 双 forward 的算力分析

每轮 spec 迭代，当 acceptance_rate=55.85% 时平均接受约 2.3 个 token，但需要：
- K=4 次 draft forward（0.5B，每次 ~0.14 次 7B 等效）
- 1 次 target verify（7B）
- 1 次 target advance（7B）

等效 target forward/token：

```
(2 × 1 + 4 × 0.5/7) / 2.3 ≈ (2 + 0.29) / 2.3 ≈ 1.0
```

理论上与 target-only（1 forward/token）计算量相当，但实际 0.65× 的差距来自系统层：

- 每轮 2 次 target forward 的 CUDA kernel launch + sync overhead
- `get_prefix_kv` 每轮从 block tensor 重建完整 KV（额外内存读写）
- 跨设备张量拷贝（cuda:0 → cuda:1 的 draft_probs align）
- Python-level K=4 循环（sequential，无法并行）

这些开销加在一起超过了 spec 减少的 target forward 收益。

---

## v2：怎样才能真正更快

v2 的核心改进是把 verify 和 advance 合并为**一次 target forward**：

```
输入：[prompt_context + draft_tokens]
use_cache=True
输出：logits（用于 rejection sampling 判断）+ KV（直接写入 block tensor）
```

这样每轮只跑一次 7B forward。被拒绝的 token 对应的 KV 写入后需要通过 rollback 清除，多一步但总代价更低。

预期 speedup（acceptance_rate=55.85%，K=4）：

```
(K × AR + 1) / 1   ÷   (2 × 7B_equiv) / (K × AR + 1)
≈ (4 × 0.56 + 1) / (1 forward per 3.24 tokens)
≈ 3.24× fewer 7B forwards relative to 1× AR
```

保守估计 v2 speedup ≈ 1.3-1.6×。

---

## 坑点总结

1. **KV 位置偏移：prefill 后 first_token 的 KV 必须写入**。不能用 `use_cache=False` 的 forward 来"获取 logit 同时不改 KV"，因为后续所有 attention 都依赖这个 KV 是否存在。

2. **跨设备 tensor 对齐要在 rejection sampling 入口统一处理**。在每个 for 循环里单独 `.to()` 容易漏，集中到 `_align_draft_prob` 包装函数里更安全。

3. **HF snapshot 路径 ≠ 模型根目录**。Qwen2.5-7B 的 HF cache 目录存在 snapshot 子目录软链接不完整的问题（shard 2-4 缺失），必须用根目录路径 + `HF_HUB_OFFLINE=1`，否则 transformers 会尝试重新下载。

4. **v1 慢是可预期的，但要量化才能确认**。benchmark 前没有理由假设 v1 能带来加速，benchmark 后才能知道到底差多少、差在哪。

---

## 总结

Phase 11 实现了 speculative decoding 的完整功能链路：draft 生成 → target 验证 → rejection sampling → KV 同步，acceptance_rate = 55.85%（K=4, greedy），输出分布等价于 target-only。

v1 的主要局限是吞吐 0.65× target-only，根本原因是双 forward 设计在系统层的 overhead 超过了算法层的收益。这不是 rejection sampling 的问题，而是 KV cache 管理方式的代价——v2 通过合并 forward 可以解决，但 v1 证明了算法正确性是第一步。

推理系统的工程规律：先跑通，再量化，再优化。每一步都需要真实数据支撑。
