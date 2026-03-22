# 知识专题：Speculative Decoding

## 主题

投机解码（Speculative Decoding）：使用小型 draft model 加速大型 target model 的自回归推理，同时保证输出分布等价于 target-only 生成。

---

## 一、问题定义

大模型自回归 decode 是 **memory-bound**，而非 compute-bound：
- 每生成一个 token，需加载全部模型权重一次（Qwen2.5-7B ≈ 14GB）
- GPU 的 compute throughput 远超 memory bandwidth 上限
- 绝大多数 CUDA kernel 在等内存，而非在算

结果：即使有空闲算力，decode 吞吐也受限于内存带宽，无法通过增大 batch 来提升单请求速度（单请求 batch=1 情况下）。

**投机解码的问题**：能否利用空闲算力，在不改变输出分布的前提下，减少 target model forward 次数？

---

## 二、核心原理

### Modified Rejection Sampling（Leviathan et al. 2023）

1. draft model 生成 K 个候选 token $x_1, ..., x_K$，记录各位置的采样分布 $q_i$
2. target model 一次 forward 验证 K 个位置，得到各位置的分布 $p_i$
3. 对每个 draft token $x_i$，按概率 $\min(1, p_i(x_i)/q_i(x_i))$ 接受，否则拒绝
4. 拒绝时，从修正分布 $\text{norm}(\max(0, p_i - q_i))$ 重采样一个 token，结束这一轮
5. 全部接受时，额外从 $p_{K+1}$ 采一个 bonus token

**无偏性证明思路**：对任意 token $x$：
$$P(\text{output}=x) = P(\text{accept})q(x) + P(\text{reject})\frac{\max(0, p(x)-q(x))}{\sum_v \max(0, p(v)-q(v))} = p(x)$$

即最终输出分布精确等于 target model 的分布，与 draft model 无关。

### acceptance_rate（α）的含义

- α 越高，每轮接受的 token 越多，等效 target forward 次数越少，加速越大
- greedy（temperature=0）下：α = P(draft argmax == target argmax)
- 实测：Qwen2.5-0.5B → Qwen2.5-7B，K=4，α ≈ 55.85%

---

## 三、工程实现方式

### KV Cache 一致性挑战

draft 和 target 各维护独立的 KV cache。每轮迭代后，两边需要同步到相同的序列长度：
- draft KV：生成 K 步后 seq_len = prompt + output + K，若有 token 被拒绝需 **rollback**
- target KV：只写入被接受的 token，不能多写也不能少写

### v1 实现（双 forward）

```
每轮：
  [1] spec_verify_target(K tokens, use_cache=False) → logits[K,vocab]
      不写 KV，只取验证 logit
  [2] rejection_sample → accepted_tokens
  [3] spec_advance_target_kv(accepted_tokens, use_cache=True) → last_logit
      写入 KV，推进 seq_len
```

优点：逻辑清晰，两步分离，验证和提交独立，易于调试。
缺点：每轮 2 次 target forward，系统 overhead 大。

### v2 优化（单 forward）

```
每轮：
  [1] target_forward([context + draft_tokens], use_cache=True)
      同时取 logits（验证用）和 past_key_values（KV）
  [2] rejection_sample → n 个接受
  [3] rollback KV 到 seq_len + n（删除被拒绝位置后的 KV）
```

代价：需要 KV rollback（在 block tensor 上清除尾部块），但节省了一次 target forward，理论 speedup 更高。

### draft 生成（_draft_k_steps）

K 步顺序 decode，每步：
1. `kv_cache.ensure_next_slot`：确保块已分配
2. `spec_decode_one`：paged decode，返回 logit，不更新 state
3. 采样 token（greedy 或 sampling）
4. `kv_cache.advance_seq_lens`：递增 seq_len（flash_attn 会 in-place 写 KV）
5. `state.append_generated(token)`：更新 state

### 跨设备张量对齐

当 draft 和 target 在不同 GPU 时：
- `draft_probs` 在 cuda:0，`target_probs` 在 cuda:1 → `.to(device)` 对齐
- vocab_size 可能不同（0.5B: 151936 vs 7B: 152064）→ zero-padding 补齐或 truncate

---

## 四、设计取舍

| 决策 | v1 选择 | 代价 | 备注 |
|------|---------|------|------|
| KV 写入时机 | verify 和 advance 分离 | 每轮 2× target forward | 逻辑清晰，易 debug |
| rollback 实现 | 释放超出 seq_len 的 block | ref_count 必须判断（prefix cache） | v1 只需 rollback draft，v2 还需 rollback target |
| batch size | 只支持 1 | 不能并行多请求 | 多请求需要独立 draft/target state 管理 |
| vocab 对齐 | zero-padding | 极少量额外计算 | 不影响正确性（padding 位置 p_draft≈0，p_target≈0） |

---

## 五、常见误区

**误区 1：spec decoding 一定更快**

不对。v1 实现因系统 overhead 可能比 target-only 更慢。speedup 取决于：
- α（acceptance rate）：越高越好
- K（draft 候选数）：越大潜在收益越高，但 draft 也更贵
- 系统实现效率：double forward、跨设备拷贝都是隐性成本

**误区 2：acceptance_rate 越高越好，所以 K 越大越好**

K 越大时，acceptance_rate 在位置 K 处会下降（前面对的后面不一定对），平均接受 token ≈ $\sum_{i=0}^{K-1} \prod_{j=0}^{i} \alpha_j$，边际收益递减，同时 draft 的 K 次 forward 成本线性增加。

**误区 3：prefill 后不需要额外处理 first_token KV**

错误。prefill 只写入 prompt token 的 KV，first_token 的 KV 不会自动写入。如果 spec 循环的第一个 `spec_verify_target` 只看到 prompt KV，attention 少了 first_token 的上下文，后续所有验证都偏移 1 位。

**误区 4：draft 和 target 可以共享 KV cache**

不行。两个模型的 KV 值不同（参数不同），必须各自维护独立的 KV cache，各自的 block tensor。

---

## 六、和 mini-infer 的关系

Phase 11 在 mini-infer 中实现了 v1 spec decoding：

- `mini_infer/kv_cache.py`：`rollback_to()` — 释放超出 seq_len 的块（支持 ref_count）
- `mini_infer/model_runner.py`：三个 spec 接口（spec_decode_one / spec_verify_target / spec_advance_target_kv）
- `mini_infer/spec_engine.py`：SpecEngine，管理双引擎 + rejection sampling + KV 同步

实测（Qwen2.5-0.5B@cuda:0 + 7B@cuda:1，K=4，greedy）：
- acceptance_rate = 55.85%
- speedup = 0.65×（v1 双 forward overhead 超过算法收益）

v2 优化路径（已在 docstring 说明，非当前阶段任务）：合并 verify + advance 为一次 target forward，预期 speedup ≈ 1.3-1.6×。

---

## 七、进一步阅读

- Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding" (arxiv 2211.17192)
- Chen et al. 2023, "Accelerating Large Language Model Decoding with Speculative Sampling" (DeepMind 版本，类似算法)
- vLLM 的 speculative decoding 实现：支持 batch > 1，使用 tree attention 并行验证
- SpecInfer：tree-based speculation，draft 生成 token tree 而非线性序列
