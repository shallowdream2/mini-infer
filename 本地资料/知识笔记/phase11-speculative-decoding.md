# 知识笔记：Speculative Decoding

## 核心思路

大模型 decode 是 memory-bound，计算单元利用率低。投机解码用小模型（draft）预测多个 token，大模型（target）批量验证——批量验证 K 个位置的代价与验证 1 个近似，如果多个 token 被接受，等效减少了大模型 forward 次数。

## Modified Rejection Sampling

关键性质：**输出分布等价于 target-only，无偏**。

设 draft 分布 q，target 分布 p，对 draft token x：
- accept probability = min(1, p(x)/q(x))
- 拒绝时：从 normalize(max(0, p-q)) 重采样，然后停止这一轮
- 全 K 个接受：bonus = target 在位置 n+K 的采样

直觉：p(x) > q(x) 时 100% 接受（draft 的建议是 target 更认同的）；p(x) < q(x) 时按比例接受。修正分布保证了整体期望等于 p。

## KV Cache 管理的两个动作

1. **verify**（`spec_verify_target`）：不写 KV，只取 logits，用于 rejection sampling 判断
2. **advance**（`spec_advance_target_kv`）：写入 KV，提交被接受的 token，供下一轮 attention 使用

v1 将两步分离，每轮 2 次 target forward → 慢。v2 合并为 1 次（verify 用 `use_cache=True`，拒绝后 rollback 多余 KV）→ 快。

## acceptance_rate 的直觉

greedy（temperature=0）下，两个模型在同一位置的 argmax 是否相同。
- AR=100%：draft 和 target 完全一致（小模型可以完全预测大模型）
- AR=55%（本项目实际值）：0.5B → 7B，超过一半位置意见一致，合理
- AR↑：draft 越接近 target，speedup 越大；AR↓：接近 target-only

## 实际 speedup 计算

每轮接受 ~(K×AR+1) 个 token，需要：v1 需 2 次 target forward，v2 需 1 次。

v1：`speedup ≈ (K×AR+1) / (2 + K×size_ratio)` → K=4, AR=55%, ratio=0.5/7 → ≈ 1.05，实际因系统 overhead 更低

v2：`speedup ≈ (K×AR+1) / 1` → ≈ 3.2，实际因 rollback/overhead ≈ 1.3-1.6×

## 容易踩的坑

1. prefill 后 first_token 的 KV 必须显式写入（不能用 `use_cache=False` 的 forward 来"只取 logit"）
2. draft/target 在不同 device 上：rejection sampling 前要对齐 device + vocab_size
3. draft seq_len 要在每步 ensure_next_slot 后手动 advance（dry_run 路径 flash_attn 不调用，不会自动推进）
4. rollback 时注意 ref_count（prefix cache 共享 block 不能直接 free）
