# Phase 11 开发日志 — Speculative Decoding

日期：2026-03-22

---

## 目标

在 mini-infer 中实现 speculative decoding：小模型（0.5B）预测 K 个候选 token，大模型（7B）一次 forward 验证，通过 rejection sampling 保证输出分布无偏。

---

## 实现过程

### Step 1：rollback_to 接口

在 `kv_cache.py` 新增 `rollback_to(request_id, seq_len)`。draft 每轮生成 K 步后可能有部分被拒绝，需要将 draft KV 回滚到正确位置。实现时注意 ref_count：prefix cache 的 block 有 ref_count > 1，不能直接 free。

### Step 2：model_runner 三个 spec 接口

- `spec_decode_one`：draft 模型一步 decode，返回 logit，不更新 state（调用方负责采样和推进 seq_len）
- `spec_verify_target`：从 block tensor 重建 KV，以 K 个 draft token 为输入跑 HF forward（`use_cache=False`），返回 logits[K, vocab]
- `spec_advance_target_kv`：以接受的 token 为输入跑 forward（`use_cache=True`），写入 block tensor，推进 seq_len

### Step 3：SpecEngine + rejection sampling

`spec_engine.py` 管理双引擎（draft@cuda:0，target@cuda:1），实现 modified rejection sampling。

关键细节：
- `_align_draft_prob`：统一处理跨设备 tensor 移动 + vocab_size pad（0.5B: 151936 vs 7B: 152064）
- `_draft_k_steps`：K 步 draft，每步 ensure_next_slot + spec_decode_one + 采样 + advance_seq_lens
- dry_run 下 spec_decode_one 不调用 flash_attn，需要手动递增 seq_len

### Step 4：GPU 调试（花时间最长的部分）

**坑 1：7B 模型路径**。忘记在对话开始时读 `project_env_state.md`，重新踩了"snapshot 子目录 shard 不完整"的坑。修复：用根目录 + `HF_HUB_OFFLINE=1`。

**坑 2：后台进程输出**。用 `run_in_background` 跑 GPU 脚本，stdout 被 pipe 缓冲，误判为"进程挂住"。修复：写日志到文件，或前台运行。

**坑 3：跨设备 tensor**。`RuntimeError: Expected all tensors on same device`，draft_probs 在 cuda:0，target_probs 在 cuda:1。修复：`_align_draft_prob` 统一 `.to(device)`。

**坑 4：vocab size 不匹配**。`size of tensor a (152064) must match b (151936)`。修复：`_align_draft_prob` 追加 zeros 补齐。

### Step 5：Review 发现 KV 位置 bug

`_get_prefill_last_logit` 用 `spec_verify_target(use_cache=False)` 获取 first_token 的 logit，但不写 KV，导致后续所有 target forward 缺少 first_token 的 KV 上下文，attention 偏移 1 位。

修复：删除 `_get_prefill_last_logit`，改为直接调 `spec_advance_target_kv([first_token])`，既写 KV 又返回 logit。

---

## 关键决策

**v1 双 forward 设计**：verify（不写 KV）+ advance（写 KV）分离，逻辑清晰，易于 debug。代价是每轮 2 次 7B forward。benchmark 结果：0.65× target-only。v2 可合并为一次 forward，预期 1.3-1.6×。

**batch=1 限制**：spec decoding 当前只支持 batch=1 顺序处理。多 prompt 情况下可以并行但需要每条 prompt 有独立的 draft/target state 管理，留给后续扩展。

---

## 最终数据

- acceptance_rate: 55.85%（K=4, greedy）
- spec throughput: ~29.0 tok/s（word count 近似）
- target-only: ~44.4 tok/s
- speedup: 0.65×
- tests: 13/13 passed
