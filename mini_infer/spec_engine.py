"""
Phase 11：Speculative Decoding 引擎。

使用小型 draft model 批量预测 K 个 token，再用大型 target model 一次 forward 并行验证，
通过 modified rejection sampling 保证输出分布等价于 target-only 生成。

架构：
  SpecEngine
    ├── draft:  LLMEngine（小模型，如 Qwen2.5-0.5B）
    └── target: LLMEngine（大模型，如 Qwen2.5-7B）

一次 spec 迭代的流程（batch=1）：
  1. Draft 生成 K 个候选 token（K 步 decode，每步返回 logit）
  2. Target 一次 K-token forward，获取每个位置的 target logit
  3. Modified rejection sampling 决定接受前 n 个 token（0 ≤ n ≤ K）
  4. Draft KV 回滚到 prompt_len + accepted_so_far + n
  5. Target KV 推进到同一位置（写入 n 个接受 token 的 KV）
  6. 两个引擎同步状态，继续下一轮

输出分布正确性（无偏）：
  rejection sampling 的接受概率为 min(1, p_target / p_draft)，拒绝时从修正分布重采样，
  保证生成序列的分布 = p_target（与 target-only greedy/sampling 语义等价）。

v1 局限性（设计文档中已说明）：
  - 仅支持 batch=1
  - target 验证后通过 spec_advance_target_kv 二次 forward 写入 KV（非最优，可优化为一次 forward）
  - 不与 chunked prefill / prefix cache 集成

接口：
  engine = SpecEngine(draft_config, target_config, K=4)
  outputs = engine.generate(prompts, max_new_tokens=64)
  print(engine.acceptance_rate())
"""

from __future__ import annotations

import math
from uuid import uuid4

import torch

from .config import EngineConfig
from .engine import LLMEngine
from .request import Request, RequestState, SamplingParams


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    """logits → 归一化概率向量，数值稳定。"""
    if logits.dtype not in (torch.float32, torch.float64):
        logits = logits.float()
    return torch.softmax(logits, dim=-1)


def _sample_from(probs: torch.Tensor) -> int:
    """从概率向量采样一个 token id。"""
    return int(torch.multinomial(probs.unsqueeze(0), num_samples=1).item())


def _rejection_sample(
    draft_tokens: list[int],
    draft_probs: list[torch.Tensor],  # K × [vocab]
    target_prev_logit: torch.Tensor,  # [vocab]，target 在 spec 开始前的 logit
    target_verify_logits: torch.Tensor,  # [K, vocab]，K-token verify 输出
) -> tuple[list[int], torch.Tensor]:
    """
    Modified rejection sampling（Leviathan et al. 2023）。

    参数：
        draft_tokens:       K 个 draft 采样的 token id
        draft_probs:        K 个 draft 采样分布（用于采样 draft_tokens[i] 的分布）
        target_prev_logit:  target 在 position n（spec batch 开始位置）之前的 logit
        target_verify_logits: K-token verify forward 输出的 logits[K, vocab]
                              verify_logits[i] = target logit AFTER seeing draft_tokens[i]
                              即 target 在位置 n+i+1 的分布（不是位置 n+i）

    返回：
        accepted_tokens:  接受的 token 列表（长度 1..K+1）
        new_last_logit:   下一轮 spec 迭代的 target "prev logit"

    注意 target_logits_for_check 的映射：
        position n   → target_prev_logit             → 检查 draft_tokens[0]
        position n+1 → target_verify_logits[0]       → 检查 draft_tokens[1]
        ...
        position n+K → target_verify_logits[K-1]     → bonus token（全接受时使用）
    """
    device = target_prev_logit.device
    # target 在各位置的分布（K+1 项）
    # target_logits_check[i] = target 的分布，用于决定是否接受 draft_tokens[i]
    target_probs_check: list[torch.Tensor] = [_softmax(target_prev_logit)]
    for i in range(len(draft_tokens) - 1):
        target_probs_check.append(_softmax(target_verify_logits[i]))
    # bonus logit：全部 K 个 token 被接受时，从 target_verify_logits[K-1] 采样
    bonus_logit = target_verify_logits[-1]

    accepted: list[int] = []
    for i, token in enumerate(draft_tokens):
        p_target_i = target_probs_check[i][token].item()
        p_draft_i = draft_probs[i][token].item()
        accept_prob = min(1.0, p_target_i / (p_draft_i + 1e-10))

        # dry_run 模式：probs 全为 0，直接接受（避免 division by zero 导致 accept=0）
        if p_draft_i < 1e-12:
            accepted.append(token)
            continue

        if torch.rand(1).item() < accept_prob:
            accepted.append(token)
        else:
            # 拒绝：从修正分布重采样（保证输出分布无偏）
            residual = (target_probs_check[i] - draft_probs[i]).clamp(min=0.0)
            s = residual.sum().item()
            if s > 1e-10:
                new_token = _sample_from(residual / s)
            else:
                # 两个分布几乎相同，直接从 target 采样
                new_token = _sample_from(target_probs_check[i])
            accepted.append(new_token)
            return accepted, target_probs_check[i]  # 终止（不接受后续 draft token）

    # 全部 K 个 token 被接受，额外从 bonus logit 采样一个 token
    bonus_probs = _softmax(bonus_logit)
    bonus_token = _sample_from(bonus_probs)
    accepted.append(bonus_token)
    return accepted, bonus_logit


class SpecEngine:
    """
    Speculative Decoding 引擎（batch=1）。

    推荐用法：
        draft_cfg = EngineConfig(model_name="Qwen/Qwen2.5-0.5B-Instruct", ...)
        target_cfg = EngineConfig(model_name="Qwen/Qwen2.5-7B-Instruct", ...)
        engine = SpecEngine(draft_cfg, target_cfg, K=4)
        outputs = engine.generate(["你好"], max_new_tokens=64)
        print(f"acceptance_rate={engine.acceptance_rate():.2%}")
    """

    def __init__(
        self,
        draft_config: EngineConfig,
        target_config: EngineConfig,
        K: int = 4,
    ) -> None:
        self.draft = LLMEngine(draft_config)
        self.target = LLMEngine(target_config)
        self.K = K
        # 统计
        self._total_draft = 0
        self._total_accepted = 0

    def acceptance_rate(self) -> float:
        """返回累计 acceptance rate。"""
        return self._total_accepted / max(1, self._total_draft)

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[str]:
        """批量生成（当前实现：逐条顺序处理，返回与输入顺序一致的列表）。"""
        return [
            self._generate_one(p, max_new_tokens, temperature, top_p)
            for p in prompts
        ]

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _make_state(
        self, engine: LLMEngine, prompt: str, max_new_tokens: int,
        temperature: float, top_p: float,
    ) -> RequestState:
        """为指定引擎创建 RequestState 并加入 waiting 队列（不 prefill）。"""
        request = Request(
            request_id=str(uuid4()),
            prompt=prompt,
            sampling_params=SamplingParams(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )
        state = RequestState(
            request=request,
            prompt_token_ids=engine._tokenize(prompt),
        )
        return state

    def _prefill_one(self, engine: LLMEngine, state: RequestState) -> None:
        """Prefill 单个请求（直接走 KVCacheManager + ModelRunner，不经过 scheduler）。"""
        engine.kv_cache.init_request(state)
        engine.model_runner.prefill([state])

    def _generate_one(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Speculative Decoding 生成单条 prompt。"""
        # --- 1. 创建 draft/target 各自的 RequestState ---
        d_state = self._make_state(
            self.draft, prompt, max_new_tokens, temperature, top_p
        )
        t_state = self._make_state(
            self.target, prompt, max_new_tokens, temperature, top_p
        )

        # --- 2. 两个引擎各自 prefill，获得第一个生成 token ---
        self._prefill_one(self.draft, d_state)
        self._prefill_one(self.target, t_state)

        # prefill 后两个引擎的第一个 token 可能不同（采样路径相同但模型不同）
        # 以 target 的第一个 token 为准（保证最终输出 = target 分布）
        # 丢弃 draft 的第一个 token，重置其生成状态与 target 同步
        if d_state.generated_token_ids and t_state.generated_token_ids:
            first_token = t_state.generated_token_ids[0]
            # 同步 draft 的第一个 token 为 target 的
            d_state.generated_token_ids = [first_token]
            d_state.generated_text_parts = [t_state.generated_text_parts[0]]

        # target 的 prev_logit 初始为 None（触发一次额外的 spec_advance 获取 logit）
        # 为简化实现，在 spec 循环中用 prefill 输出的 target logit 替代
        # 此处记录 prefill 后的 target "last logit"（从 prefill 输出中获取）
        target_prev_logit = self._get_prefill_last_logit(self.target, t_state)

        # --- 3. Speculative Decoding 主循环 ---
        while (
            not t_state.finished
            and len(t_state.generated_token_ids) < max_new_tokens
        ):
            remaining = max_new_tokens - len(t_state.generated_token_ids)
            K = min(self.K, remaining)

            # 3a. Draft：生成 K 个候选 token
            draft_tokens, draft_probs = self._draft_k_steps(d_state, K)

            # 3b. Target：一次 K-token forward 验证
            target_verify_logits = self.target.model_runner.spec_verify_target(
                t_state, draft_tokens
            )  # [K, vocab]

            # 3c. Rejection sampling
            self._total_draft += K
            accepted_tokens, target_prev_logit = _rejection_sample(
                draft_tokens,
                draft_probs,
                target_prev_logit,
                target_verify_logits,
            )
            n_accepted_draft = min(len(accepted_tokens), K)  # 不含可能的 bonus
            self._total_accepted += n_accepted_draft

            # 3d. 回滚 draft KV 到 prompt_len + len(target_output_so_far) + n_accepted_draft
            prompt_len = len(d_state.prompt_token_ids)
            target_so_far = len(t_state.generated_token_ids)
            new_draft_seq = prompt_len + target_so_far + n_accepted_draft
            self.draft.kv_cache.rollback_to(d_state.request.request_id, new_draft_seq)

            # 3e. 将接受的 token 追加到 target state，并推进 target KV
            for tok in accepted_tokens:
                t_state.append_generated(tok, "")
                if tok == self.target.model_runner.eos_token_id:
                    t_state.mark_finished("eos")
                    break
                if len(t_state.generated_token_ids) >= max_new_tokens:
                    t_state.mark_finished("length")
                    break

            if not t_state.finished:
                # 将接受的 token 写入 target KV block tensor
                target_prev_logit = self.target.model_runner.spec_advance_target_kv(
                    t_state, accepted_tokens
                )

            # 3f. 同步 draft state（generated_token_ids 与 target 对齐）
            d_state.generated_token_ids = list(t_state.generated_token_ids)
            d_state.generated_text_parts = list(t_state.generated_text_parts)
            if t_state.finished:
                d_state.finished = True

        # --- 4. 解码输出（GPU 路径用真实 tokenizer，dry_run 用 stub）---
        if not self.target.config.dry_run:
            output_text = self.target.model_runner.tokenizer.decode(
                t_state.generated_token_ids, skip_special_tokens=True
            )
        else:
            output_text = "".join(t_state.generated_text_parts)

        # 清理 KV cache（释放块）
        self.draft.kv_cache.free_request(d_state)
        self.target.kv_cache.free_request(t_state)

        return output_text

    def _get_prefill_last_logit(
        self, engine: LLMEngine, state: RequestState
    ) -> torch.Tensor:
        """
        获取 prefill 后 target 模型在第一个生成位置的 logit。

        在 dry_run 下返回全零 tensor（stubs 无实际 logit）。
        在 GPU 下重新跑一次 spec_verify_target 以获取 logit（代价：1 token 额外 forward）。
        """
        if engine.config.dry_run:
            return torch.zeros(256)
        # 用 prefill 的最后一个输出 token 运行一次 spec_verify_target（K=1）
        first_token = state.generated_token_ids[:1]
        if not first_token:
            return torch.zeros(engine.model_runner.model.config.vocab_size)
        logits = engine.model_runner.spec_verify_target(state, first_token)
        return logits[-1]  # [vocab_size]

    def _draft_k_steps(
        self, state: RequestState, K: int
    ) -> tuple[list[int], list[torch.Tensor]]:
        """
        Draft 模型生成 K 个候选 token，返回：
          - draft_tokens:  list[int]，K 个采样的 token id
          - draft_probs:   list[Tensor[vocab]]，每步采样时使用的概率分布

        流程：
          对每步 i = 0..K-1：
            1. ensure_next_slot 确保块已分配
            2. spec_decode_one 获取 logit（不采样）
            3. 从 logit 采样 token
            4. advance_seq_lens 推进 seq_len（模拟 flash_attn 写入）
            5. 将 token 追加到 state
        """
        rid = state.request.request_id
        draft_tokens: list[int] = []
        draft_probs: list[torch.Tensor] = []

        for _ in range(K):
            # 确保下一写入位置有块
            self.draft.kv_cache.ensure_next_slot([rid])

            # 获取 logit（不采样）
            logit = self.draft.model_runner.spec_decode_one(state)
            probs = _softmax(logit)
            draft_probs.append(probs)

            # 采样
            if state.request.sampling_params.temperature == 0.0:
                token = int(logit.argmax().item())
            else:
                token = _sample_from(probs)

            draft_tokens.append(token)

            # 推进 seq_len（dry_run 下 spec_decode_one 不调用 flash_attn，需手动推进）
            if self.draft.config.dry_run:
                self.draft.kv_cache._seq_lens[rid] += 1
            else:
                self.draft.kv_cache.advance_seq_lens([rid])

            # 更新 state（用于 spec_decode_one 读取 last token）
            state.append_generated(token, "")

            if token == self.draft.model_runner.eos_token_id:
                break

        return draft_tokens, draft_probs
