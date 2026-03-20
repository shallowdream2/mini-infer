"""
Phase 3 / Phase 6 Paged KV Cache 管理器。

Phase 6 新增接口（True PagedAttention）：
  - ensure_next_slot()：decode 前预分配下一 token 的物理块
  - build_block_tables()：构造 flash_attn_with_kvcache 所需的 block_table / cache_seqlens 张量
  - advance_seq_lens()：flash_attn in-place 写完 KV 后递增 _seq_lens

Phase 3 优化：gather_batch_kv() 从嵌套 Python 循环改为向量化 advanced indexing，
减少 Python 解释器开销，所有 block gather 操作合并为单次 CUDA kernel 调用。

核心设计：预分配固定大小的 GPU block tensor 池，每个请求持有一个 BlockTable
（逻辑块号 → 物理块号的映射），通过 FreeBlockPool（deque）管理可用物理块。
相比 Phase 1 的 HF past_key_values dict，优势在于：
  - 显存有上限（num_gpu_blocks 固定）
  - 支持 batch decode（不同长度请求共享同一个 block pool）
  - free_request 立即归还物理块，无显存碎片

dry_run=True 时不分配 GPU tensor，仅做块管理逻辑的元数据追踪，供无 GPU 环境的测试使用。
"""

import math
from collections import deque
from dataclasses import dataclass

import torch

from .config import EngineConfig
from .request import RequestState


def _resolve_dtype(dtype: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]


@dataclass(slots=True)
class KVAllocation:
    """记录单个请求当前的块分配情况，供外部查询和测试使用。"""

    allocated_blocks: int = 0
    cached_tokens: int = 0


class KVCacheManager:
    """
    Paged KV cache manager：预分配 GPU block tensor 池，用 BlockTable 管理每个请求。

    存储格式：
        k_cache[layer_idx] shape: [num_gpu_blocks, block_size, num_kv_heads, head_dim]
        v_cache[layer_idx] shape: 同上

    每个 block 存储 block_size 个 token 的 KV，物理块由 _free_blocks 统一管理。
    """

    def __init__(self, config: EngineConfig) -> None:
        self.block_size = config.block_size
        self.num_gpu_blocks = config.num_gpu_blocks
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.device = config.device
        self._dry_run = config.dry_run

        if not config.dry_run:
            dtype = _resolve_dtype(config.dtype)
            # k_cache[l][block_id, slot_id, num_kv_heads, head_dim]
            self.k_cache: list[torch.Tensor] = [
                torch.zeros(
                    self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_dim,
                    device=self.device, dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
            self.v_cache: list[torch.Tensor] = [
                torch.zeros(
                    self.num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_dim,
                    device=self.device, dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]

        # 空闲物理块队列
        self._free_blocks: deque[int] = deque(range(self.num_gpu_blocks))
        # 每个请求的逻辑块 → 物理块映射
        self._block_tables: dict[str, list[int]] = {}
        # 每个请求已存入的 KV token 数量
        self._seq_lens: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 块池管理
    # ------------------------------------------------------------------

    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def _allocate_block(self) -> int:
        if not self._free_blocks:
            raise RuntimeError("KV cache 已满：没有可用的空闲块，请减少并发请求数或增大 num_gpu_blocks")
        return self._free_blocks.popleft()

    # ------------------------------------------------------------------
    # 请求生命周期
    # ------------------------------------------------------------------

    def init_request(self, state: RequestState) -> None:
        """为新请求分配 prompt 所需的初始物理块，并初始化 seq_len。"""
        request_id = state.request.request_id
        prompt_len = len(state.prompt_token_ids)
        num_blocks = max(1, math.ceil(prompt_len / self.block_size))
        self._block_tables[request_id] = [self._allocate_block() for _ in range(num_blocks)]
        # seq_len 初始化为 prompt_len：prefill 写完后，KV 位置 0..prompt_len-1 会被填充
        self._seq_lens[request_id] = prompt_len

    def free_request(self, state: RequestState) -> None:
        """归还请求的所有物理块到空闲池。"""
        request_id = state.request.request_id
        blocks = self._block_tables.pop(request_id, [])
        self._free_blocks.extend(blocks)
        self._seq_lens.pop(request_id, None)

    # ------------------------------------------------------------------
    # KV 写入（prefill 阶段）
    # ------------------------------------------------------------------

    def write_prefill_kv(self, request_id: str, past_key_values: tuple) -> None:
        """
        把 HF 模型 prefill 输出的 past_key_values 写入 block tensor。

        past_key_values: tuple of (K, V) per layer
            K shape: [1, num_kv_heads, prompt_len, head_dim]
        """
        if self._dry_run:
            # dry_run 模式：seq_len 已在 init_request 中设置，无需操作
            return

        actual_layers = len(past_key_values)
        if actual_layers != self.num_layers:
            raise ValueError(
                f"模型输出了 {actual_layers} 层 KV，但 config.num_hidden_layers={self.num_layers}。"
                f"请确认 EngineConfig 的 num_hidden_layers 与模型匹配。"
            )

        prompt_len = past_key_values[0][0].shape[2]
        expected_blocks = max(1, math.ceil(prompt_len / self.block_size))
        actual_blocks = len(self._block_tables[request_id])
        if prompt_len != self._seq_lens[request_id]:
            raise ValueError(
                f"请求 {request_id!r}：init_request 时 prompt_len={self._seq_lens[request_id]}，"
                f"但 past_key_values 显示 seq_len={prompt_len}。tokenize 结果与模型输入不一致。"
            )
        if actual_blocks < expected_blocks:
            raise RuntimeError(
                f"请求 {request_id!r}：prompt 需要 {expected_blocks} 块，但只分配了 {actual_blocks} 块。"
            )

        block_table = self._block_tables[request_id]

        for l in range(self.num_layers):
            # k: [1, num_kv_heads, prompt_len, head_dim] → squeeze → permute → [prompt_len, heads, dim]
            k = past_key_values[l][0][0].permute(1, 0, 2)  # [prompt_len, num_kv_heads, head_dim]
            v = past_key_values[l][1][0].permute(1, 0, 2)

            for blk_idx, phys_blk in enumerate(block_table):
                start = blk_idx * self.block_size
                end = min(start + self.block_size, prompt_len)
                n = end - start
                if n <= 0:
                    break
                self.k_cache[l][phys_blk, :n] = k[start:end]
                self.v_cache[l][phys_blk, :n] = v[start:end]

    # ------------------------------------------------------------------
    # KV 读取（decode 阶段：gather 用于 batch forward）
    # ------------------------------------------------------------------

    def gather_batch_kv(
        self,
        request_ids: list[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        """
        为 batch decode 聚合所有请求的 KV。使用左填充（left-padding）对齐到 max_seq_len。

        Phase 3 优化：用向量化 advanced indexing 替代嵌套 Python 循环，
        将所有块的 gather 合并为单次 tensor 索引操作。

        算法：
          1. 构建 block_table_tensor [batch, max_num_blocks]
          2. 计算每个输出位置对应的 token 位置（含左填充偏移）
          3. 由 token 位置推算物理块号和块内 slot 号（两个 [batch, max_seq_len] 索引张量）
          4. 一次 advanced indexing 完成 gather，乘以 valid_mask 置零填充位置
          5. permute → [batch, num_kv_heads, max_seq_len, head_dim]

        返回：
            k_batch: list[num_layers]，每个 shape [batch, num_kv_heads, max_seq_len, head_dim]
            v_batch: 同上
            seq_lens: 每个请求的实际 seq_len
        """
        seq_lens = [self._seq_lens[rid] for rid in request_ids]
        max_seq_len = max(seq_lens)
        batch_size = len(request_ids)

        # --- 1. 构建 block table tensor [batch, max_num_blocks] ---
        max_num_blocks = max(len(self._block_tables[rid]) for rid in request_ids)
        block_table_tensor = torch.zeros(
            batch_size, max_num_blocks, dtype=torch.long, device=self.device
        )
        for b, rid in enumerate(request_ids):
            blocks = self._block_tables[rid]
            block_table_tensor[b, : len(blocks)] = torch.tensor(
                blocks, dtype=torch.long, device=self.device
            )

        # --- 2. 计算 token 位置（含左填充偏移）---
        # seq_lens_t: [batch, 1], out_positions: [1, max_seq_len]
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=self.device).unsqueeze(1)
        out_positions = torch.arange(max_seq_len, device=self.device).unsqueeze(0)
        # token_positions[b, i] = i - (max_seq_len - seq_len[b])；负值为填充区
        token_positions = out_positions - (max_seq_len - seq_lens_t)  # [batch, max_seq_len]

        # --- 3. 推算物理块号和 slot 号（填充区 clamp 到 0，结果被 mask 置零）---
        valid_mask = token_positions >= 0  # [batch, max_seq_len]
        token_pos_clamped = token_positions.clamp(min=0)

        block_indices = token_pos_clamped // self.block_size   # [batch, max_seq_len]
        slot_indices  = token_pos_clamped % self.block_size    # [batch, max_seq_len]

        # phys_blocks[b, i] = block_table_tensor[b, block_indices[b, i]]
        batch_range = (
            torch.arange(batch_size, device=self.device)
            .unsqueeze(1)
            .expand_as(block_indices)
        )
        phys_blocks = block_table_tensor[batch_range, block_indices]  # [batch, max_seq_len]

        # valid_mask_f 用于置零填充区，shape [batch, max_seq_len, 1, 1]，dtype 与 cache 一致
        cache_dtype = self.k_cache[0].dtype
        valid_mask_f = valid_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=cache_dtype)

        # --- 4 & 5. 各层 gather + permute ---
        k_batch: list[torch.Tensor] = []
        v_batch: list[torch.Tensor] = []

        for l in range(self.num_layers):
            # k_cache[l]: [num_gpu_blocks, block_size, num_kv_heads, head_dim]
            # advanced indexing → [batch, max_seq_len, num_kv_heads, head_dim]
            k_tokens = self.k_cache[l][phys_blocks, slot_indices] * valid_mask_f
            v_tokens = self.v_cache[l][phys_blocks, slot_indices] * valid_mask_f
            # permute → [batch, num_kv_heads, max_seq_len, head_dim]
            k_batch.append(k_tokens.permute(0, 2, 1, 3))
            v_batch.append(v_tokens.permute(0, 2, 1, 3))

        return k_batch, v_batch, seq_lens

    # ------------------------------------------------------------------
    # KV 写回（decode 阶段：写入新 token 的 KV）
    # ------------------------------------------------------------------

    def write_decode_kv(
        self,
        request_ids: list[str],
        k_new_layers: list[torch.Tensor] | None,
        v_new_layers: list[torch.Tensor] | None,
    ) -> None:
        """
        把 decode forward 输出的新 token KV 写回 block tensor，并递增 seq_len。

        k_new_layers[l] shape: [batch, num_kv_heads, head_dim]
        dry_run 模式或 k_new_layers=None 时，仅更新块管理元数据（seq_len + 块分配）。
        """
        for b, rid in enumerate(request_ids):
            token_pos = self._seq_lens[rid]
            block_idx = token_pos // self.block_size

            # 如需新块则分配
            if block_idx >= len(self._block_tables[rid]):
                self._block_tables[rid].append(self._allocate_block())

            if not self._dry_run and k_new_layers is not None:
                slot_idx = token_pos % self.block_size
                phys_blk = self._block_tables[rid][block_idx]
                for l in range(self.num_layers):
                    # k_new_layers[l][b] shape: [num_kv_heads, head_dim]
                    self.k_cache[l][phys_blk, slot_idx] = k_new_layers[l][b]
                    self.v_cache[l][phys_blk, slot_idx] = v_new_layers[l][b]  # type: ignore[index]

            self._seq_lens[rid] += 1

    # ------------------------------------------------------------------
    # Phase 7：Preemption — GPU ↔ CPU KV 换出 / 换入
    # ------------------------------------------------------------------

    def swap_out(self, state: "RequestState") -> None:
        """
        将请求的 GPU KV 拷贝到 CPU，释放 GPU 物理块。

        执行后：
          - state.swapped_seq_len = 换出时的 seq_len（用于 swap_in 重建块分配）
          - state.cpu_kv = [(k_cpu_l0, v_cpu_l0), ...]（真实模式，dry_run 下为 None）
          - GPU 物理块已归还到 _free_blocks
        """
        request_id = state.request.request_id
        seq_len = self._seq_lens[request_id]
        state.swapped_seq_len = seq_len

        if not self._dry_run:
            block_table = self._block_tables[request_id]
            cpu_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for l in range(self.num_layers):
                k_cpu = torch.zeros(seq_len, self.num_kv_heads, self.head_dim)
                v_cpu = torch.zeros(seq_len, self.num_kv_heads, self.head_dim)
                for blk_idx, phys_blk in enumerate(block_table):
                    start = blk_idx * self.block_size
                    end = min(start + self.block_size, seq_len)
                    n = end - start
                    if n <= 0:
                        break
                    k_cpu[start:end] = self.k_cache[l][phys_blk, :n].cpu()
                    v_cpu[start:end] = self.v_cache[l][phys_blk, :n].cpu()
                cpu_kv.append((k_cpu, v_cpu))
            state.cpu_kv = cpu_kv

        # 释放 GPU 块（dry_run 下只做元数据清理）
        self.free_request(state)

    def swap_in(self, state: "RequestState") -> None:
        """
        将 CPU KV 拷贝回 GPU，重新分配物理块，恢复请求的 KV cache 状态。

        执行后：
          - GPU 物理块已重新分配，block_table 和 seq_len 已恢复
          - state.cpu_kv = None（CPU 副本已释放）
        """
        request_id = state.request.request_id
        seq_len = state.swapped_seq_len
        num_blocks = max(1, math.ceil(seq_len / self.block_size))

        # 重新分配 GPU 块
        self._block_tables[request_id] = [self._allocate_block() for _ in range(num_blocks)]
        self._seq_lens[request_id] = seq_len

        if not self._dry_run and state.cpu_kv is not None:
            block_table = self._block_tables[request_id]
            for l in range(self.num_layers):
                k_cpu, v_cpu = state.cpu_kv[l]
                for blk_idx, phys_blk in enumerate(block_table):
                    start = blk_idx * self.block_size
                    end = min(start + self.block_size, seq_len)
                    n = end - start
                    if n <= 0:
                        break
                    self.k_cache[l][phys_blk, :n] = k_cpu[start:end].to(self.device)
                    self.v_cache[l][phys_blk, :n] = v_cpu[start:end].to(self.device)

        state.cpu_kv = None

    # ------------------------------------------------------------------
    # 查询接口（向后兼容）
    # ------------------------------------------------------------------

    def get_allocation(self, request_id: str) -> KVAllocation | None:
        """返回请求的块分配摘要，供测试和监控使用。"""
        if request_id not in self._block_tables:
            return None
        return KVAllocation(
            allocated_blocks=len(self._block_tables[request_id]),
            cached_tokens=self._seq_lens.get(request_id, 0),
        )

    def total_allocated_blocks(self) -> int:
        return sum(len(b) for b in self._block_tables.values())

    # ------------------------------------------------------------------
    # Phase 6：True PagedAttention（flash_attn block_tables）接口
    # ------------------------------------------------------------------

    def ensure_next_slot(self, request_ids: list[str]) -> None:
        """
        确保每个请求下一个 decode 位置已有物理块。
        必须在 build_block_tables() 之前调用，否则 block_table 会缺失新 token 对应的块。
        """
        for rid in request_ids:
            token_pos = self._seq_lens[rid]
            block_idx = token_pos // self.block_size
            if block_idx >= len(self._block_tables[rid]):
                self._block_tables[rid].append(self._allocate_block())

    def build_block_tables(
        self,
        request_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        为 flash_attn_with_kvcache 构造 block_table 和 cache_seqlens 张量。

        返回：
            block_table:   [batch, max_blocks_per_seq] int32，padding 填 0
            cache_seqlens: [batch] int32，新 token 写入前各请求的 cache 长度

        注意：调用前必须先调用 ensure_next_slot()，确保下一写入位置的块已分配。
        """
        batch_size = len(request_ids)
        max_num_blocks = max(len(self._block_tables[rid]) for rid in request_ids)

        block_table = torch.zeros(
            batch_size, max_num_blocks, dtype=torch.int32, device=self.device
        )
        for b, rid in enumerate(request_ids):
            blocks = self._block_tables[rid]
            block_table[b, : len(blocks)] = torch.tensor(
                blocks, dtype=torch.int32, device=self.device
            )

        cache_seqlens = torch.tensor(
            [self._seq_lens[rid] for rid in request_ids],
            dtype=torch.int32,
            device=self.device,
        )
        return block_table, cache_seqlens

    def advance_seq_lens(self, request_ids: list[str]) -> None:
        """
        flash_attn_with_kvcache 已 in-place 写入新 token KV 后，递增各请求的 seq_len。
        替代 write_decode_kv 的 seq_len 更新部分（GPU KV 写入由 flash_attn 完成）。
        """
        for rid in request_ids:
            self._seq_lens[rid] += 1

    # ------------------------------------------------------------------
    # 废弃接口（保留以减少测试迁移成本，不再有实际功能）
    # ------------------------------------------------------------------

    def append_token(self, state: RequestState, token_id: int) -> None:
        """已废弃：Phase 2 中 KV 写回通过 write_decode_kv 完成。"""
