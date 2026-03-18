"""这个文件定义初级 KV cache 元数据管理器，用于表达后续分页缓存的最小接口和状态。"""

from dataclasses import dataclass
from math import ceil

from .request import RequestState


@dataclass(slots=True)
class KVAllocation:
    """记录单个请求当前占用的缓存块数量和 token 数量。"""

    allocated_blocks: int = 0
    cached_tokens: int = 0


class KVCacheManager:
    """提供当前阶段最小可用的 KV cache 元数据管理逻辑。"""

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size 必须大于 0")
        self.block_size = block_size
        self._allocations: dict[str, KVAllocation] = {}

    def init_request(self, state: RequestState) -> None:
        prompt_tokens = len(state.prompt_token_ids)
        allocated_blocks = ceil(prompt_tokens / self.block_size) if prompt_tokens else 0
        self._allocations[state.request.request_id] = KVAllocation(
            allocated_blocks=allocated_blocks,
            cached_tokens=prompt_tokens,
        )

    def append_token(self, state: RequestState, token_id: int) -> None:
        del token_id
        allocation = self._allocations[state.request.request_id]
        total_tokens = len(state.prompt_token_ids) + len(state.generated_token_ids)
        allocation.cached_tokens = total_tokens
        allocation.allocated_blocks = ceil(total_tokens / self.block_size) if total_tokens else 0

    def free_request(self, state: RequestState) -> None:
        self._allocations.pop(state.request.request_id, None)

    def get_allocation(self, request_id: str) -> KVAllocation | None:
        return self._allocations.get(request_id)

    def total_allocated_blocks(self) -> int:
        return sum(item.allocated_blocks for item in self._allocations.values())