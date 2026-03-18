"""这个文件覆盖 KV cache 骨架的最小块增长逻辑，验证缓存块数量随 token 增长而变化。"""

from mini_infer.kv_cache import KVCacheManager
from mini_infer.request import Request, RequestState, SamplingParams


def build_state(prompt: str) -> RequestState:
    return RequestState(
        request=Request(
            request_id="req-1",
            prompt=prompt,
            sampling_params=SamplingParams(max_new_tokens=4),
        ),
        prompt_token_ids=[1 for _ in prompt],
    )


def test_kv_cache_block_growth() -> None:
    manager = KVCacheManager(block_size=4)
    state = build_state("hello")

    manager.init_request(state)
    allocation = manager.get_allocation(state.request.request_id)

    assert allocation is not None
    assert allocation.allocated_blocks == 2

    state.generated_token_ids.extend([1, 2, 3])
    manager.append_token(state, 3)
    allocation = manager.get_allocation(state.request.request_id)
    assert allocation is not None
    assert allocation.allocated_blocks == 2

    state.generated_token_ids.append(4)
    manager.append_token(state, 4)
    allocation = manager.get_allocation(state.request.request_id)
    assert allocation is not None
    assert allocation.allocated_blocks == 3