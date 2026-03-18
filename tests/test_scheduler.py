"""这个文件覆盖调度器骨架的最小批次行为，验证等待队列到运行队列的状态流转。"""

from mini_infer.request import Request, RequestState, SamplingParams
from mini_infer.scheduler import Scheduler


def build_state(request_id: str) -> RequestState:
    return RequestState(
        request=Request(
            request_id=request_id,
            prompt=f"prompt-{request_id}",
            sampling_params=SamplingParams(max_new_tokens=2),
        )
    )


def test_scheduler_batching() -> None:
    scheduler = Scheduler(max_batch_size=2)
    scheduler.add_request(build_state("a"))
    scheduler.add_request(build_state("b"))
    scheduler.add_request(build_state("c"))

    batch = scheduler.get_next_batch()

    assert [state.request.request_id for state in batch] == ["a", "b"]
    assert scheduler.num_waiting() == 1
    assert scheduler.num_running() == 2