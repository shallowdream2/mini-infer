"""这个文件定义当前阶段的最小调度器，用于表达等待队列、运行队列和批次选择逻辑。"""

from collections import deque

from .request import RequestState


class Scheduler:
    """提供当前骨架阶段的最小批次调度能力。"""

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size 必须大于 0")
        self.max_batch_size = max_batch_size
        self._waiting: deque[RequestState] = deque()
        self._running: dict[str, RequestState] = {}

    def add_request(self, state: RequestState) -> None:
        self._waiting.append(state)

    def get_next_batch(self) -> list[RequestState]:
        batch: list[RequestState] = []
        while self._waiting and len(batch) < self.max_batch_size:
            state = self._waiting.popleft()
            self._running[state.request.request_id] = state
            batch.append(state)
        return batch

    def finish_request(self, state: RequestState) -> None:
        self._running.pop(state.request.request_id, None)

    def num_waiting(self) -> int:
        return len(self._waiting)

    def num_running(self) -> int:
        return len(self._running)