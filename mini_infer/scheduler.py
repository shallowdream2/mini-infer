"""
Phase 2 调度器：在 Phase 1 基础上新增 continuous batching 所需的接口。

新增方法：
  - has_waiting()：检查等待队列是否非空
  - peek_next_waiting()：查看（不移除）下一个等待请求
  - pop_next_waiting()：取出下一个等待请求（不加入 running，由调用方决定）
  - add_to_running()：将请求加入 running dict
  - get_running_states()：返回当前所有 running 请求列表

Phase 1 的 get_next_batch() 接口仍保留，供已有测试和向后兼容使用。
"""

from collections import deque

from .request import RequestState


class Scheduler:
    """提供等待队列到运行队列的批次调度能力，Phase 2 支持 continuous batching。"""

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size 必须大于 0")
        self.max_batch_size = max_batch_size
        self._waiting: deque[RequestState] = deque()
        self._running: dict[str, RequestState] = {}

    def add_request(self, state: RequestState) -> None:
        self._waiting.append(state)

    # ------------------------------------------------------------------
    # Continuous batching 接口（Phase 2 新增）
    # ------------------------------------------------------------------

    def has_waiting(self) -> bool:
        return bool(self._waiting)

    def peek_next_waiting(self) -> RequestState | None:
        """查看等待队列头部请求，不移除。"""
        return self._waiting[0] if self._waiting else None

    def pop_next_waiting(self) -> RequestState:
        """取出等待队列头部请求（不加入 running）。"""
        return self._waiting.popleft()

    def add_to_running(self, state: RequestState) -> None:
        """将已确认可运行的请求加入 running dict。"""
        self._running[state.request.request_id] = state

    def get_running_states(self) -> list[RequestState]:
        return list(self._running.values())

    # ------------------------------------------------------------------
    # 共用接口
    # ------------------------------------------------------------------

    def finish_request(self, state: RequestState) -> None:
        self._running.pop(state.request.request_id, None)

    def num_waiting(self) -> int:
        return len(self._waiting)

    def num_running(self) -> int:
        return len(self._running)

    # ------------------------------------------------------------------
    # Phase 1 兼容接口（保留供测试使用）
    # ------------------------------------------------------------------

    def get_next_batch(self) -> list[RequestState]:
        """Phase 1 接口：一次取出最多 max_batch_size 个等待请求并加入 running。"""
        batch: list[RequestState] = []
        while self._waiting and len(batch) < self.max_batch_size:
            state = self._waiting.popleft()
            self._running[state.request.request_id] = state
            batch.append(state)
        return batch
