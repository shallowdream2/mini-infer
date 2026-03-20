"""
请求调度器，支持 continuous batching（Phase 2）和 preemption / 优先级调度（Phase 7）。

核心接口：
  waiting 队列管理：add_request, has_waiting, peek_next_waiting, pop_next_waiting
  running 队列管理：add_to_running, get_running_states, finish_request
  Preemption（Phase 7）：mark_swapped, has_swapped, move_swapped_to_running,
                          get_lowest_priority_running, un_admit

get_next_batch() 为 Phase 1 遗留接口，仅供 test_scheduler.py 使用，新代码请勿调用。
"""

from collections import deque

from .request import RequestState


class Scheduler:
    """提供等待队列到运行队列的批次调度能力，Phase 2 支持 continuous batching，Phase 7 支持抢占。"""

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size 必须大于 0")
        self.max_batch_size = max_batch_size
        self._waiting: deque[RequestState] = deque()
        self._running: dict[str, RequestState] = {}
        self._swapped: deque[RequestState] = deque()  # Phase 7：已换出到 CPU 的请求

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
    # Phase 7：Preemption 接口
    # ------------------------------------------------------------------

    def mark_swapped(self, state: RequestState) -> None:
        """将请求从 running 移到 swapped 队列（swap_out 后调用）。"""
        self._running.pop(state.request.request_id, None)
        self._swapped.append(state)

    def has_swapped(self) -> bool:
        return bool(self._swapped)

    def num_swapped(self) -> int:
        return len(self._swapped)

    def get_swapped_states(self) -> list[RequestState]:
        """按换出顺序（FIFO）返回所有换出请求。"""
        return list(self._swapped)

    def move_swapped_to_running(self, state: RequestState) -> None:
        """将请求从 swapped 移到 running（swap_in 后调用，跳过 prefill 直接参与 decode）。"""
        try:
            self._swapped.remove(state)
        except ValueError:
            pass
        self._running[state.request.request_id] = state

    def un_admit(self, state: RequestState) -> None:
        """
        撤销准入：将请求从 running 移回 waiting 队尾（不走 swap_out 路径）。

        仅用于刚准入但尚未 prefill 的请求（state.prefilled == False）。
        放到队尾而非队头，避免与高优先级请求的循环抢占。
        """
        self._running.pop(state.request.request_id, None)
        self._waiting.append(state)

    def get_lowest_priority_running(self) -> RequestState | None:
        """
        返回 running 中优先级最低的请求（priority 数值最大）。
        若 running 为空返回 None。优先级相同时返回最早加入的请求（dict 插入顺序）。
        """
        if not self._running:
            return None
        return max(self._running.values(), key=lambda s: s.request.priority)

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
