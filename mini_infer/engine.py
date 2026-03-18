"""这个文件定义统一引擎入口，串联 tokenizer、调度器、KV cache 和模型执行器。"""

from uuid import uuid4

from .config import EngineConfig
from .kv_cache import KVCacheManager
from .model_runner import ModelRunner
from .request import Request, RequestState, SamplingParams
from .scheduler import Scheduler


class LLMEngine:
    """提供 generate 接口，串联 tokenizer、调度和模型执行。"""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.kv_cache = KVCacheManager(block_size=config.block_size)
        self.scheduler = Scheduler(max_batch_size=config.max_batch_size)
        self.model_runner = ModelRunner(config=config, kv_cache=self.kv_cache)

    def generate(self, prompts: list[str], max_new_tokens: int = 128) -> list[str]:
        states: list[RequestState] = []
        for prompt in prompts:
            request = Request(
                request_id=str(uuid4()),
                prompt=prompt,
                sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
            )
            state = RequestState(
                request=request,
                prompt_token_ids=self._tokenize(prompt),
            )
            self.scheduler.add_request(state)
            states.append(state)

        outputs: dict[str, str] = {}
        while len(outputs) < len(states):
            batch = self.scheduler.get_next_batch()
            if not batch:
                break

            for state in batch:
                self.kv_cache.init_request(state)

            self.model_runner.prefill(batch)

            while any(not state.finished for state in batch):
                self.model_runner.decode_step(batch)

            for state in batch:
                # 用 tokenizer 整体 decode，避免逐 token decode 的子词拼接问题
                outputs[state.request.request_id] = self.model_runner.tokenizer.decode(
                    state.generated_token_ids, skip_special_tokens=True
                )
                self.model_runner.free_request(state)
                self.kv_cache.free_request(state)
                self.scheduler.finish_request(state)

        return [outputs[state.request.request_id] for state in states]

    def _tokenize(self, text: str) -> list[int]:
        return self.model_runner.tokenizer.encode(text, add_special_tokens=True)
