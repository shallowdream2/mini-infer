"""这个文件定义引擎级基础配置，是后续真实推理实现的统一配置入口。"""

from dataclasses import dataclass


@dataclass(slots=True)
class EngineConfig:
    """保存引擎的最小配置集合。"""

    model_name: str
    device: str = "cuda:0"
    dtype: str = "float16"
    max_batch_size: int = 8
    max_model_len: int = 2048
    block_size: int = 16
    enable_paged_kv_cache: bool = True
    tokenizer_name: str | None = None  # 若为 None，则使用 model_name
    dry_run: bool = False  # 为 True 时使用桩实现，不加载真实模型，供无 GPU 或单元测试使用

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size 必须大于 0")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len 必须大于 0")
        if self.block_size <= 0:
            raise ValueError("block_size 必须大于 0")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype 只支持 float16、bfloat16 或 float32")
        if not self.device:
            raise ValueError("device 不能为空")
