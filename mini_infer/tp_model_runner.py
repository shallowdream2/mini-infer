"""Backward-compat shim — 真实模块已移至 mini_infer.parallel.tp_model_runner。"""
from mini_infer.parallel.tp_model_runner import *  # noqa: F401, F403
from mini_infer.parallel.tp_model_runner import (  # explicit (including private names used in tests)
    TensorParallelModelRunner,
    col_shard,
    row_shard,
    _shard_qwen2_weights,
)
