"""Backward-compat shim — 真实模块已移至 mini_infer.runtime.spec_engine。"""
from mini_infer.runtime.spec_engine import *  # noqa: F401, F403
from mini_infer.runtime.spec_engine import (  # explicit (including private names used in tests)
    SpecEngine,
    _rejection_sample,
    _softmax,
)
