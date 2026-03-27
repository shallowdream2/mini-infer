"""Backward-compat shim — 真实模块已移至 mini_infer.runtime.engine。"""
from mini_infer.runtime.engine import *  # noqa: F401, F403
from mini_infer.runtime.engine import LLMEngine  # explicit
