"""Backward-compat shim — 真实模块已移至 mini_infer.core.config。"""
from mini_infer.core.config import *  # noqa: F401, F403
from mini_infer.core.config import EngineConfig  # explicit for type checkers
