"""Backward-compat shim — 真实模块已移至 mini_infer.runtime.scheduler。"""
from mini_infer.runtime.scheduler import *  # noqa: F401, F403
from mini_infer.runtime.scheduler import Scheduler  # explicit
