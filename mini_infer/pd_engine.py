"""Backward-compat shim — 真实模块已移至 mini_infer.runtime.pd_engine。"""
from mini_infer.runtime.pd_engine import *  # noqa: F401, F403
from mini_infer.runtime.pd_engine import PDEngine  # explicit
