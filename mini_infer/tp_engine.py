"""Backward-compat shim — 真实模块已移至 mini_infer.parallel.tp_engine。"""
from mini_infer.parallel.tp_engine import *  # noqa: F401, F403
from mini_infer.parallel.tp_engine import TPEngine  # explicit
