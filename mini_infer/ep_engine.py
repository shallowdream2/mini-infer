"""Backward-compat shim — 真实模块已移至 mini_infer.parallel.ep_engine。"""
from mini_infer.parallel.ep_engine import *  # noqa: F401, F403
from mini_infer.parallel.ep_engine import EPEngine  # explicit
