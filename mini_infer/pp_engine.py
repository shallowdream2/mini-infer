"""Backward-compat shim — 真实模块已移至 mini_infer.parallel.pp_engine。"""
from mini_infer.parallel.pp_engine import *  # noqa: F401, F403
from mini_infer.parallel.pp_engine import PPEngine  # explicit
