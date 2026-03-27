"""Backward-compat shim — 真实模块已移至 mini_infer.serving.server。"""
from mini_infer.serving.server import *  # noqa: F401, F403
from mini_infer.serving.server import app, _default_engine_config  # explicit
