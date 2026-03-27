"""Backward-compat shim — 真实模块已移至 mini_infer.modeling.model_runner。"""
from mini_infer.modeling.model_runner import *  # noqa: F401, F403
from mini_infer.modeling.model_runner import ModelRunner  # explicit
