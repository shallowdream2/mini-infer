"""Backward-compat shim — 真实模块已移至 mini_infer.core.request。"""
from mini_infer.core.request import *  # noqa: F401, F403
from mini_infer.core.request import Request, RequestState, SamplingParams  # explicit
