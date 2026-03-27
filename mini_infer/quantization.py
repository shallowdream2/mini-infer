"""Backward-compat shim — 真实模块已移至 mini_infer.modeling.quantization。"""
from mini_infer.modeling.quantization import *  # noqa: F401, F403
from mini_infer.modeling.quantization import QuantLinear, QuantMode, quantize_model  # explicit
