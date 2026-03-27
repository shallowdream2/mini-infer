"""Backward-compat shim — 真实模块已移至 mini_infer.modeling.moe_model。"""
from mini_infer.modeling.moe_model import *  # noqa: F401, F403
from mini_infer.modeling.moe_model import SyntheticMoEConfig, SyntheticMoEModel  # explicit
