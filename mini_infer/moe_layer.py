"""Backward-compat shim — 真实模块已移至 mini_infer.modeling.moe_layer。"""
from mini_infer.modeling.moe_layer import *  # noqa: F401, F403
from mini_infer.modeling.moe_layer import MoELayer, EPMoELayer, shard_moe_state_dict  # explicit
