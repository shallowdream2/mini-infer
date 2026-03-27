"""Backward-compat shim — 真实模块已移至 mini_infer.modeling.mla_attention。"""
from mini_infer.modeling.mla_attention import *  # noqa: F401, F403
from mini_infer.modeling.mla_attention import (  # explicit
    MLAAttentionNaive,
    MLAAttentionLatentCache,
    MLAAttentionAbsorbed,
)
