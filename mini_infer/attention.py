"""Backward-compat shim — 真实模块已移至 mini_infer.kernels.attention。"""
from mini_infer.kernels.attention import *  # noqa: F401, F403
from mini_infer.kernels.attention import PagedDecodeContext, patch_model_for_paged_decode  # explicit
