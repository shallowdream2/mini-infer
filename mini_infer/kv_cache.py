"""Backward-compat shim — 真实模块已移至 mini_infer.cache.kv_cache。"""
from mini_infer.cache.kv_cache import *  # noqa: F401, F403
from mini_infer.cache.kv_cache import KVCacheManager  # explicit
