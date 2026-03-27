"""Backward-compat shim — 真实模块已移至 mini_infer.cache.kv_transfer。"""
from mini_infer.cache.kv_transfer import *  # noqa: F401, F403
from mini_infer.cache.kv_transfer import KVPayload, KVSender, KVReceiver, extract_kv_from_past  # explicit
