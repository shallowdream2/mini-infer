"""这个文件导出 mini-infer 的公开接口，供外部统一导入。"""

from __future__ import annotations

from importlib import import_module

from .config import EngineConfig
from .request import Request, RequestState, SamplingParams

_LAZY_IMPORTS = {
    "LLMEngine": (".engine", "LLMEngine"),
    "PPEngine": (".pp_engine", "PPEngine"),
    "ReplicaEngine": (".replica_engine", "ReplicaEngine"),
    "SpecEngine": (".spec_engine", "SpecEngine"),
    "TPEngine": (".tp_engine", "TPEngine"),
}

__all__ = [
    "EngineConfig",
    "LLMEngine",
    "PPEngine",
    "ReplicaEngine",
    "SpecEngine",
    "TPEngine",  # 向后兼容，新代码请使用 PPEngine
    "Request",
    "RequestState",
    "SamplingParams",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
