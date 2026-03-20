"""
向后兼容模块：TPEngine 是 PPEngine 的别名。

原 tp_engine.py 已重命名为 pp_engine.py，类名从 TPEngine 改为 PPEngine，
原因：device_map="balanced" 实现的是 Pipeline Parallel（PP），不是 Tensor Parallel（TP）。
  - PP：不同层运行在不同 GPU，层间传递激活张量
  - TP（真正的张量并行）：同一层按 head 切分到多 GPU，层内需要 all-reduce

此文件保留以避免破坏已有导入（如 from mini_infer.tp_engine import TPEngine）。
新代码请使用 mini_infer.pp_engine.PPEngine。
"""

from .pp_engine import PPEngine as TPEngine

__all__ = ["TPEngine"]
