"""mini_infer.modeling — 模型执行器、量化、MLA、MoE。"""

from .mla_attention import (
    MLAAttentionAbsorbed,
    MLAAttentionLatentCache,
    MLAAttentionNaive,
)
from .model_runner import ModelRunner
from .quantization import QuantLinear, QuantMode, quantize_model

try:
    from .moe_layer import EPMoELayer, MoELayer, shard_moe_state_dict
    from .moe_model import SyntheticMoEConfig, SyntheticMoEModel
    _moe_available = True
except ImportError:
    _moe_available = False

__all__ = [
    "ModelRunner",
    "QuantLinear", "QuantMode", "quantize_model",
    "MLAAttentionNaive", "MLAAttentionLatentCache", "MLAAttentionAbsorbed",
]

if _moe_available:
    __all__ += ["MoELayer", "EPMoELayer", "shard_moe_state_dict",
                "SyntheticMoEConfig", "SyntheticMoEModel"]
