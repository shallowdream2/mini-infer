"""Real-model CPU inference smoke tests."""

from __future__ import annotations

import pytest
from transformers import AutoConfig

from mini_infer import EngineConfig, LLMEngine


@pytest.mark.real_model
def test_cpu_real_model_generates_one_token(real_model_path: str) -> None:
    """CPU 路径应能用真实 HF 模型完成最小 generate 链路。"""
    hf_config = AutoConfig.from_pretrained(real_model_path, trust_remote_code=True)
    head_dim = getattr(
        hf_config,
        "head_dim",
        hf_config.hidden_size // hf_config.num_attention_heads,
    )
    num_kv_heads = getattr(
        hf_config,
        "num_key_value_heads",
        hf_config.num_attention_heads,
    )

    config = EngineConfig(
        model_name=real_model_path,
        tokenizer_name=real_model_path,
        device="cpu",
        dtype="float32",
        max_batch_size=1,
        max_model_len=128,
        block_size=16,
        num_gpu_blocks=32,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    engine = LLMEngine(config)
    outputs = engine.generate(["Hello"], max_new_tokens=1, temperature=0.0)

    assert len(outputs) == 1
    assert isinstance(outputs[0], str)
