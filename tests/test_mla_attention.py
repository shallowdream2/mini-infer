"""
Phase 14 MLA 注意力测试。

覆盖范围：
  1. MLAAttentionNaive 与 MLAAttentionLatentCache 在相同权重下的数值等价性（CPU，无需 GPU）
  2. KV cache 大小压缩比：MLA latent / GQA(4KV,128dim) < 60%
  3. prefill 后继续 decode 的增量 forward（模拟生成循环）
"""

import torch
import pytest

from mini_infer.mla_attention import (
    MLAConfig,
    MLAAttentionNaive,
    MLAAttentionLatentCache,
    compute_kv_cache_bytes,
)


# ─────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────


def _copy_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """把 src 的所有参数复制到 dst（按名称匹配）。"""
    src_dict = dict(src.named_parameters())
    dst_dict = dict(dst.named_parameters())
    assert set(src_dict.keys()) == set(dst_dict.keys()), (
        f"参数名不一致:\n  src: {sorted(src_dict)}\n  dst: {sorted(dst_dict)}"
    )
    with torch.no_grad():
        for name, param in src_dict.items():
            dst_dict[name].copy_(param)


# ─────────────────────────────────────────
# 测试 1：数值等价性（prefill）
# ─────────────────────────────────────────


@pytest.fixture
def cfg():
    """使用 V2-Lite 超参，但 hidden_size 缩小到 64 以加快 CPU 测试。"""
    return MLAConfig(
        hidden_size=64,
        num_heads=4,
        q_lora_rank=None,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        kv_lora_rank=32,
        v_head_dim=16,
    )


def test_naive_vs_latent_prefill(cfg):
    """
    相同随机权重，prefill 时两个实现输出完全一致。
    允许误差 atol=1e-5（CPU float32）。
    """
    torch.manual_seed(42)
    naive = MLAAttentionNaive(cfg)
    latent = MLAAttentionLatentCache(cfg)
    _copy_weights(naive, latent)
    naive.eval()
    latent.eval()

    x = torch.randn(2, 8, cfg.hidden_size)  # batch=2, seq=8

    with torch.no_grad():
        out_naive, cache_naive = naive(x)
        out_latent, cache_latent = latent(x)

    assert out_naive.shape == out_latent.shape, "output shape mismatch"
    assert torch.allclose(out_naive, out_latent, atol=1e-5), (
        f"max diff = {(out_naive - out_latent).abs().max().item():.2e}"
    )


# ─────────────────────────────────────────
# 测试 2：数值等价性（prefill + decode 步骤）
# ─────────────────────────────────────────


def test_naive_vs_latent_decode_step(cfg):
    """
    prefill 后，第一个 decode step（seq_len=1）的输出两者一致。
    """
    torch.manual_seed(0)
    naive = MLAAttentionNaive(cfg)
    latent = MLAAttentionLatentCache(cfg)
    _copy_weights(naive, latent)
    naive.eval()
    latent.eval()

    # prefill
    x_prefill = torch.randn(1, 6, cfg.hidden_size)
    with torch.no_grad():
        _, cache_naive = naive(x_prefill)
        _, cache_latent = latent(x_prefill)

    # decode step（1 个新 token）
    x_decode = torch.randn(1, 1, cfg.hidden_size)
    with torch.no_grad():
        out_naive, _ = naive(x_decode, past_cache=cache_naive)
        out_latent, _ = latent(x_decode, past_cache=cache_latent)

    assert torch.allclose(out_naive, out_latent, atol=1e-5), (
        f"decode step max diff = {(out_naive - out_latent).abs().max().item():.2e}"
    )


# ─────────────────────────────────────────
# 测试 3：KV cache 大小压缩比
# ─────────────────────────────────────────


def test_kv_cache_compression_ratio():
    """
    MLA latent cache 相对于 GQA(4KV, 128dim) 的压缩比必须 < 60%。
    DeepSeek-V2-Lite 理论值：(512+64)*2 / (4*128*2*2) = 1152/2048 ≈ 56.25%。
    """
    result = compute_kv_cache_bytes(
        seq_len=1024,
        num_layers=27,
        gqa_num_kv_heads=4,
        gqa_head_dim=128,
        mla_num_heads=16,
        mla_q_head_dim=192,
        mla_v_head_dim=128,
        mla_kv_lora_rank=512,
        mla_qk_rope_head_dim=64,
    )
    ratio = result["mla_latent_vs_gqa_ratio"]
    assert ratio < 0.60, (
        f"MLA latent/GQA 压缩比 {ratio:.4f} >= 0.60，超出预期"
    )
    # 验证绝对值：latent 约 1,152 bytes/token/layer（V2-Lite）
    latent_per = result["mla_latent_bytes_per_token_layer"]
    assert latent_per == (512 + 64) * 2, (
        f"latent bytes per token/layer = {latent_per}，期望 {(512+64)*2}"
    )


# ─────────────────────────────────────────
# 测试 4：KV cache dataclass 形状检查
# ─────────────────────────────────────────


def test_cache_shapes(cfg):
    """
    检查两种 cache 的 tensor shape 符合预期。
    """
    torch.manual_seed(1)
    naive = MLAAttentionNaive(cfg)
    latent = MLAAttentionLatentCache(cfg)
    naive.eval()
    latent.eval()

    bsz, seq = 2, 5
    x = torch.randn(bsz, seq, cfg.hidden_size)

    with torch.no_grad():
        _, cache_n = naive(x)
        _, cache_l = latent(x)

    # naive: (bsz, num_heads, seq, q_head_dim) + (bsz, num_heads, seq, v_head_dim)
    assert cache_n.key_states.shape == (bsz, cfg.num_heads, seq, cfg.q_head_dim), (
        f"key_states shape = {cache_n.key_states.shape}"
    )
    assert cache_n.value_states.shape == (bsz, cfg.num_heads, seq, cfg.v_head_dim), (
        f"value_states shape = {cache_n.value_states.shape}"
    )

    # latent: (bsz, seq, kv_lora_rank) + (bsz, seq, qk_rope_head_dim)
    assert cache_l.compressed_kv.shape == (bsz, seq, cfg.kv_lora_rank), (
        f"compressed_kv shape = {cache_l.compressed_kv.shape}"
    )
    assert cache_l.k_pe.shape == (bsz, seq, cfg.qk_rope_head_dim), (
        f"k_pe shape = {cache_l.k_pe.shape}"
    )


# ─────────────────────────────────────────
# 测试 5：q_lora_rank 路径（非 V2-Lite，有 Q 压缩）
# ─────────────────────────────────────────


def test_q_lora_rank_path():
    """
    测试 q_lora_rank != None 的路径（V2/V3 用，V2-Lite 不用，但代码分支需覆盖）。
    """
    cfg_qlora = MLAConfig(
        hidden_size=64,
        num_heads=4,
        q_lora_rank=24,   # ← 有 Q 压缩
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        kv_lora_rank=32,
        v_head_dim=16,
    )
    torch.manual_seed(99)
    naive = MLAAttentionNaive(cfg_qlora)
    latent = MLAAttentionLatentCache(cfg_qlora)
    _copy_weights(naive, latent)
    naive.eval()
    latent.eval()

    x = torch.randn(1, 4, cfg_qlora.hidden_size)
    with torch.no_grad():
        out_n, _ = naive(x)
        out_l, _ = latent(x)

    assert torch.allclose(out_n, out_l, atol=1e-5), (
        f"q_lora_rank 路径 max diff = {(out_n - out_l).abs().max().item():.2e}"
    )
