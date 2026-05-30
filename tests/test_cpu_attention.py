"""CPU/MPS SDPA fallback attention tests."""

import torch
import torch.nn.functional as F

from mini_infer.kernels.attention import sdpa_paged_decode_attention


def test_sdpa_paged_decode_attention_writes_kv_and_matches_sdpa() -> None:
    """SDPA fallback 应写入新 KV，并与连续 KV 的 SDPA 结果一致。"""
    k_cache = torch.zeros(2, 2, 1, 2, dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    k_cache[0, 0, 0] = torch.tensor([1.0, 0.0])
    k_cache[0, 1, 0] = torch.tensor([0.0, 1.0])
    v_cache[0, 0, 0] = torch.tensor([1.0, 2.0])
    v_cache[0, 1, 0] = torch.tensor([3.0, 4.0])

    q = torch.tensor([[[[1.0, 1.0]]]])
    k_new = torch.tensor([[[[1.0, 1.0]]]])
    v_new = torch.tensor([[[[5.0, 6.0]]]])
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    cache_seqlens = torch.tensor([2], dtype=torch.int32)

    out = sdpa_paged_decode_attention(
        q, k_new, v_new, k_cache, v_cache, block_table, cache_seqlens
    )

    expected_k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    expected_v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        expected_k,
        expected_v,
        attn_mask=torch.ones(1, 1, 1, 3, dtype=torch.bool),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)

    assert out.shape == (1, 1, 1, 2)
    assert torch.allclose(out, expected)
    assert torch.equal(k_cache[1, 0, 0], k_new[0, 0, 0])
    assert torch.equal(v_cache[1, 0, 0], v_new[0, 0, 0])


def test_sdpa_paged_decode_attention_repeats_kv_heads_for_gqa() -> None:
    """GQA 场景下 KV heads 应 repeat 到 query heads。"""
    k_cache = torch.zeros(1, 1, 1, 2, dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    k_new = torch.tensor([[[[1.0, 1.0]]]])
    v_new = torch.tensor([[[[2.0, 3.0]]]])
    block_table = torch.tensor([[0]], dtype=torch.int32)
    cache_seqlens = torch.tensor([0], dtype=torch.int32)

    out = sdpa_paged_decode_attention(
        q, k_new, v_new, k_cache, v_cache, block_table, cache_seqlens
    )

    assert out.shape == (1, 1, 2, 2)
    assert torch.allclose(out[0, 0, 0], v_new[0, 0, 0])
    assert torch.allclose(out[0, 0, 1], v_new[0, 0, 0])
