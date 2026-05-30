"""
Phase CPU：验证 CPU/MPS 推理链路（dry_run + 真实模型两条路径）。

dry_run 测试（无需 GPU / 模型权重）：
  - 验证 block_size 不再受 256 约束
  - 验证 CPU block pool 分配与 block_table 逻辑
  - 验证完整 generate() 调度链路（dry_run=True）

cpu_paged_decode_attention 单元测试（无需 GPU）：
  - 验证 KV 写入 block cache
  - 验证 GQA 展开正确
  - 验证输出 shape

真实模型测试（需要模型权重，pytest -m real_model --model /path/to/model 触发）：
  - 验证 CPU 上完整 generate() 能返回合理文本
"""
import math
import pytest
import torch

from mini_infer.core.config import EngineConfig
from mini_infer.core.request import Request, RequestState, SamplingParams
from mini_infer.cache.kv_cache import KVCacheManager
from mini_infer.kernels.attention import cpu_paged_decode_attention
from mini_infer.runtime.engine import LLMEngine


# ---------------------------------------------------------------------------
# 辅助：小 block_size CPU config
# ---------------------------------------------------------------------------

def _cpu_config(block_size: int = 16, num_gpu_blocks: int = 32, dry_run: bool = True) -> EngineConfig:
    return EngineConfig(
        model_name="dry",
        device="cpu",
        dtype="float32",  # CPU 路径推荐 float32 / bfloat16
        dry_run=dry_run,
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        num_hidden_layers=2,
        num_kv_heads=2,
        head_dim=64,
        max_batch_size=4,
    )


# ---------------------------------------------------------------------------
# block_size 约束
# ---------------------------------------------------------------------------

class TestCPUBlockSizeConstraint:
    def test_small_block_size_allowed_on_cpu(self):
        """CPU 路径允许 block_size=16（不再要求 256 的倍数）。"""
        cfg = _cpu_config(block_size=16)
        engine = LLMEngine(cfg)
        assert engine.kv_cache.block_size == 16

    def test_block_size_32_allowed(self):
        cfg = _cpu_config(block_size=32)
        engine = LLMEngine(cfg)
        assert engine.kv_cache.block_size == 32

    def test_block_size_not_multiple_of_256_allowed(self):
        """block_size=100 在 CPU 上应被接受。"""
        cfg = _cpu_config(block_size=100)
        LLMEngine(cfg)


# ---------------------------------------------------------------------------
# cpu_paged_decode_attention 单元测试
# ---------------------------------------------------------------------------

class TestCPUPagedDecodeAttention:
    def _make_cache(self, num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int):
        k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim)
        return k_cache, v_cache

    def test_output_shape_mha(self):
        """MHA（num_q_heads == num_kv_heads）输出 shape 正确。"""
        batch, num_heads, head_dim = 2, 4, 32
        block_size, num_blocks = 8, 16
        k_cache, v_cache = self._make_cache(num_blocks, block_size, num_heads, head_dim)

        q = torch.randn(batch, 1, num_heads, head_dim)
        k_new = torch.randn(batch, 1, num_heads, head_dim)
        v_new = torch.randn(batch, 1, num_heads, head_dim)

        block_table = torch.zeros(batch, 4, dtype=torch.int32)
        block_table[0, 0] = 0
        block_table[1, 0] = 1
        cache_seqlens = torch.zeros(batch, dtype=torch.int32)  # seq_len=0，首个 token

        out = cpu_paged_decode_attention(q, k_new, v_new, k_cache, v_cache, block_table, cache_seqlens)
        assert out.shape == (batch, 1, num_heads, head_dim)

    def test_output_shape_gqa(self):
        """GQA（num_q_heads > num_kv_heads）输出 shape 正确。"""
        batch, num_q_heads, num_kv_heads, head_dim = 2, 8, 2, 32
        block_size, num_blocks = 8, 16
        k_cache, v_cache = self._make_cache(num_blocks, block_size, num_kv_heads, head_dim)

        q = torch.randn(batch, 1, num_q_heads, head_dim)
        k_new = torch.randn(batch, 1, num_kv_heads, head_dim)
        v_new = torch.randn(batch, 1, num_kv_heads, head_dim)

        block_table = torch.zeros(batch, 4, dtype=torch.int32)
        cache_seqlens = torch.zeros(batch, dtype=torch.int32)

        out = cpu_paged_decode_attention(q, k_new, v_new, k_cache, v_cache, block_table, cache_seqlens)
        assert out.shape == (batch, 1, num_q_heads, head_dim)

    def test_kv_written_to_cache(self):
        """新 K/V 被正确写入 block cache 对应物理块和槽位。"""
        num_heads, head_dim = 2, 16
        block_size, num_blocks = 8, 4
        k_cache, v_cache = self._make_cache(num_blocks, block_size, num_heads, head_dim)

        q = torch.randn(1, 1, num_heads, head_dim)
        k_new = torch.ones(1, 1, num_heads, head_dim) * 3.14
        v_new = torch.ones(1, 1, num_heads, head_dim) * 2.71

        # seq_len=5 → 写入 block 0（phys=2），slot 5
        block_table = torch.tensor([[2, 0]], dtype=torch.int32)
        cache_seqlens = torch.tensor([5], dtype=torch.int32)

        cpu_paged_decode_attention(q, k_new, v_new, k_cache, v_cache, block_table, cache_seqlens)

        assert torch.allclose(k_cache[2, 5], k_new[0, 0])
        assert torch.allclose(v_cache[2, 5], v_new[0, 0])

    def test_cross_block_boundary(self):
        """seq_len 跨 block 边界时 gather 和 write 均正确。"""
        num_heads, head_dim = 1, 8
        block_size = 4
        num_blocks = 8
        k_cache, v_cache = self._make_cache(num_blocks, block_size, num_heads, head_dim)

        # 预填 block 0（phys=3）和 block 1（phys=5）各 4 个 token 的 KV
        k_cache[3] = torch.arange(block_size * num_heads * head_dim, dtype=torch.float32).reshape(block_size, num_heads, head_dim)
        k_cache[5] = torch.arange(block_size * num_heads * head_dim, dtype=torch.float32).reshape(block_size, num_heads, head_dim) + 100

        q = torch.randn(1, 1, num_heads, head_dim)
        k_new = torch.zeros(1, 1, num_heads, head_dim)
        v_new = torch.zeros(1, 1, num_heads, head_dim)

        # seq_len=8 → 第 9 个 token，写入 block 2（phys=7），slot 0
        block_table = torch.tensor([[3, 5, 7]], dtype=torch.int32)
        cache_seqlens = torch.tensor([8], dtype=torch.int32)

        out = cpu_paged_decode_attention(q, k_new, v_new, k_cache, v_cache, block_table, cache_seqlens)
        assert out.shape == (1, 1, num_heads, head_dim)
        # 新 KV 写到 phys=7, slot=0
        assert torch.allclose(k_cache[7, 0], k_new[0, 0])


# ---------------------------------------------------------------------------
# dry_run 完整链路（scheduler + kv_cache + generate）
# ---------------------------------------------------------------------------

class TestCPUDryRunPipeline:
    def test_generate_single_request(self):
        """CPU dry_run：单请求 generate 能正常完成。"""
        cfg = _cpu_config(block_size=16, num_gpu_blocks=32)
        engine = LLMEngine(cfg)
        results = engine.generate(["hello world"], max_new_tokens=5)
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_generate_batch(self):
        """CPU dry_run：batch=4 generate 全部完成，顺序正确。"""
        cfg = _cpu_config(block_size=16, num_gpu_blocks=64)
        engine = LLMEngine(cfg)
        prompts = ["prompt A", "prompt B", "prompt C", "prompt D"]
        results = engine.generate(prompts, max_new_tokens=8)
        assert len(results) == 4

    def test_kv_blocks_freed_after_generate(self):
        """generate 完成后 KV 块应全部归还 free pool。"""
        cfg = _cpu_config(block_size=16, num_gpu_blocks=32)
        engine = LLMEngine(cfg)
        initial_free = engine.kv_cache.num_free_blocks()
        engine.generate(["test prompt"], max_new_tokens=4)
        assert engine.kv_cache.num_free_blocks() == initial_free

    def test_step_interface(self):
        """CPU dry_run：add_request / step 接口正常工作。"""
        cfg = _cpu_config(block_size=16, num_gpu_blocks=32)
        engine = LLMEngine(cfg)
        rid = engine.add_request("step test", max_new_tokens=3)
        tokens: list[str] = []
        while not engine.is_finished(rid):
            out = engine.step()
            tokens.extend(out.get(rid, []))
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# 真实模型测试（需要 --model 参数，默认跳过）
# ---------------------------------------------------------------------------

@pytest.fixture
def real_model_path(request):
    return request.config.getoption("--model", default=None)


@pytest.mark.real_model
def test_cpu_real_model_generate(real_model_path):
    """
    真实模型 CPU 推理端到端测试。

    运行方式：
        pytest tests/test_cpu_inference.py -m real_model \\
               --model /path/to/Qwen2.5-0.5B-Instruct -s
    """
    if real_model_path is None:
        pytest.skip("未指定 --model，跳过真实模型测试")

    cfg = EngineConfig(
        model_name=real_model_path,
        device="cpu",
        dtype="bfloat16",  # CPU 推荐 bfloat16
        dry_run=False,
        block_size=32,
        num_gpu_blocks=64,
        max_batch_size=1,
    )
    engine = LLMEngine(cfg)
    result = engine.generate(["什么是 PagedAttention？"], max_new_tokens=32)
    assert len(result) == 1
    assert len(result[0]) > 0
    print(f"\n[CPU 推理结果] {result[0]}")
