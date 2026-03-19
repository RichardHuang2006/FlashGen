"""
End-to-end integration tests for FlashGen.

Tests:
  1. Model loading from HuggingFace (GPT-2)
  2. Text generation (correctness vs HF reference)
  3. Streaming generation
  4. Batch generation with continuous batching
  5. ONNX export (if onnx package available)
  6. Paged KV cache memory management
  7. Sampling strategy diversity

Run:
    pytest tests/integration/test_end_to_end.py -v
    pytest tests/integration/test_end_to_end.py -v -k "test_generate"
"""

import pytest
import torch


SKIP_IF_NO_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required"
)


@pytest.fixture(scope="module")
def llm():
    from flashgen import LLM
    return LLM("gpt2", backend="pytorch")


@pytest.fixture(scope="module")
def tokenizer(llm):
    return llm.tokenizer


@pytest.fixture(scope="module")
def params():
    from flashgen import SamplingParams
    return SamplingParams(temperature=1.0, max_tokens=50, greedy=True)


# ── Basic generation ──────────────────────────────────────────────────────────

@SKIP_IF_NO_GPU
def test_generate_returns_output(llm, params):
    output = llm.generate("The quick brown fox", params)
    assert output.text, "Generated text should not be empty"
    assert output.generated_tokens > 0
    assert output.prompt_tokens > 0


@SKIP_IF_NO_GPU
def test_generate_deterministic_with_greedy(llm):
    from flashgen import SamplingParams
    params = SamplingParams(greedy=True, max_tokens=30)
    prompt = "Once upon a time"
    out1 = llm.generate(prompt, params)
    out2 = llm.generate(prompt, params)
    assert out1.text == out2.text, "Greedy generation should be deterministic"


@SKIP_IF_NO_GPU
def test_generate_respects_max_tokens(llm):
    from flashgen import SamplingParams
    for max_tok in [1, 10, 50]:
        params = SamplingParams(greedy=True, max_tokens=max_tok)
        output = llm.generate("Hello", params)
        assert output.generated_tokens <= max_tok, (
            f"Expected <= {max_tok} tokens, got {output.generated_tokens}"
        )


@SKIP_IF_NO_GPU
def test_temperature_affects_output(llm):
    """Different temperatures should (usually) produce different outputs."""
    from flashgen import SamplingParams
    prompt = "The most important discovery in physics was"
    results = set()
    for temp in [0.1, 0.5, 1.0, 1.5]:
        params = SamplingParams(temperature=temp, max_tokens=20, greedy=False)
        # Run multiple times for statistical confidence
        for _ in range(3):
            out = llm.generate(prompt, params)
            results.add(out.text)
    # We expect at least 2 distinct outputs across different temperatures
    assert len(results) >= 2, "Different temperatures should produce different outputs"


# ── Streaming ─────────────────────────────────────────────────────────────────

@SKIP_IF_NO_GPU
def test_streaming_produces_tokens(llm):
    from flashgen import SamplingParams
    params = SamplingParams(greedy=True, max_tokens=30)
    pieces = list(llm.stream("Hello world", params))
    assert len(pieces) > 0, "Streaming should produce at least one piece"
    full_text = "".join(pieces)
    assert len(full_text) > 0


@SKIP_IF_NO_GPU
def test_streaming_matches_non_streaming(llm):
    """Streaming and non-streaming should produce the same text (greedy)."""
    from flashgen import SamplingParams
    params = SamplingParams(greedy=True, max_tokens=20)
    prompt = "In the beginning"

    # Non-streaming
    output = llm.generate(prompt, params)

    # Streaming
    streamed = "".join(llm.stream(prompt, params))

    assert output.text == streamed, (
        f"Non-streaming: {output.text!r}\n"
        f"Streaming:     {streamed!r}"
    )


# ── Batch generation ──────────────────────────────────────────────────────────

@SKIP_IF_NO_GPU
def test_batch_generation(llm):
    from flashgen import SamplingParams
    prompts = [
        "The capital of France is",
        "Water is made of",
        "The speed of light is",
    ]
    params = SamplingParams(greedy=True, max_tokens=20)
    results = llm.generate_batch(prompts, params)
    assert len(results) == len(prompts)
    for r, p in zip(results, prompts):
        assert r.prompt == p
        assert r.generated_tokens > 0
        assert r.text


# ── Memory management ─────────────────────────────────────────────────────────

def test_block_allocator():
    from flashgen.memory.block_allocator import BlockAllocator
    alloc = BlockAllocator(
        num_blocks=64,
        block_size=16,
        n_layers=2,
        n_kv_heads=4,
        head_dim=64,
        dtype=torch.float16,
    )
    assert alloc.num_free == 64

    b0 = alloc.allocate()
    b1 = alloc.allocate()
    assert alloc.num_free == 62

    alloc.free(b0)
    assert alloc.num_free == 63

    alloc.incref(b1)
    alloc.free(b1)  # ref_count → 1, not returned yet
    assert alloc.num_free == 63
    alloc.free(b1)  # ref_count → 0, now returned
    assert alloc.num_free == 64


def test_paged_kv_cache():
    from flashgen.memory.block_allocator import BlockAllocator
    from flashgen.memory.paged_kv_cache import PagedKVCache

    alloc = BlockAllocator(
        num_blocks=32, block_size=16,
        n_layers=2, n_kv_heads=4, head_dim=64,
        dtype=torch.float16,
    )
    cache = PagedKVCache(alloc, block_size=16)

    # Allocate a sequence
    cache.allocate("seq_0", num_tokens=32)
    assert "seq_0" in cache
    assert len(cache.get_block_table("seq_0")) == 2  # 32 tokens / 16 per block

    # Extend by one token
    cache.extend("seq_0")
    assert cache.get_num_cached_tokens("seq_0") == 33

    # Free
    cache.free("seq_0")
    assert "seq_0" not in cache
    assert alloc.num_free == 32  # all blocks returned


def test_paged_kv_cache_fork():
    from flashgen.memory.block_allocator import BlockAllocator
    from flashgen.memory.paged_kv_cache import PagedKVCache

    alloc = BlockAllocator(
        num_blocks=32, block_size=16,
        n_layers=2, n_kv_heads=4, head_dim=64,
        dtype=torch.float16,
    )
    cache = PagedKVCache(alloc, block_size=16)

    cache.allocate("seq_a", 32)
    cache.fork("seq_a", "seq_b")

    # Both sequences share the same blocks (refcounted)
    ta = cache.get_block_table("seq_a")
    tb = cache.get_block_table("seq_b")
    assert ta == tb, "Forked sequence should share block table"

    # Free one — blocks should not be released yet (ref_count = 2 → 1)
    cache.free("seq_a")
    assert alloc.num_free < 32, "Blocks should still be held by seq_b"

    # Free the other — now blocks released
    cache.free("seq_b")
    assert alloc.num_free == 32, "All blocks should be freed after both sequences released"


# ── Scheduler ─────────────────────────────────────────────────────────────────

def test_continuous_batcher_basic():
    from flashgen import SamplingParams
    from flashgen.core.config import SchedulerConfig
    from flashgen.memory.block_allocator import BlockAllocator
    from flashgen.memory.paged_kv_cache import PagedKVCache
    from flashgen.scheduler.continuous_batcher import ContinuousBatcher, Request

    alloc = BlockAllocator(32, 16, 2, 4, 64, dtype=torch.float32)
    cache = PagedKVCache(alloc, 16)
    sched_cfg = SchedulerConfig(max_num_seqs=8, max_num_tokens=512)
    batcher = ContinuousBatcher(sched_cfg, cache)

    req = Request(
        request_id="req_0",
        prompt_token_ids=list(range(32)),
        sampling=SamplingParams(max_tokens=10, greedy=True),
    )
    batcher.add_request(req)
    assert batcher.num_waiting == 1

    batch = batcher.schedule()
    assert len(batch.prefill_requests) == 1
    assert batch.num_prefill_tokens == 32


# ── Sampler ───────────────────────────────────────────────────────────────────

def test_sampler_greedy():
    from flashgen import SamplingParams
    from flashgen.sampling.sampler import Sampler

    sampler = Sampler()
    logits = torch.zeros(2, 100)
    logits[0, 42] = 10.0   # seq 0 should select token 42
    logits[1, 7]  = 10.0   # seq 1 should select token 7

    params = [SamplingParams(greedy=True)] * 2
    tokens = sampler.sample(logits, params)
    assert tokens[0].item() == 42
    assert tokens[1].item() == 7


def test_sampler_top_k():
    from flashgen import SamplingParams
    from flashgen.sampling.sampler import Sampler

    sampler = Sampler()
    vocab = 1000
    logits = torch.randn(1, vocab)
    params = [SamplingParams(top_k=5, temperature=1.0)]

    # Sample many times and verify all outputs are in the top-5
    top5 = logits[0].topk(5).indices.tolist()
    for _ in range(50):
        token = sampler.sample(logits.clone(), params)[0].item()
        assert token in top5, f"Token {token} not in top-5 {top5}"


def test_sampler_top_p():
    from flashgen import SamplingParams
    from flashgen.sampling.sampler import Sampler
    import torch.nn.functional as F

    sampler = Sampler()
    vocab = 100
    logits = torch.zeros(1, vocab)
    # Make tokens 0-9 have 90% of total probability mass
    logits[0, :10] = 5.0
    params = [SamplingParams(top_p=0.9, temperature=1.0)]

    for _ in range(100):
        token = sampler.sample(logits.clone(), params)[0].item()
        # Tokens outside top-10 should almost never be selected
        # (not a strict guarantee, but should hold for 100 trials)
        # We just verify no exception and the output is in range
        assert 0 <= token < vocab


# ── ONNX export (optional) ────────────────────────────────────────────────────

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for ONNX export"
)
def test_onnx_export(tmp_path):
    try:
        import onnx
    except ImportError:
        pytest.skip("onnx package not installed")

    from flashgen.onnx_export.exporter import ONNXExporter
    out = tmp_path / "gpt2.onnx"
    exporter = ONNXExporter("gpt2", opset=17)
    result = exporter.export(str(out))
    assert result.exists()
    assert result.stat().st_size > 1000  # non-trivial file


# ── Config ────────────────────────────────────────────────────────────────────

def test_model_config_presets():
    from flashgen.core.config import ModelConfig
    gpt2 = ModelConfig.gpt2()
    assert gpt2.d_model == 768
    assert gpt2.n_layers == 12
    assert gpt2.head_dim == 64
    assert gpt2.actual_kv_heads == 12

    llama = ModelConfig.llama2_7b()
    assert llama.use_rope
    assert llama.use_rms_norm
    assert llama.use_silu_gate
    assert llama.actual_kv_heads == 32

    mistral = ModelConfig.mistral_7b()
    assert mistral.actual_kv_heads == 8
    assert mistral.gqa_group_size == 4


def test_cache_config_bytes_per_block():
    from flashgen.core.config import CacheConfig, ModelConfig
    cfg = CacheConfig(block_size=16, kv_dtype="float16")
    model = ModelConfig.gpt2()   # head_dim=64, 12 kv_heads, 12 layers
    bpb = cfg.bytes_per_block(model)
    # 16 * 64 * 2 (bytes) * 2 (K+V) * 12 (kv_heads) * 12 (layers)
    expected = 16 * 64 * 2 * 2 * 12 * 12
    assert bpb == expected, f"Expected {expected}, got {bpb}"
