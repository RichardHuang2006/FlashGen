"""
AsyncLLMEngine — the main inference engine orchestrating all components.

Architecture
────────────
The engine integrates:
  ┌─────────────────────────────────────────────────────────┐
  │                    AsyncLLMEngine                        │
  │                                                          │
  │  HFModelLoader → weights                                 │
  │      ↓                                                   │
  │  Backend (CUDA custom kernels or TRT engine)             │
  │      ↓                                                   │
  │  BlockAllocator → PagedKVCache → ContinuousBatcher       │
  │      ↓                                                   │
  │  Sampler → streaming callbacks → RequestOutput           │
  └─────────────────────────────────────────────────────────┘

The engine runs a background thread that executes the continuous batching
loop. Python callers submit requests via add_request() and receive results
either synchronously (generate()) or via callbacks (submit()).

Backend selection:
  "cuda" — uses the PyBind11 C++ InferenceEngine with custom CUDA kernels.
            Full paged attention, FlashAttention-2, continuous batching.
  "trt"  — uses the TRTEngine (Python) for transformer forward pass,
            with Python-side paged KV cache and scheduler.
  "pytorch" — pure PyTorch HuggingFace model, for correctness baseline.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from typing import AsyncGenerator, Callable, Dict, Iterator, List, Optional

import torch

from flashgen.core.config import EngineConfig, SamplingParams
from flashgen.memory.block_allocator import BlockAllocator
from flashgen.memory.paged_kv_cache import PagedKVCache
from flashgen.model_loader.hf_loader import HFModelLoader
from flashgen.model_loader.tokenizer import Tokenizer
from flashgen.outputs import CompletionOutput, FinishReason, RequestOutput
from flashgen.profiling.nsight import NsightProfiler, nvtx_range
from flashgen.sampling.sampler import Sampler
from flashgen.scheduler.continuous_batcher import (
    ContinuousBatcher,
    Request,
    RequestStatus,
    SchedulerOutput,
)

logger = logging.getLogger(__name__)


class AsyncLLMEngine:
    """Production-grade LLM inference engine with continuous batching.

    Example
    -------
    >>> engine = AsyncLLMEngine.from_config(
    ...     EngineConfig(hf_model_id="gpt2", backend="pytorch")
    ... )
    >>> output = engine.generate("Hello, world!", SamplingParams(max_tokens=50))
    >>> print(output.text)
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config=None,
        weights=None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.config = engine_config
        self._tokenizer = tokenizer
        self._profiler = NsightProfiler()
        self._sampler = Sampler()

        # Load model + tokenizer
        if weights is None or model_config is None:
            loader = HFModelLoader(
                engine_config.hf_model_id,
                precision=engine_config.precision,
            )
            model_config, weights = loader.load()
        self._model_config = model_config

        if tokenizer is None:
            self._tokenizer = Tokenizer(engine_config.hf_model_id)

        # Build memory allocator + KV cache
        torch.cuda.set_device(engine_config.device_id)
        free_bytes, _ = torch.cuda.mem_get_info()
        cache_cfg = engine_config.cache
        kv_dtype = torch.float16 if cache_cfg.kv_dtype == "float16" else torch.float32

        self._allocator = BlockAllocator(
            num_blocks=cache_cfg.auto_num_blocks(model_config, free_bytes)
            if cache_cfg.num_gpu_blocks < 0 else cache_cfg.num_gpu_blocks,
            block_size=cache_cfg.block_size,
            n_layers=model_config.n_layers,
            n_kv_heads=model_config.actual_kv_heads,
            head_dim=model_config.head_dim,
            dtype=kv_dtype,
        )
        self._kv_cache = PagedKVCache(self._allocator, cache_cfg.block_size)
        self._scheduler = ContinuousBatcher(engine_config.scheduler, self._kv_cache)

        # Build model backend
        self._backend = self._build_backend(engine_config, model_config, weights)

        # Engine loop
        self._request_queue: queue.Queue = queue.Queue()
        self._response_map: Dict[str, queue.Queue] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.info(
            f"Engine ready: model={model_config.name}, "
            f"backend={engine_config.backend}, "
            f"blocks={self._allocator.num_blocks} "
            f"({self._allocator.num_blocks * cache_cfg.block_size} tokens capacity)"
        )

    @classmethod
    def from_config(cls, config: EngineConfig) -> "AsyncLLMEngine":
        return cls(config)

    # ── Backend construction ──────────────────────────────────────────────────

    def _build_backend(self, cfg: EngineConfig, model_config, weights):
        backend = cfg.backend.lower()

        if backend == "trt":
            if cfg.trt_engine_path is None:
                raise ValueError("trt_engine_path must be set when backend='trt'.")
            from flashgen.trt_builder.engine_runner import TRTEngine
            return TRTEngine(
                cfg.trt_engine_path,
                n_layers=model_config.n_layers,
                n_kv_heads=model_config.actual_kv_heads,
                head_dim=model_config.head_dim,
            )

        if backend == "cuda":
            # Try to load the C++ extension
            try:
                from flashgen import _C
                return _C.InferenceEngine  # type: ignore[attr-defined]
            except ImportError:
                logger.warning(
                    "flashgen._C not built — falling back to PyTorch backend. "
                    "Run 'pip install -e .' to build CUDA kernels."
                )
                backend = "pytorch"

        if backend == "pytorch":
            return self._build_pytorch_backend(cfg, model_config, weights)

        raise ValueError(f"Unknown backend: {cfg.backend!r}. Choose 'cuda', 'trt', or 'pytorch'.")

    def _build_pytorch_backend(self, cfg, model_config, weights):
        """Pure PyTorch HuggingFace backend for correctness testing."""
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_model_id,
            dtype=torch.float16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start the background engine loop thread."""
        self._running = True
        self._thread = threading.Thread(target=self._engine_loop, daemon=True)
        self._thread.start()
        logger.info("Engine loop started.")

    def shutdown(self):
        """Stop the engine loop and wait for it to finish."""
        self._running = False
        self._request_queue.put(None)  # wake up the loop
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Engine shut down.")

    # ── Request API ───────────────────────────────────────────────────────────

    def submit_with_callback(
        self,
        prompt_ids: List[int],
        sampling: SamplingParams,
        request_id: Optional[str] = None,
        stream_callback: Optional[Callable[[int, bool], None]] = None,
    ) -> str:
        """Submit a request asynchronously. Tokens arrive via stream_callback."""
        rid = request_id or str(uuid.uuid4())
        req = Request(
            request_id=rid,
            prompt_token_ids=prompt_ids,
            sampling=sampling,
            stream_callback=stream_callback,
        )
        self._request_queue.put(req)
        return rid

    def generate(self, prompt: str, params: SamplingParams) -> RequestOutput:
        """Synchronous generation. Blocks until generation is complete."""
        prompt_ids = self._tokenizer.encode(prompt)
        resp_q: queue.Queue = queue.Queue()

        collected_tokens: List[int] = []
        t_start = time.perf_counter()
        t_first: Optional[float] = None

        def on_token(token_id: int, is_final: bool):
            nonlocal t_first
            if t_first is None:
                t_first = time.perf_counter()
            collected_tokens.append(token_id)
            if is_final:
                resp_q.put(True)

        rid = self.submit_with_callback(prompt_ids, params, stream_callback=on_token)

        # Block until done
        if not self._running:
            # Run synchronously if engine loop not started
            self._run_synchronous(prompt_ids, params, on_token)
        resp_q.get(timeout=300)  # 5 min timeout

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000
        ttft_ms = ((t_first or t_end) - t_start) * 1000
        n = len(collected_tokens)

        text = self._tokenizer.decode(collected_tokens)
        return RequestOutput(
            request_id=rid,
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            outputs=[CompletionOutput(
                index=0, text=text, token_ids=collected_tokens,
                finish_reason=FinishReason.MAX_TOKENS,
            )],
            finished=True,
            prompt_tokens=len(prompt_ids),
            generated_tokens=n,
            latency_ms=total_ms,
            ttft_ms=ttft_ms,
            tpot_ms=(total_ms - ttft_ms) / n if n > 1 else 0.0,
        )

    def generate_stream(
        self, prompt: str, params: SamplingParams
    ) -> Iterator[str]:
        """Synchronous streaming generator. Yields decoded text pieces."""
        prompt_ids = self._tokenizer.encode(prompt)
        token_q: queue.Queue = queue.Queue()

        def on_token(token_id: int, is_final: bool):
            token_q.put((token_id, is_final))

        self.submit_with_callback(prompt_ids, params, stream_callback=on_token)

        while True:
            token_id, is_final = token_q.get()
            yield self._tokenizer.decode_token(token_id)
            if is_final:
                break

    # ── Engine loop ───────────────────────────────────────────────────────────

    def _engine_loop(self):
        """Main inference loop running on background thread."""
        while self._running:
            # Drain incoming requests
            self._drain_incoming()

            if not self._scheduler.has_work():
                time.sleep(0.001)
                continue

            self._step()

    def _drain_incoming(self):
        """Move all pending requests from queue to scheduler."""
        try:
            while True:
                item = self._request_queue.get_nowait()
                if item is None:
                    self._running = False
                    return
                self._scheduler.add_request(item)
        except queue.Empty:
            pass

    def _step(self):
        """Execute one iteration: schedule → forward → sample → update."""
        with nvtx_range("scheduling"):
            batch = self._scheduler.schedule()

        if batch.is_empty:
            return

        # Forward pass
        new_tokens: Dict[str, int] = {}

        if batch.prefill_requests:
            with nvtx_range("prefill"):
                new_tokens.update(self._run_prefill(batch.prefill_requests))

        if batch.decode_requests:
            with nvtx_range("decode"):
                new_tokens.update(self._run_decode(batch.decode_requests))

        # Update scheduler state
        self._scheduler.update(batch, new_tokens)

    # ── Forward pass (PyTorch backend) ────────────────────────────────────────

    @torch.no_grad()
    def _run_prefill(self, requests: List[Request]) -> Dict[str, int]:
        """Run prefill for a batch of requests. Returns {req_id: first_token}."""
        results = {}

        for req in requests:
            input_ids = torch.tensor(
                req.prompt_token_ids, dtype=torch.long, device="cuda"
            ).unsqueeze(0)  # [1, prompt_len]

            with nvtx_range("attention"):
                out = self._backend(input_ids=input_ids, use_cache=True)

            logit = out.logits[0, -1, :]  # last token logit
            token = self._sampler._sample_one(logit, req.sampling).item()
            results[req.request_id] = token

        return results

    @torch.no_grad()
    def _run_decode(self, requests: List[Request]) -> Dict[str, int]:
        """Run decode for one new token per request."""
        results = {}

        for req in requests:
            last_token = req.output_token_ids[-1] if req.output_token_ids else req.prompt_token_ids[-1]
            input_ids = torch.tensor(
                [[last_token]], dtype=torch.long, device="cuda"
            )

            with nvtx_range("attention"):
                out = self._backend(input_ids=input_ids, use_cache=True)

            logit = out.logits[0, -1, :]
            token = self._sampler._sample_one(logit, req.sampling).item()
            results[req.request_id] = token

        return results

    def _run_synchronous(
        self,
        prompt_ids: List[int],
        params: SamplingParams,
        callback: Callable[[int, bool], None],
    ):
        """Run generation without the background thread (single-request path)."""
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device="cuda").unsqueeze(0)

        with torch.no_grad():
            out = self._backend(input_ids=input_ids, use_cache=True)

        token_ids: List[int] = []
        past_kv = out.past_key_values

        for step in range(params.max_tokens):
            logit = out.logits[0, -1, :]
            token = self._sampler._sample_one(logit, params).item()
            token_ids.append(token)

            is_final = (
                step == params.max_tokens - 1
                or token in params.stop_token_ids
            )
            callback(token, is_final)
            if is_final:
                break

            # Next step
            next_ids = torch.tensor([[token]], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = self._backend(
                    input_ids=next_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            past_kv = out.past_key_values
