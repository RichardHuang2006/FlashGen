"""
LLM — synchronous high-level API for simple generation use cases.

This is the user-facing entry point analogous to vLLM's LLM class.
It wraps AsyncLLMEngine with a simpler synchronous interface.

Example
-------
    from flashgen import LLM, SamplingParams

    llm = LLM("gpt2")                              # loads model
    params = SamplingParams(temperature=0.7, max_tokens=100)

    # Single prompt
    result = llm.generate("The meaning of life is", params)
    print(result.text)

    # Batch of prompts
    results = llm.generate_batch(["Hello", "World"], params)

    # Streaming
    for piece in llm.stream("Once upon a time", params):
        print(piece, end="", flush=True)

    # ONNX export
    llm.export_onnx("model.onnx")

    # TRT engine build
    llm.build_trt_engine("model.onnx", "model.trt", precision="fp16")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

from flashgen.core.config import EngineConfig, ModelConfig, Precision, SamplingParams
from flashgen.engine import AsyncLLMEngine
from flashgen.outputs import RequestOutput

logger = logging.getLogger(__name__)


class LLM:
    """Synchronous LLM interface for generation, export, and benchmarking.

    Parameters
    ----------
    model : str
        HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
        or local path containing model files.
    precision : str
        "fp16" (default), "fp32", or "int8" (requires TRT backend).
    device : int
        CUDA device ID (0-based). Default: 0.
    backend : str
        "pytorch" (default, no build required), "cuda" (custom kernels),
        or "trt" (TensorRT, requires trt_engine_path).
    trt_engine_path : str | None
        Path to a pre-built .trt engine file. Required when backend="trt".
    gpu_memory_utilization : float
        Fraction of GPU memory to use for KV cache (default: 0.9).
    max_num_seqs : int
        Maximum concurrent sequences in the engine (default: 256).
    block_size : int
        KV cache block size in tokens (default: 16).
    """

    def __init__(
        self,
        model: str = "gpt2",
        precision: str = "fp16",
        device: int = 0,
        backend: str = "pytorch",
        trt_engine_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 256,
        block_size: int = 16,
    ):
        self._model_id = model
        prec = Precision(precision)

        cfg = EngineConfig(
            hf_model_id=model,
            precision=prec,
            device_id=device,
            backend=backend,
            trt_engine_path=trt_engine_path,
        )
        cfg.cache.gpu_memory_utilization = gpu_memory_utilization
        cfg.cache.block_size = block_size
        cfg.scheduler.max_num_seqs = max_num_seqs

        self._engine = AsyncLLMEngine.from_config(cfg)

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, params: Optional[SamplingParams] = None) -> RequestOutput:
        """Generate text for a single prompt (synchronous).

        Returns a RequestOutput containing the generated text, token IDs,
        and latency statistics.
        """
        if params is None:
            params = SamplingParams()
        return self._engine.generate(prompt, params)

    def generate_batch(
        self,
        prompts: List[str],
        params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """Generate text for a batch of prompts.

        Internally uses continuous batching: all prompts are submitted
        simultaneously and processed in interleaved iterations, maximizing
        GPU utilization.
        """
        if params is None:
            params = SamplingParams()

        # Start engine loop for concurrent processing
        self._engine.start()
        try:
            import queue as q
            result_map: dict = {}
            pending = set()

            for prompt in prompts:
                import queue as _q
                resp_q: _q.Queue = _q.Queue()
                tokens: List[int] = []

                def make_cb(rq, tl):
                    def cb(token_id, is_final):
                        tl.append(token_id)
                        if is_final:
                            rq.put(True)
                    return cb

                rid = self._engine.submit_with_callback(
                    prompt_ids=self._engine._tokenizer.encode(prompt),
                    sampling=params,
                    stream_callback=make_cb(resp_q, tokens),
                )
                result_map[rid] = (prompt, resp_q, tokens)
                pending.add(rid)

            outputs = []
            for rid, (prompt, resp_q, tokens) in result_map.items():
                resp_q.get(timeout=300)
                text = self._engine._tokenizer.decode(tokens)
                from flashgen.outputs import CompletionOutput, FinishReason
                outputs.append(RequestOutput(
                    request_id=rid,
                    prompt=prompt,
                    prompt_token_ids=self._engine._tokenizer.encode(prompt),
                    outputs=[CompletionOutput(
                        index=0, text=text, token_ids=tokens,
                        finish_reason=FinishReason.MAX_TOKENS
                    )],
                    finished=True,
                    generated_tokens=len(tokens),
                ))
        finally:
            self._engine.shutdown()

        return outputs

    def stream(self, prompt: str, params: Optional[SamplingParams] = None) -> Iterator[str]:
        """Stream generated text token by token.

        Yields decoded text pieces as each token is generated.
        Enables real-time display for interactive applications.
        """
        if params is None:
            params = SamplingParams()
        yield from self._engine.generate_stream(prompt, params)

    # ── Export ────────────────────────────────────────────────────────────────

    def export_onnx(self, output_path: str, opset: int = 17) -> Path:
        """Export the model to ONNX with KV cache as explicit I/O.

        The exported graph takes past_key/past_value tensors as inputs and
        outputs present_key/present_value, enabling efficient incremental
        decoding in TensorRT.
        """
        from flashgen.onnx_export.exporter import ONNXExporter
        exporter = ONNXExporter(self._model_id, opset=opset)
        return exporter.export(output_path)

    def build_trt_engine(
        self,
        onnx_path: str,
        output_path: str,
        precision: str = "fp16",
        max_batch: int = 32,
        max_seq: int = 2048,
        calibration_data: Optional[List[List[int]]] = None,
    ) -> None:
        """Build a TensorRT engine from an ONNX file.

        Parameters
        ----------
        onnx_path   : Path to the ONNX file (from export_onnx).
        output_path : Where to save the .trt engine.
        precision   : "fp16" or "int8".
        max_batch   : Maximum batch size to optimize for.
        max_seq     : Maximum sequence length to optimize for.
        calibration_data : Required for INT8 — list of tokenized prompts.
        """
        from flashgen.trt_builder.builder import TRTEngineBuilder
        cfg = self._engine._model_config
        builder = TRTEngineBuilder(
            precision=precision,
            max_batch=max_batch,
            max_seq=max_seq,
            calibration_data=calibration_data,
            n_layers=cfg.n_layers,
            n_kv_heads=cfg.actual_kv_heads,
            head_dim=cfg.head_dim,
        )
        builder.build(onnx_path, output_path)
        logger.info(f"TRT engine saved to {output_path}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def tokenizer(self):
        return self._engine._tokenizer

    @property
    def model_config(self) -> ModelConfig:
        return self._engine._model_config
