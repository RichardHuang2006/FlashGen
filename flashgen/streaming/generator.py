"""
Streaming token generation — real-time token-by-token output.

Why streaming matters for LLM UX:
  Without streaming, the user waits the full generation latency before seeing
  any output (Time-To-Last-Token). With streaming, the user sees the first
  token after TTFT (Time-To-First-Token) and subsequent tokens every TPOT ms.

Implementation:
  The engine backend emits tokens via a callback (C++ streaming callback or
  Python asyncio.Queue). This module provides:
    - StreamingGenerator: synchronous generator (for REPL / CLI use)
    - AsyncStreamingGenerator: asyncio generator (for FastAPI / web servers)

Token decoding strategy:
  Most tokenizers use byte-pair encoding. Single tokens may decode to partial
  words or include leading spaces. We use the tokenizer's incremental decode
  to reconstruct the full text piece by piece, yielding each decoded piece.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import AsyncGenerator, Callable, Generator, Iterator, List, Optional

from flashgen.core.config import SamplingParams
from flashgen.outputs import CompletionOutput, FinishReason, RequestOutput


_SENTINEL = object()  # signals end of stream


class StreamingGenerator:
    """Synchronous streaming generator for CLI and REPL use.

    Usage
    -----
    >>> gen = StreamingGenerator(engine, tokenizer)
    >>> for token_text in gen.generate("Hello, my name is", params):
    ...     print(token_text, end="", flush=True)
    """

    def __init__(self, engine, tokenizer):
        """
        Parameters
        ----------
        engine     : LLMEngine instance (has .generate_stream method)
        tokenizer  : Tokenizer instance (has .decode_token method)
        """
        self.engine = engine
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Yield decoded text pieces as each token is generated.

        Yields the token text immediately after generation, enabling
        real-time display. The full generated text is the concatenation of
        all yielded pieces.
        """
        token_q: queue.Queue = queue.Queue()
        prompt_ids = self.tokenizer.encode(prompt)

        def on_token(token_id: int, is_final: bool):
            token_q.put((token_id, is_final))

        # Submit asynchronously with streaming callback
        self.engine.submit_with_callback(
            prompt_ids=prompt_ids,
            sampling=params,
            request_id=request_id,
            stream_callback=on_token,
        )

        # Yield tokens as they arrive
        while True:
            token_id, is_final = token_q.get()
            text = self.tokenizer.decode_token(token_id)
            yield text
            if is_final:
                break

    def generate_with_stats(
        self,
        prompt: str,
        params: SamplingParams,
    ) -> RequestOutput:
        """Generate synchronously and return a RequestOutput with timing stats."""
        t_start = time.perf_counter()
        t_first_token: Optional[float] = None
        token_ids: List[int] = []
        texts: List[str] = []

        for piece in self.generate(prompt, params):
            if t_first_token is None:
                t_first_token = time.perf_counter()
            texts.append(piece)
            token_ids.append(self.tokenizer.encode(piece)[-1] if piece else -1)

        t_end = time.perf_counter()
        prompt_ids = self.tokenizer.encode(prompt)
        total_ms = (t_end - t_start) * 1000
        ttft_ms = ((t_first_token or t_end) - t_start) * 1000
        n_out = len(token_ids)
        tpot_ms = (total_ms - ttft_ms) / n_out if n_out > 1 else 0.0

        return RequestOutput(
            request_id="",
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            outputs=[CompletionOutput(
                index=0,
                text="".join(texts),
                token_ids=token_ids,
                finish_reason=FinishReason.MAX_TOKENS,
            )],
            finished=True,
            prompt_tokens=len(prompt_ids),
            generated_tokens=n_out,
            latency_ms=total_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
        )


class AsyncStreamingGenerator:
    """Asyncio-native streaming generator for FastAPI / web server use.

    Usage
    -----
    >>> gen = AsyncStreamingGenerator(engine, tokenizer)
    >>> async for piece in gen.generate("Tell me about GPU memory", params):
    ...     await websocket.send_text(piece)
    """

    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer

    async def generate(
        self,
        prompt: str,
        params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Async generator yielding decoded token text pieces."""
        loop = asyncio.get_event_loop()
        token_q: asyncio.Queue = asyncio.Queue()
        prompt_ids = self.tokenizer.encode(prompt)

        def on_token(token_id: int, is_final: bool):
            # Called from engine thread — schedule into event loop
            loop.call_soon_threadsafe(token_q.put_nowait, (token_id, is_final))

        self.engine.submit_with_callback(
            prompt_ids=prompt_ids,
            sampling=params,
            request_id=request_id,
            stream_callback=on_token,
        )

        while True:
            token_id, is_final = await token_q.get()
            yield self.tokenizer.decode_token(token_id)
            if is_final:
                break

    async def generate_sse(
        self,
        prompt: str,
        params: SamplingParams,
    ) -> AsyncGenerator[str, None]:
        """Yield Server-Sent Events (SSE) formatted strings for HTTP streaming."""
        async for piece in self.generate(prompt, params):
            # SSE format: "data: <content>\n\n"
            yield f"data: {piece}\n\n"
        yield "data: [DONE]\n\n"
