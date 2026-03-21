"""
Lightweight profiling hooks for AsyncLLMEngine.

Optional NVTX integration was removed to keep the core package minimal.
`nvtx_range` is a no-op context manager; `NsightProfiler` provides empty stubs.
"""

from __future__ import annotations

import contextlib
from typing import Iterator


@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """No-op NVTX-style range (keeps engine.py call sites unchanged)."""
    del name
    yield


class NsightProfiler:
    """Stub profiler — start/stop/profile are no-ops."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @contextlib.contextmanager
    def profile(self) -> Iterator[None]:
        yield
