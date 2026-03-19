"""
NVTX range markers for Nsight Systems / Nsight Compute profiling.

Usage in code:
    with nvtx_range("prefill_forward"):
        model.batch_prefill(...)

Or as a decorator:
    @nvtx_range("attention_kernel")
    def run_attention(...): ...

In Nsight Systems:
    nsys profile --trace cuda,nvtx python benchmark.py
    nsys-ui report.nsys-rep      # visual timeline

In Nsight Compute:
    ncu --set full python -c "import flashgen; ..."

NVTX categories used in FlashGen:
    "prefill"           — prompt processing forward pass
    "decode"            — decode step forward pass
    "attention"         — attention kernel (flash or paged)
    "kv_cache_write"    — writing K,V to cache
    "sampling"          — token sampling
    "scheduling"        — scheduler.schedule() call
    "memory_alloc"      — block allocation
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import functools
import os
from typing import Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable)


# ── NVTX via ctypes (no pynvtx dependency) ───────────────────────────────────

class _NVTXLib:
    """Thin ctypes wrapper around libnvToolsExt.so / nvToolsExt64_1.dll."""

    def __init__(self):
        self._lib = None
        self._push = None
        self._pop = None
        self._enabled = False
        self._try_load()

    def _try_load(self):
        lib_name = ctypes.util.find_library("nvToolsExt")
        if lib_name is None:
            # Try common paths
            for candidate in [
                "libnvToolsExt.so.1",
                "/usr/local/cuda/lib64/libnvToolsExt.so.1",
                "/usr/lib/x86_64-linux-gnu/libnvToolsExt.so.1",
            ]:
                try:
                    self._lib = ctypes.CDLL(candidate)
                    break
                except OSError:
                    continue
        else:
            try:
                self._lib = ctypes.CDLL(lib_name)
            except OSError:
                pass

        if self._lib is not None:
            try:
                self._push = self._lib.nvtxRangePushA
                self._push.argtypes = [ctypes.c_char_p]
                self._push.restype = ctypes.c_int
                self._pop = self._lib.nvtxRangePop
                self._pop.argtypes = []
                self._pop.restype = ctypes.c_int
                self._enabled = True
            except AttributeError:
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def push(self, name: str):
        if self._enabled and self._push:
            self._push(name.encode("utf-8"))

    def pop(self):
        if self._enabled and self._pop:
            self._pop()


_nvtx = _NVTXLib()


@contextlib.contextmanager
def nvtx_range(name: str):
    """Context manager that wraps code in an NVTX range for profiling."""
    _nvtx.push(name)
    try:
        yield
    finally:
        _nvtx.pop()


def nvtx_annotate(name: Optional[str] = None):
    """Decorator that wraps a function in an NVTX range."""
    def decorator(fn: F) -> F:
        range_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _nvtx.push(range_name)
            try:
                return fn(*args, **kwargs)
            finally:
                _nvtx.pop()

        return wrapper  # type: ignore[return-value]
    return decorator


class NsightProfiler:
    """Manages Nsight Systems profiling sessions via CUDA API.

    Start/stop profiling programmatically so that benchmark warmup
    is excluded from the profile trace.

    Usage:
        profiler = NsightProfiler()
        profiler.start()         # begins capturing
        run_benchmark()
        profiler.stop()          # stops capturing
    """

    def __init__(self):
        self._cudart = None
        self._try_load()

    def _try_load(self):
        try:
            import ctypes
            self._cudart = ctypes.CDLL("libcudart.so")
        except OSError:
            pass

    def start(self):
        """Enable CUDA profiler (equivalent to cudaProfilerStart)."""
        if self._cudart:
            self._cudart.cudaProfilerStart()

    def stop(self):
        """Disable CUDA profiler (equivalent to cudaProfilerStop)."""
        if self._cudart:
            self._cudart.cudaProfilerStop()

    @contextlib.contextmanager
    def profile(self):
        """Context manager for a profiling section."""
        self.start()
        try:
            yield
        finally:
            self.stop()
