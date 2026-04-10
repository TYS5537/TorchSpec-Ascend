# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Hardware-agnostic device abstraction for CUDA and Ascend NPU.

Set ``TORCHSPEC_DEVICE_TYPE=npu`` to force NPU backend.
Without the env var the module auto-detects: if ``torch_npu`` is importable
and an NPU device is visible it picks NPU, otherwise falls back to CUDA.

Usage::

    from torchspec.utils import accelerator as accel

    accel.set_device(local_id)
    stream = accel.Stream(device=accel.current_device_obj())
    event  = accel.Event(enable_timing=True)
    with accel.stream(stream):
        ...
"""

import os
from contextlib import contextmanager
from typing import Union

import torch

# ---------------------------------------------------------------------------
# Backend detection (runs once at import time)
# ---------------------------------------------------------------------------


def _detect_device_type() -> str:
    explicit = os.environ.get("TORCHSPEC_DEVICE_TYPE", "").lower()
    if explicit in ("npu", "cuda"):
        return explicit
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu"
    except ImportError:
        pass
    return "cuda"


_DEVICE_TYPE: str = _detect_device_type()


# ---------------------------------------------------------------------------
# Backend queries
# ---------------------------------------------------------------------------


def get_device_type() -> str:
    """Return the active accelerator type: ``'cuda'`` or ``'npu'``."""
    return _DEVICE_TYPE


def is_npu() -> bool:
    return _DEVICE_TYPE == "npu"


def is_cuda() -> bool:
    return _DEVICE_TYPE == "cuda"


# ---------------------------------------------------------------------------
# Core device API
# ---------------------------------------------------------------------------


def is_available() -> bool:
    if is_npu():
        return torch.npu.is_available()
    return torch.cuda.is_available()


def is_initialized() -> bool:
    if is_npu():
        return torch.npu.is_initialized()
    return torch.cuda.is_initialized()


def current_device() -> int:
    """Return the current device index as an integer."""
    if is_npu():
        return torch.npu.current_device()
    return torch.cuda.current_device()


def current_device_obj() -> torch.device:
    """Return the current device as a :class:`torch.device`."""
    return torch.device(f"{_DEVICE_TYPE}:{current_device()}")


def set_device(device: Union[int, torch.device]) -> None:
    if is_npu():
        torch.npu.set_device(device)
    else:
        torch.cuda.set_device(device)


def synchronize(device=None) -> None:
    if is_npu():
        torch.npu.synchronize(device)
    else:
        torch.cuda.synchronize(device)


def empty_cache() -> None:
    if is_npu():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Device capability
# ---------------------------------------------------------------------------


def get_device_capability(device=None) -> tuple[int, int]:
    """Return ``(major, minor)`` compute capability.

    Ascend NPUs do not use SM capability; this returns ``(9, 0)`` as a
    sentinel meaning "Hopper-equivalent or better" so feature gates that
    check ``sm_major >= N`` remain enabled on NPU.
    """
    if is_npu():
        return (9, 0)
    return torch.cuda.get_device_capability(device)


# ---------------------------------------------------------------------------
# Memory info
# ---------------------------------------------------------------------------


def mem_get_info(device=None) -> tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` for *device*."""
    if is_npu():
        return torch.npu.mem_get_info(device)
    return torch.cuda.mem_get_info(device)


def memory_allocated(device=None) -> int:
    if is_npu():
        return torch.npu.memory_allocated(device)
    return torch.cuda.memory_allocated(device)


def memory_reserved(device=None) -> int:
    if is_npu():
        return torch.npu.memory_reserved(device)
    return torch.cuda.memory_reserved(device)


def max_memory_allocated(device=None) -> int:
    if is_npu():
        return torch.npu.max_memory_allocated(device)
    return torch.cuda.max_memory_allocated(device)


def reset_peak_memory_stats(device=None) -> None:
    if is_npu():
        torch.npu.reset_peak_memory_stats(device)
    else:
        torch.cuda.reset_peak_memory_stats(device)


def get_device_name(device=None) -> str:
    if is_npu():
        return getattr(torch.npu, "get_device_name", lambda d=None: "Ascend NPU")(device)
    return torch.cuda.get_device_name(device)


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------


def get_rng_state_all():
    if is_npu():
        return torch.npu.get_rng_state_all()
    return torch.cuda.get_rng_state_all()


def set_rng_state_all(state) -> None:
    if is_npu():
        torch.npu.set_rng_state_all(state)
    else:
        torch.cuda.set_rng_state_all(state)


# ---------------------------------------------------------------------------
# Stream & Event wrappers
# ---------------------------------------------------------------------------


class Stream:
    """Backend-agnostic stream.

    Wraps ``torch.cuda.Stream`` or ``torch.npu.Stream`` transparently.
    Unknown attributes are delegated to the inner stream object.
    """

    def __init__(self, device=None, **kwargs):
        if is_npu():
            self._inner = torch.npu.Stream(device=device, **kwargs)
        else:
            self._inner = torch.cuda.Stream(device=device, **kwargs)

    @property
    def device(self) -> torch.device:
        return self._inner.device

    def wait_event(self, event: "Event") -> None:
        inner_event = event._inner if isinstance(event, Event) else event
        self._inner.wait_event(inner_event)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class Event:
    """Backend-agnostic event.

    Wraps ``torch.cuda.Event`` or ``torch.npu.Event`` transparently.
    Unknown attributes are delegated to the inner event object.
    """

    def __init__(self, enable_timing: bool = False, **kwargs):
        if is_npu():
            self._inner = torch.npu.Event(enable_timing=enable_timing, **kwargs)
        else:
            self._inner = torch.cuda.Event(enable_timing=enable_timing, **kwargs)

    def record(self, stream: "Stream | None" = None) -> None:
        if stream is None:
            self._inner.record()
        else:
            inner_stream = stream._inner if isinstance(stream, Stream) else stream
            self._inner.record(inner_stream)

    def elapsed_time(self, end_event: "Event") -> float:
        inner_end = end_event._inner if isinstance(end_event, Event) else end_event
        return self._inner.elapsed_time(inner_end)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


@contextmanager
def stream(s: "Stream"):
    """Context manager that activates *s* as the current stream."""
    inner = s._inner if isinstance(s, Stream) else s
    if is_npu():
        with torch.npu.stream(inner):
            yield
    else:
        with torch.cuda.stream(inner):
            yield


def record_stream(tensor: torch.Tensor, s: "Stream") -> None:
    """Record *tensor* on *s*, unwrapping our :class:`Stream` wrapper."""
    inner = s._inner if isinstance(s, Stream) else s
    tensor.record_stream(inner)


# ---------------------------------------------------------------------------
# Tensor device check helper
# ---------------------------------------------------------------------------


def is_on_accelerator(tensor: torch.Tensor) -> bool:
    """Return True if *tensor* lives on any accelerator (CUDA or NPU)."""
    return tensor.device.type in ("cuda", "npu")
