"""Backend detection: tensorrt / torch-cuda / torch-mps / ncnn Vulkan / torch-cpu.

Priority (auto): NVIDIA TensorRT -> NVIDIA CUDA -> Apple Metal (MPS) ->
ncnn Vulkan (Intel/AMD/any Vulkan GPU) -> torch CPU.
CLARITY_BACKEND=torch|ncnn|tensorrt forces a family (read dynamically so
the CLI's --backend flag can override it at runtime). Heavy imports are
lazy so tests stay torch-free.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any

from video_upscaler import config

_LABELS = {
    "tensorrt": "TensorRT (fp16)",
    "torch-cuda": "CUDA (torch fp16)",
    "torch-mps": "Metal (MPS)",
    "ncnn": "ncnn Vulkan",
    "torch-cpu": "CPU (torch)",
}


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _torch_mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    is_available = getattr(mps, "is_available", None)
    return bool(is_available is not None and is_available())


def _tensorrt_available() -> bool:
    try:
        from video_upscaler import tensorrt_backend
    except ImportError:
        return False
    return tensorrt_backend.tensorrt_available()


def _ncnn_tool_present() -> bool:
    try:
        ncnn = importlib.import_module("video_upscaler.ncnn")
    except ImportError:
        return False
    return ncnn.ncnn_exe("realcugan").is_file()


def detect_backend() -> str:
    """Return the active backend: tensorrt | torch-cuda | torch-mps | ncnn | torch-cpu."""
    preferred = config.BACKEND_PREF
    if preferred == "ncnn":
        return "ncnn"
    if preferred == "torch":
        if _torch_cuda_available():
            return "torch-cuda"
        if _torch_mps_available():
            return "torch-mps"
        return "torch-cpu"
    if preferred == "tensorrt":
        return "tensorrt" if _tensorrt_available() else "torch-cpu"
    # auto
    if _torch_cuda_available():
        return "tensorrt" if _tensorrt_available() else "torch-cuda"
    if _torch_mps_available():
        return "torch-mps"
    if _ncnn_tool_present():
        return "ncnn"
    return "torch-cpu"


def backend_label(backend: str) -> str:
    """Human-readable label for a backend key."""
    return _LABELS.get(backend, backend)


_detection_lock = threading.Lock()
_cached_detection: dict[str, str] | None = None
_priming_started = False


def cached_detection() -> dict[str, str] | None:
    """Return the cached backend/device detection dict or None if cold."""
    return _cached_detection


def prime_detection() -> dict[str, str]:
    """Compute and cache backend + device detection once (thread-safe, idempotent).

    Falls back to 'torch-cpu' / 'cpu' if detection fails.
    """
    global _cached_detection, _priming_started
    with _detection_lock:
        if _cached_detection is not None:
            return _cached_detection

        _priming_started = True

        try:
            b = detect_backend()
        except Exception:
            b = "torch-cpu"

        try:
            from video_upscaler.cugan import detect_device

            d = detect_device()
        except Exception:
            d = "cpu"

        _cached_detection = {
            "backend": b,
            "backend_label": backend_label(b),
            "device": d,
        }
        return _cached_detection


def ensure_prime_detection() -> None:
    """Kick off prime_detection in a background thread if not already running."""
    global _priming_started
    if _cached_detection is not None or _priming_started:
        return
    with _detection_lock:
        if _cached_detection is not None or _priming_started:
            return
        _priming_started = True
        threading.Thread(
            target=prime_detection,
            name="clarity-prime-detection",
            daemon=True,
        ).start()


def _reset_detection_cache() -> None:
    """Reset cached detection (intended for tests)."""
    global _cached_detection, _priming_started
    with _detection_lock:
        _cached_detection = None
        _priming_started = False

