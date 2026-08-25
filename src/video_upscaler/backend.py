"""Backend detection: tensorrt / torch-cuda / torch-mps / ncnn Vulkan / torch-cpu.

Priority (auto): NVIDIA TensorRT -> NVIDIA CUDA -> Apple Metal (MPS) ->
ncnn Vulkan (Intel/AMD/any Vulkan GPU) -> torch CPU.
CLARITY_BACKEND=torch|ncnn|tensorrt forces a family (read dynamically so
the CLI's --backend flag can override it at runtime). Heavy imports are
lazy so tests stay torch-free.
"""

from __future__ import annotations

import importlib

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
