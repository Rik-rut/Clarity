"""One-time fp32 -> fp16 weight cache (pre-quantized on disk).

Mirrors Auto-Broll's "no runtime quantization" pattern: the expensive
dtype conversion happens once, offline, and every run loads the already-
fp16 file (half the disk read, zero cast time at load). CPU runs keep
using the original fp32 files.
"""

from __future__ import annotations

import os
from pathlib import Path

from video_upscaler.config import MODELS_DIR


def fp16_path(model_name: str) -> Path:
    """Cache path for a model filename (models/<name>.fp16.pth)."""
    return MODELS_DIR / f"{model_name.removesuffix('.pth')}.fp16.pth"


def is_fresh(cache: Path, source: Path) -> bool:
    """Return True when the cache exists and is not older than its source."""
    return cache.is_file() and cache.stat().st_mtime_ns >= source.stat().st_mtime_ns


def _cast_half(payload):
    """Recursively cast float32 tensors to float16; keep other values."""
    if hasattr(payload, "dtype") and str(payload.dtype) == "torch.float32":
        return payload.half()
    if isinstance(payload, dict):
        return {key: _cast_half(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return type(payload)(_cast_half(item) for item in payload)
    return payload


def convert_to_fp16(source: Path, dest: Path) -> None:
    """Load fp32 weights from ``source`` and save them fp16 to ``dest``.

    Writes atomically (tmp file + os.replace) so a killed conversion never
    leaves a half-written cache behind.
    """
    import torch

    weights = torch.load(source, map_location="cpu")
    weights = _cast_half(weights)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        torch.save(weights, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def ensure_fp16(model_name: str) -> Path:
    """Return the fp16 cache path, converting once if missing or stale."""
    source = MODELS_DIR / model_name
    cache = fp16_path(model_name)
    if is_fresh(cache, source):
        return cache
    print(f"Converting {model_name} to fp16 (first run only)...")
    convert_to_fp16(source, cache)
    return cache
