"""Checkpoint resolution and process-level singleton caching.

Torch-free until ``get_model()`` is called: the vendored factory import and
torch happen inside the loader so web-server startup stays light. The cache
is keyed by (checkpoint sha256 prefix, device) so a replaced checkpoint is
picked up automatically across jobs.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from video_upscaler.models import MATANYONE_CKPT_NAME

_lock = threading.Lock()
_cache: dict[tuple[str, str], object] = {}


class MatAnyoneModelMissing(RuntimeError):
    """The MatAnyone2 checkpoint is not installed."""


def checkpoint_path() -> Path:
    """Return the expected checkpoint location, raising if absent."""
    from video_upscaler import config

    path = config.MODELS_DIR / "matanyone" / MATANYONE_CKPT_NAME
    if not path.is_file():
        raise MatAnyoneModelMissing(
            f"MatAnyone2 checkpoint not found:\n{path}\n\n"
            "Install it with:\n"
            "  uv run main.py --download-models all\n"
            "or place matanyone2.pth at the path above."
        )
    return path


def _ckpt_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _load_vendor_model(ckpt_path: str, device: str):
    from .vendor.matanyone2.utils.get_default_model import get_matanyone2_model

    return get_matanyone2_model(ckpt_path, device)


def get_model(device: str | None = None):
    """Return the cached MatAnyone2 model for (fingerprint, device)."""
    import torch

    ckpt = checkpoint_path()
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (_ckpt_fingerprint(ckpt), resolved)
    with _lock:
        instance = _cache.get(key)
        if instance is None:
            instance = _load_vendor_model(str(ckpt), resolved)
            _cache[key] = instance
    return instance


def release_model(device: str | None = None) -> None:
    """Drop cached model(s); call between jobs or in tests."""
    with _lock:
        if device is None:
            _cache.clear()
        else:
            for key in [k for k in _cache if k[1] == device]:
                del _cache[key]