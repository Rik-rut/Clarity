"""Real-CUGAN engine (bilibili/ailab): anime super-resolution with CUDA.

Real-CUGAN is a U-Net-based anime upscaler trained on a million-scale anime
patch dataset. Unlike Anime4KCPP's ACNet it is a heavier, higher-quality
model; on an RTX 3050 expect ~1-3 s per 1080p frame (fp16, tiled).

Weights are the official v3 checkpoints, fetched from the Clarity model hub
(manifest-driven; see ``modelhub.py``) into the project-local ``models/``
directory. On first anime run the CLI offers to download them (explicit,
integrity-checked) so other users of this project get them automatically.

The architecture is vendored in ``upcunet_v3.py`` (MIT, unmodified except
that invalid tile modes raise instead of calling os._exit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from video_upscaler.config import CUGAN_TILE_MODE, MODELS_DIR, PREFERRED_DEVICE
from video_upscaler.models import scale_for_model


def scale_from_model(model_name: str) -> int:
    """Derive the upscale factor from a model filename (deprecated wrapper)."""
    return scale_for_model(model_name)


def _model_path(model_name: str) -> Path:
    """Project-local path for a model filename (models/<name>)."""
    return MODELS_DIR / model_name


def check_cugan(model_name: str) -> str | None:
    """Return None if the model exists, else a clear error message."""
    model_path = _model_path(model_name)
    if model_path.is_file():
        return None
    return (
        "Required Real-CUGAN model not found:\n\n"
        f"{model_path}\n\n"
        "Run the app again and answer [Y] to download it automatically, or\n"
        "run once with --download-models essential|all.\n"
    )


def download_cugan_models() -> None:
    """Download all Real-CUGAN weights from the Clarity model hub.

    Raises HubError (RuntimeError) with an actionable message on failure.
    """
    from video_upscaler import modelhub

    modelhub.install_tier(group="cugan")


def resolve_cache_mode() -> int:
    """Return the configured Real-CUGAN cache mode (torch-free helper)."""
    from video_upscaler.config import CUGAN_CACHE_MODE

    return CUGAN_CACHE_MODE


def detect_device() -> str:
    """Return the processing device ("cuda", "mps", or "cpu") without loading a model."""
    import torch

    preferred = PREFERRED_DEVICE
    if preferred == "cuda":
        return "cuda"
    if preferred == "mps":
        return "mps"
    if preferred == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def _mps_available(torch) -> bool:
    """True when Apple Metal (MPS) GPU acceleration is usable."""
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    is_available = getattr(mps, "is_available", None)
    return bool(is_available is not None and is_available())


class RealCUGANEngine:
    """Real-CUGAN upscaling engine for one model (in-process, torch)."""

    def __init__(self, model_name: str) -> None:
        model_path = _model_path(model_name)
        missing = check_cugan(model_name)
        if missing:
            raise FileNotFoundError(missing)

        import torch

        if PREFERRED_DEVICE == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested via CLARITY_DEVICE but is not available "
                "on this system."
            )
        if PREFERRED_DEVICE == "mps" and not _mps_available(torch):
            raise RuntimeError(
                "MPS was requested via CLARITY_DEVICE but is not available "
                "on this system."
            )
        device = detect_device()
        if device == "cuda":
            print("Device: NVIDIA CUDA")
        elif device == "mps":
            print("Device: Apple Silicon (MPS)")
        else:
            print("Device: CPU")
            print("No GPU acceleration is available.")
            print("Falling back to CPU.")
            print()
            print("Processing may be significantly slower.")

        # Lazily import the vendored architecture (torch-only, no cv2).
        from video_upscaler import upcunet_v3
        from video_upscaler.quantize import ensure_fp16

        scale = scale_for_model(model_name)
        self._scale = scale
        self._half = device == "cuda"
        self._device = device
        weight_path = ensure_fp16(model_name) if self._half else model_path
        self._upscaler = upcunet_v3.RealWaifuUpScaler(
            scale, str(weight_path), half=self._half, device=device
        )
        self._tile_mode = CUGAN_TILE_MODE
        self._cache_mode = resolve_cache_mode()
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            from video_upscaler import fast_path

            fast_path.tritonize(self._upscaler.model)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Enhance one RGB frame; returns a numpy array (uint8, scaled)."""
        result = self._upscaler(
            frame, tile_mode=self._tile_mode, cache_mode=self._cache_mode, alpha=1.0
        )
        return np.ascontiguousarray(result)
