"""MultiPassDedup backend: model resolution, device detection, and weights verification."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from video_upscaler import config

DEDUP_MODELS: Final[tuple[str, ...]] = ("gmfss", "rife", "gimm")

DEDUP_MODEL_NAMES: Final[dict[str, str]] = {
    "gmfss": "GMFSS Fortuna (Best Quality, Default)",
    "rife": "Practical-RIFE (Faster)",
    "gimm": "GIMM-VFI",
}

DEDUP_MODEL_WEIGHT_FILES: Final[dict[str, str]] = {
    "gmfss": "train_log_pg104",
    "rife": "rife48.pkl",
    "gimm": "gimmvfi_r_arb_lpips.pt",
}


def validate_model_type(model_type: str) -> str:
    """Validate and normalize the MultiPassDedup model type name.

    Raises ValueError if the model type is not recognized.
    """
    normalized = model_type.strip().lower()
    if normalized not in DEDUP_MODELS:
        raise ValueError(
            f"Invalid MultiPassDedup model {model_type!r}. Supported models: {', '.join(DEDUP_MODELS)}."
        )
    return normalized


def parse_npass(npass: str | int) -> int:
    """Parse npass cadence parameter into the upstream integer representation.

    0 = auto-detect, 2 = on-twos, 3 = on-threes.
    """
    if isinstance(npass, str):
        val = npass.strip().lower()
        if val in ("auto", "0"):
            return 0
        if val == "2":
            return 2
        if val == "3":
            return 3
        try:
            parsed = int(val)
            if parsed >= 0:
                return parsed
        except ValueError:
            pass
        raise ValueError(
            f"Invalid npass {npass!r}. Supported values: 'auto', 2, 3 (or integer >= 0)."
        )
    if isinstance(npass, int):
        if npass >= 0:
            return npass
        raise ValueError(f"npass must be non-negative, got {npass}.")
    raise TypeError(f"Expected str or int for npass, got {type(npass).__name__}.")


def get_dedup_weights_path(model_type: str) -> Path:
    """Return the expected weights path for a given model type."""
    model_key = validate_model_type(model_type)
    return config.DEDUP_MODELS_DIR / DEDUP_MODEL_WEIGHT_FILES[model_key]


def check_dedup_weights(model_type: str) -> str | None:
    """Check if model weights exist in models/multipassdedup/.

    Returns None if weights exist, or a helpful error string if missing.
    """
    model_key = validate_model_type(model_type)
    target_path = get_dedup_weights_path(model_key)
    if target_path.exists():
        return None

    filename = DEDUP_MODEL_WEIGHT_FILES[model_key]
    return (
        f"MultiPassDedup weights for {model_key.upper()} are not installed.\n"
        f"Missing: {target_path}\n\n"
        f"Run the app again and answer [Y] to download them automatically,\n"
        f"or run once with --download-models all. Files land in:\n"
        f"  {config.DEDUP_MODELS_DIR}"
    )


def _torch_cuda_available() -> bool:
    """Check CUDA availability without pulling torch into module top level."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_dedup_device() -> str:
    """Resolve the device for MultiPassDedup (cuda or cpu)."""
    pref = config.PREFERRED_DEVICE
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if _torch_cuda_available() else "cpu"
    # auto
    return "cuda" if _torch_cuda_available() else "cpu"
