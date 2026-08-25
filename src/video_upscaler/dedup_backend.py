"""MultiPassDedup backend: model resolution, device detection, and weights verification."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from video_upscaler import config

DEDUP_MODELS: Final[tuple[str, ...]] = ("gmfss", "rife")

DEDUP_MODEL_NAMES: Final[dict[str, str]] = {
    "gmfss": "GMFSS Fortuna (Best Quality, Default)",
    "rife": "Practical-RIFE (Faster)",
}

# Every weight file each model loads at runtime (relative to DEDUP_MODELS_DIR).
# GMFSS pg104 loads all five train_log_pg104 pickles (GMFSS.py:57-61) —
# checking only the directory's existence let partial downloads pass.
DEDUP_MODEL_WEIGHT_FILES: Final[dict[str, tuple[str, ...]]] = {
    "gmfss": (
        "train_log_pg104/feat.pkl",
        "train_log_pg104/flownet.pkl",
        "train_log_pg104/fusionnet.pkl",
        "train_log_pg104/metric.pkl",
        "train_log_pg104/rife.pkl",
    ),
    "rife": ("rife48.pkl",),
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


def dedup_weight_files(model_type: str) -> tuple[str, ...]:
    """Return the weight files (relative to DEDUP_MODELS_DIR) a model loads."""
    model_key = validate_model_type(model_type)
    return DEDUP_MODEL_WEIGHT_FILES[model_key]


def missing_dedup_weights(model_type: str) -> list[Path]:
    """Return the required weight files that are absent on disk."""
    model_key = validate_model_type(model_type)
    return [
        config.DEDUP_MODELS_DIR / rel
        for rel in DEDUP_MODEL_WEIGHT_FILES[model_key]
        if not (config.DEDUP_MODELS_DIR / rel).is_file()
    ]


def check_dedup_weights(model_type: str) -> str | None:
    """Check if all model weights exist in models/multipassdedup/.

    Returns None if every required file exists, or a helpful error string
    listing the missing files.
    """
    model_key = validate_model_type(model_type)
    missing = missing_dedup_weights(model_key)
    if not missing:
        return None

    listed = "\n".join(f"  - {path}" for path in missing)
    return (
        f"MultiPassDedup weights for {model_key.upper()} are incomplete.\n"
        f"Missing files:\n{listed}\n\n"
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
