"""Matting backend selection and session protocol.

Torch-free at import. Mirrors the AMT selector's contract: auto resolves to
the fastest available implementation, explicit requests that cannot be
honored raise instead of silently degrading. The TensorRT engine path arrives
in Phase 2 behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import torch

VALID_BACKENDS = ("auto", "pytorch", "tensorrt")


class MatAnyone2BackendUnavailable(RuntimeError):
    """An explicitly requested matting backend cannot be used."""


@dataclass(frozen=True)
class BackendSelection:
    name: str          # "pytorch"; "tensorrt" arrives in Phase 2
    precision: str     # "fp16" | "fp32"
    device: str        # "cuda" | "cpu"
    fallback_reason: str | None = None


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a core dep
        return False
    return bool(torch.cuda.is_available())


class MattingSession(Protocol):
    """One sequence-bound matting session over a loaded model."""

    def start(
        self, first_frame_chw: "torch.Tensor", mask_hw: "torch.Tensor",
        warmup: int = 10,
    ) -> None:
        """Encode the first-frame mask and prime recurrent state."""
        ...

    def step(self, frame_chw: "torch.Tensor") -> "np.ndarray":
        """Process one frame; returns HW float32 probabilities in [0, 1]."""
        ...

    def close(self) -> None:
        """Release per-session resources (the shared model stays cached)."""
        ...


def select_backend(requested: str = "auto", precision: str = "fp16") -> BackendSelection:
    requested = (requested or "auto").strip().lower()
    precision = (precision or "fp16").strip().lower()
    if requested not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown MatAnyone2 backend '{requested}'. "
            f"Expected one of: {', '.join(VALID_BACKENDS)}."
        )
    if precision not in ("fp16", "fp32"):
        raise ValueError("Precision must be 'fp16' or 'fp32'.")

    if requested == "tensorrt":
        # Phase 2 wires the engine path here. An explicit request must fail
        # loudly rather than silently changing behavior (AMT convention).
        raise MatAnyone2BackendUnavailable(
            "TensorRT matting is not available yet; choose Auto or PyTorch."
        )

    cuda = _cuda_available()
    if cuda:
        return BackendSelection("pytorch", precision, "cuda")

    reason = "CUDA unavailable; running MatAnyone2 on CPU with FP32."
    return BackendSelection("pytorch", "fp32", "cpu", reason)
