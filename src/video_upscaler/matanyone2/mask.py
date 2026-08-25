"""First-frame mask handling for MatAnyone2.

Masks travel from the browser canvas as base64-encoded grayscale PNGs where
nonzero pixels mark the target. Preprocessing mirrors the official demo
exactly: dilation runs BEFORE erosion, and downscaling (when a max-size cap
is active) uses nearest-neighbour resampling so masks stay binary.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

import cv2
import numpy as np


class MaskValidationError(ValueError):
    """Raised when a submitted mask cannot be used for the given video."""


def decode_mask_b64(data_b64: str) -> np.ndarray:
    """Decode a base64 grayscale PNG into a uint8 HxW mask array."""
    if not data_b64 or not isinstance(data_b64, str):
        raise MaskValidationError("Mask payload is empty.")
    if data_b64.startswith("data:") and "," in data_b64[:128]:
        data_b64 = data_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise MaskValidationError(f"Mask is not valid base64: {exc}") from exc
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise MaskValidationError("Mask payload is not a decodable image.")
    return image


def validate_mask(mask: np.ndarray, width: int, height: int) -> None:
    """Ensure the mask geometry matches the source video frame exactly."""
    if mask.ndim != 2 or mask.shape != (height, width):
        got = f"{mask.shape[1]}x{mask.shape[0]}" if mask.ndim == 2 else str(mask.shape)
        raise MaskValidationError(
            f"Mask size {got} does not match video frame {width}x{height}. "
            "Re-select the target on the current video."
        )


def _morph(mask: np.ndarray, radius: int, operation) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    return operation(mask, kernel, iterations=1)


def preprocess_mask(
    mask: np.ndarray,
    dilate: int = 10,
    erode: int = 10,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply dilation then erosion (upstream order); optionally resize."""
    processed = mask
    if dilate > 0:
        processed = _morph(processed, dilate, cv2.dilate)
    if erode > 0:
        processed = _morph(processed, erode, cv2.erode)
    if target_size is not None:
        width, height = target_size
        processed = cv2.resize(
            processed, (width, height), interpolation=cv2.INTER_NEAREST
        )
    return processed


def processing_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Processing resolution under the upstream max-size rule."""
    if max_size <= 0:
        return width, height
    min_side = min(width, height)
    if min_side <= max_size:
        return width, height
    new_h = int(height / min_side * max_size)
    new_w = int(width / min_side * max_size)
    return max(1, new_w), max(1, new_h)


def save_mask(mask: np.ndarray, path: Path) -> Path:
    """Persist a mask as PNG beside job artifacts; returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"Failed to encode mask PNG for {path.name}")
    path.write_bytes(buffer.tobytes())
    return path
