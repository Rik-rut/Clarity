"""SAM-based click-to-segment for the first-frame mask editor.

Primary auto-detection backend, mirroring the official MatAnyone demo
(which drives its first-frame target selection with a Segment Anything
predictor). One or more positive clicks are prompted through SAM and the
resulting mask is returned for the browser canvas to composite.

The checkpoint (sam_vit_b) is managed through the model manifest like every
other Clarity model. When it is absent, callers fall back to the GrabCut
heuristic in :mod:`video_upscaler.matanyone2.segment`.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

SAM_CKPT_NAME = "sam_vit_b_01ec64.pth"
# vit_b is the quality/size sweet spot (~375 MB); the demo ships vit_h
# (~2.4 GB) which is impractical for consumer machines.

_lock = threading.Lock()
_predictor: object | None = None
_device_used: str | None = None
# The image embedding is expensive to compute; cache the last encoded frame
# so multi-click refinement only pays for it once.
_embedding_key: tuple | None = None


class SamModelMissing(RuntimeError):
    """The SAM checkpoint is not installed."""


def checkpoint_path() -> Path:
    from video_upscaler import config

    return config.MODELS_DIR / "sam" / SAM_CKPT_NAME


def sam_installed() -> bool:
    return checkpoint_path().is_file()


def _load_segment_anything():
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "The segment-anything package is not installed."
        ) from exc
    return sam_model_registry, SamPredictor


def _get_predictor():
    """Lazy singleton SamPredictor on the best available device."""
    global _predictor, _device_used
    ckpt = checkpoint_path()
    if not ckpt.is_file():
        raise SamModelMissing(
            f"SAM checkpoint not found:\n{ckpt}\n\n"
            "Install it with:\n"
            "  uv run --all-extras main.py --download-models all"
        )
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with _lock:
        if _predictor is None or _device_used != device:
            registry, predictor_cls = _load_segment_anything()
            model = registry["vit_b"](checkpoint=str(ckpt))
            model.to(device=device)
            model.eval()
            _predictor = predictor_cls(model)
            _device_used = device
    return _predictor, device


def release_sam() -> None:
    """Drop the cached predictor (between jobs / in tests)."""
    global _predictor, _device_used, _embedding_key
    with _lock:
        _predictor = None
        _device_used = None
        _embedding_key = None


def _ensure_image_encoded(predictor, image_rgb: np.ndarray, cache_key: tuple) -> None:
    """Run set_image unless this exact frame is already encoded."""
    global _embedding_key
    if _embedding_key == cache_key and getattr(predictor, "features", None) is not None:
        return
    predictor.set_image(image_rgb.astype(np.uint8))
    _embedding_key = cache_key


def detect_subject_mask_sam(
    image_rgb: np.ndarray,
    points_xy: list[tuple[float, float]],
    labels: list[int] | None = None,
    cache_key: tuple | None = None,
) -> np.ndarray:
    """Return a uint8 HxW 0/255 mask for the clicked point(s).

    ``image_rgb`` is HxWx3 uint8 RGB. Multiple points accumulate as a
    multi-click prompt (all positive unless ``labels`` says otherwise).
    ``cache_key`` (e.g. video path + mtime) lets repeated clicks on the
    same first frame reuse the encoded image embedding.
    Raises SamModelMissing when the checkpoint is absent so callers can
    fall back to GrabCut.
    """
    if not points_xy:
        raise ValueError("At least one click point is required.")
    predictor, device = _get_predictor()
    import torch

    coords = np.array([[float(x), float(y)] for x, y in points_xy], dtype=np.float32)
    if labels is None:
        label_arr = np.ones(len(coords), dtype=np.int64)
    else:
        label_arr = np.asarray(labels, dtype=np.int64)
        if len(label_arr) != len(coords):
            raise ValueError("labels must match points in length.")

    with torch.inference_mode():
        if cache_key is not None:
            _ensure_image_encoded(predictor, image_rgb, cache_key)
        else:
            predictor.set_image(image_rgb.astype(np.uint8))
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=label_arr,
            multimask_output=True,
        )
    # Highest-score mask wins; binarize.
    best = int(np.argmax(scores))
    mask = (masks[best] > 0.0).astype(np.uint8) * 255
    if not mask.any():
        raise ValueError(
            "No subject found near that point. Try clicking closer to the "
            "person/object, or paint the target with the brush."
        )
    return mask
