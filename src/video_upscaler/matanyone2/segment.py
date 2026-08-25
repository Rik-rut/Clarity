"""Click-based first-frame subject auto-detection.

The official MatAnyone2 demo picks the target with a single click; this
module brings that workflow to Clarity's mask editor without pulling in a
segmentation model. A user click seeds OpenCV GrabCut:

- a small sure-foreground disc at the clicked pixel,
- a larger probable-foreground halo around it,
- a probable-background ring beyond the halo,
- a sure-background image border.

Runs on CPU in well under a second at preview resolution, so the browser
receives an editable mask instantly and can refine it with brush strokes.
Swapping in SAM/SAM2 later only requires replacing ``detect_subject_mask``
behind the same signature (spec §2 "Preferred segmentation implementation").
"""

from __future__ import annotations

import cv2
import numpy as np


class SubjectDetectionError(ValueError):
    """Raised when no usable subject could be found near the click."""


# Processing cap for the long side; GrabCut cost grows super-linearly with
# area and the mask is upscaled back to native size afterwards anyway.
_MAX_DETECT_SIDE = 1024


def _resize_for_detect(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= _MAX_DETECT_SIDE:
        return image, 1.0
    scale = _MAX_DETECT_SIDE / float(long_side)
    resized = cv2.resize(
        image, (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def detect_subject_mask(
    image_bgr: np.ndarray,
    point_xy: tuple[int, int],
    *,
    seed_radius_frac: float = 0.012,
    halo_radius_frac: float = 0.045,
    iterations: int = 5,
) -> np.ndarray:
    """Return a uint8 HxW mask (255 = subject) for a click on ``image_bgr``.

    Raises :class:`SubjectDetectionError` when GrabCut finds nothing
    foreground-like near the click (e.g. empty frame, click on pure
    background).
    """
    if image_bgr is None or image_bgr.ndim != 3:
        raise SubjectDetectionError("Frame image is not readable.")
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise SubjectDetectionError("Frame image has zero size.")

    work, scale = _resize_for_detect(image_bgr)
    wh, ww = work.shape[:2]
    min_side = min(wh, ww)
    cx = int(round(point_xy[0] * scale))
    cy = int(round(point_xy[1] * scale))
    cx = min(max(cx, 0), ww - 1)
    cy = min(max(cy, 0), wh - 1)

    seed_r = max(2, int(round(min_side * seed_radius_frac)))
    halo_r = max(seed_r + 2, int(round(min_side * halo_radius_frac)))

    gc_input = np.full((wh, ww), cv2.GC_PR_BGD, dtype=np.uint8)
    # Image border is background in practice; a thin sure-BGD frame anchors
    # the GMMs even when the click sits near the edge.
    border = max(1, int(round(min_side * 0.02)))
    gc_input[:border, :] = cv2.GC_BGD
    gc_input[-border:, :] = cv2.GC_BGD
    gc_input[:, :border] = cv2.GC_BGD
    gc_input[:, -border:] = cv2.GC_BGD
    # Probable-foreground halo around the click, with a small sure-foreground
    # seed disc. Everything between halo and border stays probable-background,
    # so GrabCut is free to claim whatever matches the foreground model —
    # marking it sure-background would forbid any growth beyond the halo.
    cv2.circle(gc_input, (cx, cy), halo_r, cv2.GC_PR_FGD, thickness=-1)
    cv2.circle(gc_input, (cx, cy), seed_r, cv2.GC_FGD, thickness=-1)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            work,
            gc_input,
            None,
            bgd_model,
            fgd_model,
            int(iterations),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        raise SubjectDetectionError(f"GrabCut failed on this frame: {exc}") from exc

    subject = np.isin(gc_input, (cv2.GC_FGD, cv2.GC_PR_FGD))
    if not subject.any():
        raise SubjectDetectionError(
            "No subject found near that point. Try clicking closer to the "
            "person/object, or paint the target with the brush."
        )

    mask = (subject.astype(np.uint8)) * 255
    if scale != 1.0:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    # Close pinholes so downstream dilation/erosion starts from a solid blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def mask_to_white_png_b64(mask: np.ndarray) -> str:
    """Encode a binary mask as a white-on-transparent PNG data URL.

    White + alpha lets the browser composite it directly onto the stroke
    canvas with a plain drawImage call.
    """
    rgba = np.zeros((*mask.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = mask
    ok, buffer = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("Failed to encode detection mask PNG.")
    import base64

    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
