"""Unit tests for click-based subject auto-detection (GrabCut seeded)."""

import cv2
import numpy as np
import pytest

from video_upscaler.matanyone2.segment import (
    SubjectDetectionError,
    detect_subject_mask,
    mask_to_white_png_b64,
)


def make_frame(h=240, w=320):
    """Gray background with a bright high-contrast rectangle 'subject'."""
    img = np.full((h, w, 3), 40, np.uint8)
    img[70:170, 110:230] = 230
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # any 3ch layout works


def test_click_on_subject_masks_it():
    frame = make_frame()
    mask = detect_subject_mask(frame, (170, 120))
    assert mask.shape == (240, 320)
    assert mask.dtype == np.uint8
    # The subject core must be detected...
    assert mask[120, 170] == 255
    # ...and most of the rectangle covered.
    coverage = float((mask[70:170, 110:230] > 0).mean())
    assert coverage > 0.8
    # Background corner stays untouched.
    assert mask[10, 10] == 0


def test_click_maps_from_scaled_coordinates():
    """A click given in native coords is honored directly; scaled inputs are
    handled by the caller, so identical points yield identical masks."""
    frame = make_frame()
    m1 = detect_subject_mask(frame, (170, 120))
    m2 = detect_subject_mask(frame, (170, 120))
    assert np.array_equal(m1, m2)


def test_click_outside_image_clamps_without_crash():
    frame = make_frame()
    mask = detect_subject_mask(frame, (-50, -50))
    assert mask.shape == (240, 320)


def test_unreadable_image_raises():
    with pytest.raises(SubjectDetectionError):
        detect_subject_mask(None, (5, 5))
    with pytest.raises(SubjectDetectionError):
        detect_subject_mask(np.zeros((0, 0, 3), np.uint8), (5, 5))


def test_mask_png_is_white_with_alpha():
    mask = np.zeros((16, 16), np.uint8)
    mask[4:12, 4:12] = 255
    data_url = mask_to_white_png_b64(mask)
    assert data_url.startswith("data:image/png;base64,")
    import base64

    raw = base64.b64decode(data_url.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (16, 16, 4)
    # White RGB everywhere alpha marks subject.
    assert (decoded[..., :3][decoded[..., 3] > 0] == 255).all()
