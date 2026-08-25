"""Unit tests for MatAnyone2 mask decoding, validation, preprocessing."""

import base64

import cv2
import numpy as np
import pytest

from video_upscaler.matanyone2.mask import (
    MaskValidationError,
    decode_mask_b64,
    preprocess_mask,
    processing_size,
    save_mask,
    validate_mask,
)


def _png_b64(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _blob(h=8, w=8, radius=2):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), radius, 255, -1)
    return m


def test_decode_roundtrip():
    mask = _blob()
    decoded = decode_mask_b64(_png_b64(mask))
    assert decoded.dtype == np.uint8
    assert decoded.shape == mask.shape
    assert int(decoded.sum()) == int(mask.sum())


def test_decode_accepts_data_url_prefix():
    mask = _blob()
    payload = "data:image/png;base64," + _png_b64(mask)
    assert decode_mask_b64(payload).shape == mask.shape


def test_decode_rejects_garbage():
    with pytest.raises(MaskValidationError):
        decode_mask_b64("not-a-real-image")


def test_decode_rejects_empty():
    with pytest.raises(MaskValidationError):
        decode_mask_b64("")


def test_validate_rejects_dimension_mismatch():
    with pytest.raises(MaskValidationError):
        validate_mask(_blob(8, 8), width=16, height=8)


def test_dilate_grows_target_region():
    mask = _blob(radius=2)
    grown = preprocess_mask(mask, dilate=3, erode=0)
    assert int((grown > 0).sum()) > int((mask > 0).sum())


def test_erode_shrinks_target_region():
    mask = _blob(radius=4)
    shrunk = preprocess_mask(mask, dilate=0, erode=3)
    assert 0 < int((shrunk > 0).sum()) < int((mask > 0).sum())


def test_resize_nearest_keeps_binary_and_size():
    mask = _blob(16, 24, radius=5)
    out = preprocess_mask(mask, dilate=0, erode=0, target_size=(12, 8))
    assert out.shape == (8, 12)
    assert set(np.unique(out)).issubset({0, 255})


def test_processing_size_passthrough_when_no_cap():
    assert processing_size(1920, 1080, -1) == (1920, 1080)
    assert processing_size(1280, 720, 720) == (1280, 720)


def test_processing_size_matches_upstream_formula():
    # upstream: new_h=int(h/min*max), new_w=int(w/min*max)
    assert processing_size(3840, 2160, 1080) == (1920, 1080)
    w, h = processing_size(1000, 667, 500)
    assert (w, h) == (int(1000 / 667 * 500), int(667 / 667 * 500))


def test_save_mask_writes_readable_png(tmp_path):
    mask = _blob()
    out = save_mask(mask, tmp_path / "jobdir" / "mask.png")
    assert out.is_file()
    back = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert back.shape == mask.shape
