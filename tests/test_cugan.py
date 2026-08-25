"""Torch-free tests for Real-CUGAN model scale derivation and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

import video_upscaler.config as config
from video_upscaler.cugan import check_cugan, resolve_cache_mode, scale_from_model
from video_upscaler.config import MODELS_DIR


def test_scale_from_model() -> None:
    assert scale_from_model("up2x-latest-denoise2x.pth") == 2
    assert scale_from_model("up3x-latest-no-denoise.pth") == 3
    assert scale_from_model("up4x-latest-conservative.pth") == 4


def test_scale_from_model_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        scale_from_model("some_unknown_model.pth")


def test_check_cugan_missing_returns_helpful_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("video_upscaler.cugan.MODELS_DIR", tmp_path)
    message = check_cugan("up2x-latest-denoise2x.pth")
    assert message is not None
    assert "Required Real-CUGAN model not found" in message
    assert "up2x-latest-denoise2x.pth" in message


def test_check_cugan_returns_none_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("video_upscaler.cugan.MODELS_DIR", tmp_path)
    (tmp_path / "up2x-latest-denoise2x.pth").write_bytes(b"dummy")
    assert check_cugan("up2x-latest-denoise2x.pth") is None


def test_models_dir_exists() -> None:
    # The project ships no weights in git; the dir is created at startup.
    assert MODELS_DIR.name == "models"


def test_resolve_cache_mode_reads_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "CUGAN_CACHE_MODE", 3)
    assert resolve_cache_mode() == 3
    monkeypatch.setattr(config, "CUGAN_CACHE_MODE", 2)
    assert resolve_cache_mode() == 2
