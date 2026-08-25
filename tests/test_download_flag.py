"""Tests for the --download-models CLI flag parsing (torch-free)."""

from __future__ import annotations

import pytest

from video_upscaler.cli import _parse_download_models_arg


def test_absent_flag_returns_none() -> None:
    assert _parse_download_models_arg([]) is None
    assert _parse_download_models_arg(["--backend", "ncnn"]) is None


def test_essential_and_all_tiers() -> None:
    assert _parse_download_models_arg(["--download-models", "essential"]) == "essential"
    assert _parse_download_models_arg(["--download-models", "all"]) == "all"


def test_missing_value_raises() -> None:
    with pytest.raises(SystemExit, match="requires a value"):
        _parse_download_models_arg(["--download-models"])


def test_invalid_tier_raises() -> None:
    with pytest.raises(SystemExit, match="Invalid --download-models"):
        _parse_download_models_arg(["--download-models", "everything"])
