"""Tests for the CLI --backend flag parsing (torch-free)."""

from __future__ import annotations

import pytest

from video_upscaler import config
from video_upscaler.cli import _apply_cli_backend


def test_no_flag_leaves_preference_untouched() -> None:
    config.BACKEND_PREF = "auto"
    _apply_cli_backend([])
    assert config.BACKEND_PREF == "auto"
    _apply_cli_backend(["--profile", "2x_Balanced"])
    assert config.BACKEND_PREF == "auto"


def test_flag_sets_preference() -> None:
    config.BACKEND_PREF = "auto"
    _apply_cli_backend(["--backend", "ncnn"])
    assert config.BACKEND_PREF == "ncnn"


def test_flag_case_insensitive() -> None:
    config.BACKEND_PREF = "auto"
    _apply_cli_backend(["--backend", "TensorRT"])
    assert config.BACKEND_PREF == "tensorrt"


def test_flag_after_other_args() -> None:
    config.BACKEND_PREF = "auto"
    _apply_cli_backend(["--video", "x.mp4", "--backend", "torch"])
    assert config.BACKEND_PREF == "torch"


def test_missing_value_raises() -> None:
    with pytest.raises(SystemExit, match="requires a value"):
        _apply_cli_backend(["--backend"])


def test_invalid_value_raises() -> None:
    with pytest.raises(SystemExit, match="Invalid --backend"):
        _apply_cli_backend(["--backend", "cuda"])


def test_backend_detection_honors_runtime_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    monkeypatch.setattr(config, "BACKEND_PREF", "ncnn")
    monkeypatch.setitem(
        sys.modules, "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
    )
    from video_upscaler import backend

    assert backend.detect_backend() == "ncnn"
    monkeypatch.setattr(config, "BACKEND_PREF", "auto")
    monkeypatch.setattr(backend, "_tensorrt_available", lambda: True)
    assert backend.detect_backend() == "tensorrt"
