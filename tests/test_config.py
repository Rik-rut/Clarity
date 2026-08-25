"""Tests for central configuration."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import video_upscaler.config as config


def test_default_dirs_resolve_under_base_dir() -> None:
    assert config.INPUT_DIR == config.BASE_DIR / "input"
    assert config.OUTPUT_DIR == config.BASE_DIR / "output"
    assert config.MODELS_DIR == config.BASE_DIR / "models"


def test_supported_extensions_contains_all_spec_extensions() -> None:
    expected = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}
    assert expected <= config.SUPPORTED_EXTENSIONS


def test_env_overrides_via_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLARITY_INPUT_DIR", str(tmp_path / "in"))
    monkeypatch.setenv("CLARITY_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CLARITY_MODELS_DIR", str(tmp_path / "mdls"))
    module = importlib.reload(config)
    try:
        assert module.INPUT_DIR == tmp_path / "in"
        assert module.OUTPUT_DIR == tmp_path / "out"
        assert module.MODELS_DIR == tmp_path / "mdls"
    finally:
        monkeypatch.delenv("CLARITY_INPUT_DIR", raising=False)
        monkeypatch.delenv("CLARITY_OUTPUT_DIR", raising=False)
        monkeypatch.delenv("CLARITY_MODELS_DIR", raising=False)
        importlib.reload(config)


def test_ffmpeg_path_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLARITY_FFMPEG", "C:/tools/ffmpeg.exe")
    assert config.ffmpeg_path() == "C:/tools/ffmpeg.exe"


def test_use_nvenc_env_via_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLARITY_NVENC", "1")
    module = importlib.reload(config)
    try:
        assert module.USE_NVENC is True
    finally:
        monkeypatch.delenv("CLARITY_NVENC", raising=False)
        importlib.reload(config)


def test_use_nvenc_default_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLARITY_NVENC", raising=False)
    module = importlib.reload(config)
    try:
        assert module.USE_NVENC is False
    finally:
        importlib.reload(config)


def test_max_output_width_warning_default() -> None:
    assert config.MAX_OUTPUT_WIDTH_WARNING == 7680


def test_ensure_directories_creates_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "INPUT_DIR", tmp_path / "input")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "DEDUP_MODELS_DIR", tmp_path / "models" / "multipassdedup")
    config.ensure_directories()
    assert (tmp_path / "input").is_dir()
    assert (tmp_path / "output").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "models" / "multipassdedup").is_dir()


def test_dedup_env_overrides_via_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLARITY_DEDUP_MODEL", "rife")
    monkeypatch.setenv("CLARITY_DEDUP_NPASS", "2")
    monkeypatch.setenv("CLARITY_DEDUP_SCALE", "0.5")
    monkeypatch.setenv("CLARITY_DEDUP_SCDET", "0")
    monkeypatch.setenv("CLARITY_DEDUP_SCDET_THRESHOLD", "0.4")
    monkeypatch.setenv("CLARITY_DEDUP_BACKEND", "torch+cupy")
    module = importlib.reload(config)
    try:
        assert module.DEDUP_MODEL_DEFAULT == "rife"
        assert module.DEDUP_NPASS_DEFAULT == "2"
        assert module.DEDUP_SCALE_DEFAULT == 0.5
        assert module.DEDUP_SCDET_DEFAULT is False
        assert module.DEDUP_SCDET_THRESHOLD == 0.4
        assert module.DEDUP_BACKEND_PREF == "torch+cupy"
    finally:
        for k in (
            "CLARITY_DEDUP_MODEL",
            "CLARITY_DEDUP_NPASS",
            "CLARITY_DEDUP_SCALE",
            "CLARITY_DEDUP_SCDET",
            "CLARITY_DEDUP_SCDET_THRESHOLD",
            "CLARITY_DEDUP_BACKEND",
        ):
            monkeypatch.delenv(k, raising=False)
        importlib.reload(config)

