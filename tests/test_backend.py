"""Tests for backend detection (torch-cuda / ncnn / torch-cpu)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _fake_torch(cuda_available: bool) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    torch.__version__ = "2.13.0"
    return torch


def _no_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub tensorrt detection off so tests stay deterministic."""
    monkeypatch.setattr("video_upscaler.backend._tensorrt_available", lambda: False)


def test_detects_torch_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_tensorrt(monkeypatch)
    monkeypatch.setattr("video_upscaler.config.BACKEND_PREF", "auto")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    from video_upscaler import backend

    assert backend.detect_backend() == "torch-cuda"


def test_detects_tensorrt_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video_upscaler.config.BACKEND_PREF", "auto")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    monkeypatch.setattr("video_upscaler.backend._tensorrt_available", lambda: True)
    from video_upscaler import backend

    assert backend.detect_backend() == "tensorrt"


def test_forced_tensorrt_falls_back_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("video_upscaler.config.BACKEND_PREF", "tensorrt")
    monkeypatch.setattr("video_upscaler.backend._tensorrt_available", lambda: False)
    from video_upscaler import backend

    assert backend.detect_backend() == "torch-cpu"


def test_detects_ncnn_when_no_cuda_but_tool_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("video_upscaler.config.BACKEND_PREF", "auto")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    fake = tmp_path / "realcugan-ncnn-vulkan.exe"
    fake.touch()
    fake_ncnn = types.SimpleNamespace(ncnn_exe=lambda tool: fake)
    monkeypatch.setitem(sys.modules, "video_upscaler.ncnn", fake_ncnn)
    from video_upscaler import backend

    assert backend.detect_backend() == "ncnn"


def test_detects_torch_cpu_when_nothing_else(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video_upscaler.config.BACKEND_PREF", "auto")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    fake_ncnn = types.SimpleNamespace(
        ncnn_exe=lambda tool: Path("missing.exe")
    )
    monkeypatch.setitem(sys.modules, "video_upscaler.ncnn", fake_ncnn)
    from video_upscaler import backend

    assert backend.detect_backend() == "torch-cpu"


def test_forced_torch_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video_upscaler.config.BACKEND_PREF", "torch")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    from video_upscaler import backend

    assert backend.detect_backend() == "torch-cuda"


def test_forced_ncnn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("video_upscaler.config.BACKEND_PREF", "ncnn")
    fake = tmp_path / "realcugan-ncnn-vulkan.exe"
    fake.touch()
    fake_ncnn = types.SimpleNamespace(ncnn_exe=lambda tool: fake)
    monkeypatch.setitem(sys.modules, "video_upscaler.ncnn", fake_ncnn)
    from video_upscaler import backend

    assert backend.detect_backend() == "ncnn"


def test_backend_label() -> None:
    from video_upscaler import backend

    assert backend.backend_label("tensorrt") == "TensorRT (fp16)"
    assert backend.backend_label("torch-cuda") == "CUDA (torch fp16)"
    assert backend.backend_label("ncnn") == "ncnn Vulkan"
    assert backend.backend_label("torch-cpu") == "CPU (torch)"
