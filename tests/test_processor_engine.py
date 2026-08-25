"""Tests for build_engine backend routing (torch-free)."""

from __future__ import annotations

import sys
import types

import pytest

from video_upscaler import processor


def _default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(processor, "config", types.SimpleNamespace(
        PREFERRED_DEVICE="auto", BACKEND_PREF="auto",
    ))


def _fake_cugan(monkeypatch: pytest.MonkeyPatch, cls=None) -> None:
    class _Default:
        def __init__(self, model_name):
            pass

    monkeypatch.setitem(
        sys.modules, "video_upscaler.cugan",
        types.SimpleNamespace(RealCUGANEngine=cls or _Default),
    )


def test_torch_backend_builds_cugan_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    built = {}

    class _FakeCugan:
        def __init__(self, model_name):
            built["cugan"] = model_name

    _default_config(monkeypatch)
    _fake_cugan(monkeypatch, _FakeCugan)

    assert isinstance(processor.build_engine("2x_Balanced", "torch-cuda"), _FakeCugan)
    assert built["cugan"] == "up2x-latest-denoise2x.pth"


def test_ncnn_backend_routes_to_ncnn_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    built = {}

    class _FakeNCNN:
        def __init__(self, profile):
            built["ncnn"] = profile

    _default_config(monkeypatch)
    _fake_cugan(monkeypatch)
    monkeypatch.setitem(sys.modules, "video_upscaler.ncnn", types.SimpleNamespace(NCNNEngine=_FakeNCNN))

    assert isinstance(processor.build_engine("4x_Deep", "ncnn"), _FakeNCNN)
    assert built["ncnn"] == "4x_Deep"


def test_tensorrt_backend_builds_cugan_2x(monkeypatch: pytest.MonkeyPatch) -> None:
    built = {}

    class _FakeTRT:
        def __init__(self, model_name):
            built["trt"] = model_name

    _default_config(monkeypatch)
    monkeypatch.setitem(sys.modules, "video_upscaler.tensorrt_backend", types.SimpleNamespace(RealCUGANTensorRTEngine=_FakeTRT))
    assert isinstance(processor.build_engine("2x_Balanced", "tensorrt"), _FakeTRT)
    assert built["trt"] == "up2x-latest-denoise2x.pth"


def test_tensorrt_auto_falls_back_to_torch_for_3x_and_4x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = {}

    class _FakeCugan:
        def __init__(self, model_name):
            built["cugan"] = model_name

    _default_config(monkeypatch)
    _fake_cugan(monkeypatch, _FakeCugan)

    # Auto mode: 3x/4x profiles have no TensorRT engine -> torch.
    assert isinstance(processor.build_engine("3x_Deep", "tensorrt"), _FakeCugan)
    assert built["cugan"] == "up3x-latest-denoise3x.pth"
    assert isinstance(processor.build_engine("4x_Deep", "tensorrt"), _FakeCugan)
    assert built["cugan"] == "up4x-latest-denoise3x.pth"


def test_tensorrt_forced_2x_builds_trt_even_with_trt_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = {}

    class _FakeTRT:
        def __init__(self, model_name):
            built["trt"] = model_name

    monkeypatch.setattr(processor, "config", types.SimpleNamespace(
        PREFERRED_DEVICE="auto", BACKEND_PREF="tensorrt",
    ))
    monkeypatch.setitem(sys.modules, "video_upscaler.tensorrt_backend", types.SimpleNamespace(RealCUGANTensorRTEngine=_FakeTRT))
    assert isinstance(processor.build_engine("2x_Balanced", "tensorrt"), _FakeTRT)
    assert built["trt"] == "up2x-latest-denoise2x.pth"


def test_effective_backend_2x_keeps_tensorrt() -> None:
    assert processor.effective_backend("tensorrt", "2x_Balanced") == "tensorrt"


def test_effective_backend_3x_falls_back_in_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _default_config(monkeypatch)
    assert processor.effective_backend("tensorrt", "3x_Deep") == "torch-cuda"
    assert processor.effective_backend("tensorrt", "4x_Deep") == "torch-cuda"


def test_effective_backend_forced_tensorrt_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(processor, "config", types.SimpleNamespace(
        PREFERRED_DEVICE="auto", BACKEND_PREF="tensorrt",
    ))
    assert processor.effective_backend("tensorrt", "4x_Deep") == "tensorrt"
