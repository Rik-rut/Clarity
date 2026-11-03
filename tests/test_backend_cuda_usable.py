"""Tests for kernel-level CUDA usability check and fallback."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from video_upscaler import backend
from video_upscaler.backend import cuda_usable


def test_cuda_usable_success(monkeypatch: pytest.MonkeyPatch):
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        synchronize=MagicMock(),
    )
    fake_torch.cuda = fake_cuda
    fake_torch.zeros = MagicMock(return_value=1)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert cuda_usable(reset_cache=True) is True
    assert fake_cuda.synchronize.called


def test_cuda_usable_broken_kernel_image_cuda_error(monkeypatch: pytest.MonkeyPatch):
    """Simulates cudaErrorNoKernelImageForDevice (e.g. cu130 on GTX 1060 or cu126 on RTX 5060)."""
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        synchronize=MagicMock(),
    )
    fake_torch.cuda = fake_cuda

    def broken_zeros(*args, **kwargs):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    fake_torch.zeros = broken_zeros
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.warns(RuntimeWarning, match="no kernel image is available"):
        assert cuda_usable(reset_cache=True) is False


def test_cuda_usable_cuda_not_available(monkeypatch: pytest.MonkeyPatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert cuda_usable(reset_cache=True) is False


def test_cuda_usable_caching(monkeypatch: pytest.MonkeyPatch):
    fake_torch = types.ModuleType("torch")
    call_count = {"count": 0}

    def count_zeros(*args, **kwargs):
        call_count["count"] += 1
        return 1

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        synchronize=MagicMock(),
    )
    fake_torch.cuda = fake_cuda
    fake_torch.zeros = count_zeros
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert cuda_usable(reset_cache=True) is True
    assert call_count["count"] == 1

    # Second call uses cache
    assert cuda_usable(reset_cache=False) is True
    assert call_count["count"] == 1

    # Reset cache forces re-test
    assert cuda_usable(reset_cache=True) is True
    assert call_count["count"] == 2


def test_backend_detect_falls_back_when_kernel_broken(monkeypatch: pytest.MonkeyPatch):
    """When CUDA kernel fails, detect_backend() must fall back to torch-cpu (or ncnn)."""
    monkeypatch.setattr("video_upscaler.config.BACKEND_PREF", "auto")
    monkeypatch.setattr("video_upscaler.backend._tensorrt_available", lambda: False)
    monkeypatch.setattr("video_upscaler.backend._ncnn_tool_present", lambda: False)
    monkeypatch.setattr("video_upscaler.backend._torch_mps_available", lambda: False)

    # Force broken CUDA kernel
    monkeypatch.setattr("video_upscaler.backend.cuda_usable", lambda reset_cache=False: False)

    assert backend.detect_backend() == "torch-cpu"


def test_cugan_detect_device_falls_back_when_kernel_broken(monkeypatch: pytest.MonkeyPatch):
    from video_upscaler import cugan

    monkeypatch.setattr("video_upscaler.cugan.PREFERRED_DEVICE", "auto")
    monkeypatch.setattr("video_upscaler.cugan._mps_available", lambda torch: False)
    monkeypatch.setattr("video_upscaler.backend.cuda_usable", lambda reset_cache=False: False)

    assert cugan.detect_device() == "cpu"


def test_dedup_and_ma2_fall_back_when_kernel_broken(monkeypatch: pytest.MonkeyPatch):
    from video_upscaler import dedup_backend
    from video_upscaler.matanyone2 import backend as ma2_backend

    monkeypatch.setattr("video_upscaler.backend.cuda_usable", lambda reset_cache=False: False)

    assert dedup_backend._torch_cuda_available() is False
    assert ma2_backend._cuda_available() is False
