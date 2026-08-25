"""Backend selection semantics mirror the AMT selector's contract."""

import pytest

from video_upscaler.matanyone2 import backend
from video_upscaler.matanyone2.backend import (
    BackendSelection,
    MatAnyone2BackendUnavailable,
    select_backend,
)


@pytest.fixture
def cuda_on(monkeypatch):
    monkeypatch.setattr(backend, "_cuda_available", lambda: True)


@pytest.fixture
def cuda_off(monkeypatch):
    monkeypatch.setattr(backend, "_cuda_available", lambda: False)


def test_auto_with_cuda_selects_pytorch_fp16(cuda_on):
    selection = select_backend("auto", "fp16")
    assert selection.name == "pytorch"
    assert selection.precision == "fp16"
    assert selection.device == "cuda"
    assert selection.fallback_reason is None


def test_auto_without_cuda_downgrades_precision(cuda_off):
    selection = select_backend("auto", "fp16")
    assert selection.name == "pytorch"
    assert selection.precision == "fp32"
    assert selection.device == "cpu"
    assert selection.fallback_reason is not None
    assert "CUDA" in selection.fallback_reason


def test_explicit_tensorrt_raises(cuda_on):
    with pytest.raises(MatAnyone2BackendUnavailable):
        select_backend("tensorrt", "fp16")


def test_explicit_pytorch_without_cuda_downgrades(cuda_off):
    selection = select_backend("pytorch", "fp16")
    assert selection.precision == "fp32"
    assert selection.device == "cpu"


def test_invalid_backend_raises(cuda_on):
    with pytest.raises(ValueError):
        select_backend("ncnn")


def test_invalid_precision_raises(cuda_on):
    with pytest.raises(ValueError):
        select_backend("auto", "int8")


def test_defaults_are_auto_fp16(cuda_on):
    selection = select_backend()
    assert (selection.name, selection.precision, selection.device) == (
        "pytorch", "fp16", "cuda",
    )


def test_session_protocol_methods():
    proto = backend.MattingSession
    for method in ("start", "step", "close"):
        assert hasattr(proto, method)
    assert isinstance(BackendSelection(name="pytorch", precision="fp16", device="cuda"), BackendSelection)
