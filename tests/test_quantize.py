"""Torch-free tests for the one-time fp16 weight cache."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from video_upscaler import quantize
from video_upscaler.config import MODELS_DIR


def test_fp16_path_suffix() -> None:
    assert quantize.fp16_path("up2x-latest-denoise2x.pth") == (
        MODELS_DIR / "up2x-latest-denoise2x.fp16.pth"
    )
    assert quantize.fp16_path("up4x-latest-conservative.pth") == (
        MODELS_DIR / "up4x-latest-conservative.fp16.pth"
    )


def test_is_fresh(tmp_path: Path) -> None:
    source = tmp_path / "m.pth"
    cache = tmp_path / "m.fp16.pth"
    source.write_bytes(b"x")
    assert not quantize.is_fresh(cache, source)
    cache.write_bytes(b"y")
    # same-second mtime tolerance: bump cache mtime ahead
    newer = source.stat().st_mtime_ns + 10_000_000
    import os

    os.utime(cache, ns=(newer, newer))
    assert quantize.is_fresh(cache, source)


def test_ensure_fp16_skips_conversion_when_fresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("video_upscaler.quantize.MODELS_DIR", tmp_path)
    source = tmp_path / "up2x-latest-denoise2x.pth"
    cache = quantize.fp16_path("up2x-latest-denoise2x.pth")
    source.write_bytes(b"weights")
    cache.write_bytes(b"weights")
    import os

    newer = source.stat().st_mtime_ns + 10_000_000
    os.utime(cache, ns=(newer, newer))

    def _must_not_run(source_path, dest_path):
        raise AssertionError("conversion must be skipped when cache is fresh")

    monkeypatch.setattr("video_upscaler.quantize.convert_to_fp16", _must_not_run)
    assert quantize.ensure_fp16("up2x-latest-denoise2x.pth") == cache


def test_convert_to_fp16_casts_float32_tensors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    saved = {}

    class _FakeTensor:
        def __init__(self, dtype):
            self.dtype = dtype
            self.half_called = False

        def half(self):
            self.half_called = True
            return self

    class _FakeTorch:
        def load(self, path, map_location=None):
            return {
                "params": {"w": _FakeTensor("torch.float32"), "b": _FakeTensor("torch.float32")},
                "meta": 123,  # non-tensor values must survive untouched
            }

        def save(self, payload, path):
            Path(path).write_bytes(b"saved")  # real torch.save creates the file
            saved["payload"] = payload
            saved["path"] = path

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    source = tmp_path / "m.pth"
    dest = tmp_path / "m.fp16.pth"
    quantize.convert_to_fp16(source, dest)

    assert str(saved["path"]).endswith(".tmp")
    assert not Path(saved["path"]).exists()  # tmp cleaned up after os.replace
    assert dest.is_file()
    params = saved["payload"]["params"]
    assert params["w"].half_called
    assert params["b"].half_called
    assert saved["payload"]["meta"] == 123
