"""Tests for the TensorRT backend module (torch/tensorrt-free paths)."""

from __future__ import annotations

from pathlib import Path

import pytest


def _fake_trt_root(tmp_path: Path) -> Path:
    install = tmp_path / "TensorRT-11.1.0.106"
    (install / "bin").mkdir(parents=True)
    (install / "bin" / "nvinfer_11.dll").touch()
    return install


def test_tensorrt_available_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from video_upscaler import tensorrt_backend as tb

    monkeypatch.setattr(tb, "_TRT_ENV_READY", False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorrt":
            raise ImportError("no tensorrt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert tb.tensorrt_available() is False


def test_tensorrt_available_missing_install(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_upscaler import tensorrt_backend as tb

    monkeypatch.setattr(tb, "_TRT_ENV_READY", False)
    monkeypatch.setattr(tb, "_trt_bin_dir", lambda: None)
    assert tb.tensorrt_available() is False


def test_prepare_env_sets_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os

    from video_upscaler import tensorrt_backend as tb

    install = _fake_trt_root(tmp_path)
    monkeypatch.setattr(tb, "_TRT_ENV_READY", False)
    monkeypatch.setattr(tb, "_trt_bin_dir", lambda: install / "bin")
    monkeypatch.setattr(tb, "_torch_lib_dir", lambda: None)
    monkeypatch.delenv("PATH", raising=False)

    assert tb.prepare_env() is None
    assert str(install / "bin") == os.environ["PATH"].split(os.pathsep)[0]
    assert tb.prepare_env() is None  # idempotent


def test_trt_site_packages_wheel_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys
    from video_upscaler import tensorrt_backend as tb

    site_dir = tmp_path / "Lib" / "site-packages" / "tensorrt_libs"
    site_dir.mkdir(parents=True)
    (site_dir / "nvinfer_11.dll").touch()

    monkeypatch.setattr(sys, "path", [str(tmp_path / "Lib" / "site-packages")])
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(tb, "_trt_install_dirs", lambda: [])

    found_dirs = tb._trt_site_packages_dirs()
    assert site_dir in found_dirs
    assert tb._trt_bin_dir() == site_dir


def test_prepare_env_error_without_install(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_upscaler import tensorrt_backend as tb

    monkeypatch.setattr(tb, "_TRT_ENV_READY", False)
    monkeypatch.setattr(tb, "_trt_bin_dir", lambda: None)
    error = tb.prepare_env()
    assert error is not None
    assert "TensorRT" in error


def test_engine_path_naming(monkeypatch: pytest.MonkeyPatch) -> None:
    from video_upscaler import tensorrt_backend as tb

    monkeypatch.setattr(tb, "gpu_name", lambda: "NVIDIA_GeForce_RTX_3050")
    path = tb.engine_path("up2x-latest-denoise2x.pth", 2)
    assert path.name == "up2x-latest-denoise2x_x2_fp16_NVIDIA_GeForce_RTX_3050.engine"
    assert path.parent.name == "tensorrt"


def test_metadata_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from video_upscaler import tensorrt_backend as tb

    monkeypatch.setattr(tb, "MODELS_DIR", tmp_path)
    entry = {"model": "m.pth", "scale": 2, "engine_version": 1}
    tb.save_metadata(entry)
    assert tb.load_metadata()["m.pth"] == entry
    tb.save_metadata({"model": "m.pth", "scale": 2, "engine_version": 2})
    assert tb.load_metadata()["m.pth"]["engine_version"] == 2


def test_ensure_engine_rebuilds_on_stale_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import tensorrt_backend as tb

    monkeypatch.setattr(tb, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(tb, "gpu_name", lambda: "GPU_X")
    model_file = tmp_path / "m.pth"
    model_file.write_bytes(b"weights")
    monkeypatch.setattr(tb, "_trt_version", lambda: "11.1")
    monkeypatch.setattr(tb, "_cuda_version", lambda: "12.9")

    engine = tmp_path / "tensorrt" / "m_x2_fp16_GPU_X.engine"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"old")
    # Stale metadata (wrong hash) must trigger a rebuild.
    tb.save_metadata({"model": "m.pth", "scale": 2, "engine_version": 2,
                      "model_hash": "deadbeef", "gpu": "GPU_X",
                      "tensorrt": "11.1", "cuda": "12.9"})
    built = []

    def fake_build(model_name, scale, **_):
        built.append((model_name, scale))
        return engine

    monkeypatch.setattr(tb, "build_engine", fake_build)
    assert tb.ensure_engine("m.pth", 2) == engine
    assert built == [("m.pth", 2)]


def test_tile_grid_1080p_landscape() -> None:
    from video_upscaler.tensorrt_backend import _tile_grid

    crop_h, crop_w, ph, pw = _tile_grid(1080, 1920)
    assert (crop_h, crop_w) == (1080, 960)
    assert (ph, pw) == (1080, 1920)
    # Padded size: ph+36 x pw+36; tiles cover (h-36) x (w-36) exactly.
    h, w = ph + 36, pw + 36
    assert (h - 36) % crop_h == 0
    assert (w - 36) % crop_w == 0
    tiles_h = (h - 36) // crop_h
    tiles_w = (w - 36) // crop_w
    assert (tiles_h, tiles_w) == (1, 2)


def test_tile_grid_portrait() -> None:
    from video_upscaler.tensorrt_backend import _tile_grid

    crop_h, crop_w, ph, pw = _tile_grid(1920, 1080)
    assert (crop_h, crop_w) == (960, 1080)
    assert (ph, pw) == (1920, 1080)


def test_tile_grid_odd_sizes() -> None:
    from video_upscaler.tensorrt_backend import _tile_grid

    crop_h, crop_w, ph, pw = _tile_grid(1081, 1921)
    # crop sizes stay even; padded dims are exact multiples of the crops
    # and never smaller than the source.
    assert crop_h % 2 == 0
    assert crop_w % 2 == 0
    assert ph >= 1081 and ph % crop_h == 0
    assert pw >= 1921 and pw % crop_w == 0
    assert ph - 1081 < crop_h
    assert pw - 1921 < crop_w


def test_tile_grid_small_frame_is_single_tile() -> None:
    from video_upscaler.tensorrt_backend import _tile_grid

    crop_h, crop_w, ph, pw = _tile_grid(256, 384)
    assert (crop_h, crop_w) == (256, 192)
    assert (ph, pw) == (256, 384)
    assert (ph + 36 - 36) // crop_h == 1
    assert (pw + 36 - 36) // crop_w == 2


def test_ensure_engine_reuses_valid_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import tensorrt_backend as tb

    monkeypatch.setattr(tb, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(tb, "gpu_name", lambda: "GPU_X")
    model_file = tmp_path / "m.pth"
    model_file.write_bytes(b"weights")

    current = {
        "model": "m.pth",
        "scale": 2,
        "precision": "fp16",
        "model_hash": tb._model_hash(model_file),
        "gpu": "GPU_X",
        "tensorrt": "11.1",
        "cuda": "12.9",
        "engine_version": 2,
    }
    tb.save_metadata(current)
    engine = tmp_path / "tensorrt" / "m_x2_fp16_GPU_X.engine"
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(b"engine")

    monkeypatch.setattr(tb, "_trt_version", lambda: "11.1")
    monkeypatch.setattr(tb, "_cuda_version", lambda: "12.9")
    called = []

    def fake_build(*args, **kwargs):
        called.append(args)
        return engine

    monkeypatch.setattr(tb, "build_engine", fake_build)
    assert tb.ensure_engine("m.pth", 2) == engine
    assert called == []
