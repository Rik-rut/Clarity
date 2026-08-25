"""Tests for the benchmark mode (torch-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_upscaler.benchmark import (
    BASELINE_TIME,
    BENCHMARK_VIDEO,
    comparison_report,
    format_duration,
    fps,
    parse_args,
    resolve_backend,
    resolve_video,
    seconds_per_frame,
)


def test_format_duration() -> None:
    assert format_duration(45) == "45s"
    assert format_duration(16 * 60 + 37) == "16m 37s"
    assert format_duration(3725) == "62m 05s"


def test_comparison_report_matches_required_format() -> None:
    report = comparison_report(BASELINE_TIME, 600)
    assert "Baseline time: 16m 37s" in report
    assert "Optimized time: 10m 00s" in report
    assert "Absolute improvement: 6m 37s" in report
    assert "Speedup: 1.66x" in report
    assert "Percentage improvement: 39.8%" in report


def test_comparison_report_slower_case() -> None:
    report = comparison_report(997, 1200)
    assert "Speedup: 0.83x" in report
    assert "Percentage improvement: -20.4%" in report


def test_fps_and_seconds_per_frame() -> None:
    assert fps(10.0, 100) == 10.0
    assert seconds_per_frame(10.0, 100) == 0.1
    assert fps(0.0, 100) == 0.0
    assert seconds_per_frame(10.0, 0) == 0.0


def test_resolve_video_finds_benchmark_in_input_dir(tmp_path: Path) -> None:
    video = tmp_path / BENCHMARK_VIDEO
    video.write_bytes(b"fake")
    assert resolve_video(None, input_dir=tmp_path) == video


def test_resolve_video_explicit_path(tmp_path: Path) -> None:
    other = tmp_path / "other.mp4"
    other.write_bytes(b"fake")
    assert resolve_video(str(other), input_dir=tmp_path) == other


def test_resolve_video_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_video(None, input_dir=tmp_path)


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.backend == "auto"
    assert args.profile is None
    assert args.frames == 0
    assert args.batch == 1
    assert not args.pipeline


def test_parse_args_override() -> None:
    args = parse_args(
        ["--backend", "ncnn", "--frames", "200",
         "--batch", "4", "--pipeline", "--video", "clip.mp4"]
    )
    assert args.backend == "ncnn"
    assert args.frames == 200
    assert args.batch == 4
    assert args.pipeline
    assert args.video == "clip.mp4"


def test_resolve_backend_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video_upscaler.benchmark.detect_backend", lambda: "torch-cuda")
    assert resolve_backend("auto") == "torch-cuda"


def test_resolve_backend_auto_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video_upscaler.benchmark.detect_backend", lambda: "tensorrt")
    assert resolve_backend("auto") == "tensorrt"


def test_resolve_backend_torch_maps_to_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "video_upscaler.backend._torch_cuda_available", lambda: False
    )
    assert resolve_backend("torch") == "torch-cpu"


def test_resolve_backend_ncnn_forced() -> None:
    assert resolve_backend("ncnn") == "ncnn"


def test_resolve_backend_tensorrt_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "video_upscaler.tensorrt_backend":
            raise ImportError("no tensorrt yet")
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(RuntimeError, match="TensorRT is not installed"):
        resolve_backend("tensorrt")


def test_resolve_backend_tensorrt_missing_install_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_upscaler import tensorrt_backend

    monkeypatch.setattr(
        tensorrt_backend, "tensorrt_available", lambda: False
    )
    with pytest.raises(RuntimeError, match="install not found"):
        resolve_backend("tensorrt")


def test_load_records_empty_when_no_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from video_upscaler import benchmark

    monkeypatch.setattr(benchmark, "RESULT_LOG", tmp_path / "missing.json")
    assert benchmark.load_records() == []


def test_append_and_load_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from video_upscaler import benchmark

    log = tmp_path / "results.json"
    monkeypatch.setattr(benchmark, "RESULT_LOG", log)
    benchmark.append_record({"a": 1})
    benchmark.append_record({"a": 2})
    records = benchmark.load_records()
    assert [r["a"] for r in records] == [1, 2]
    data = json.loads(log.read_text(encoding="utf-8"))
    assert len(data) == 2


def test_load_records_ignores_corrupt_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import benchmark

    log = tmp_path / "results.json"
    log.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(benchmark, "RESULT_LOG", log)
    assert benchmark.load_records() == []
