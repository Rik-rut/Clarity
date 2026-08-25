"""Focused tests for the AMT benchmark and baseline record."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import time

import numpy as np
import pytest

from video_upscaler.amt_benchmark import (
    _parse_resolution,
    _run_inference,
    _timed_engine,
    _validate,
    main,
    output_frame_count,
    parse_args,
    realtime_factor,
    run_benchmark,
)


@pytest.mark.parametrize(
    ("source_frames", "factor", "expected"),
    [(1, 2, 1), (5, 2, 9), (5, 4, 17), (5, 8, 33), (0, 2, 0)],
)
def test_output_frame_count(source_frames: int, factor: int, expected: int) -> None:
    assert output_frame_count(source_frames, factor) == expected


def test_parse_args_accepts_amt_benchmark_options() -> None:
    args = parse_args(
        [
            "--input",
            "clip.mp4",
            "--mode",
            "4x",
            "--backend",
            "pytorch-fp16",
            "--device",
            "cuda",
            "--warmup",
            "3",
            "--iterations",
            "2",
            "--batch-size",
            "2",
            "--frames",
            "120",
            "--resolution",
            "1920x1080",
            "--pipeline",
        ]
    )

    assert args.input == "clip.mp4"
    assert args.mode == "4x"
    assert args.backend == "pytorch-fp16"
    assert args.device == "cuda"
    assert args.warmup == 3
    assert args.iterations == 2
    assert args.batch_size == 2
    assert args.frames == 120
    assert args.resolution == "1920x1080"
    assert args.pipeline
    assert args.model == "AMT-S"  # default


def test_parse_args_accepts_amt_model_choice() -> None:
    args = parse_args(
        ["--input", "clip.mp4", "--model", "AMT-L", "--backend", "tensorrt-fp16"]
    )
    assert args.model == "AMT-L"


def test_parse_args_rejects_unsupported_amt_model() -> None:
    import pytest as _pytest

    with _pytest.raises(SystemExit):
        parse_args(["--input", "clip.mp4", "--model", "AMT-X"])


def test_realtime_factor_uses_input_duration_over_processing_time() -> None:
    assert realtime_factor(10.0, 2.5) == 4.0
    assert realtime_factor(10.0, 0.0) == 0.0


def test_run_benchmark_returns_json_compatible_amt_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import amt_benchmark

    source = tmp_path / "clip.mp4"
    source.touch()
    frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(3)]

    class FakeEngine:
        device = "cpu"

        def interpolate(self, input_frames: list[np.ndarray], niters: int):
            return iter(input_frames)

    monkeypatch.setattr(
        amt_benchmark,
        "probe",
        lambda path: {
            "width": 3,
            "height": 2,
            "fps": 30.0,
            "duration": 0.1,
            "codec_name": "h264",
            "has_audio": True,
            "rotation": 0,
        },
    )
    monkeypatch.setattr(amt_benchmark, "decode_frames", lambda path: iter(frames))
    monkeypatch.setattr(amt_benchmark, "AMTInterpEngine", lambda model: FakeEngine())

    record = run_benchmark(
        Namespace(
            input=str(source),
            model="AMT-S",
            mode="2x",
            backend="pytorch",
            device="cpu",
            warmup=0,
            iterations=1,
            batch_size=1,
            frames=0,
            resolution=None,
            pipeline=False,
        )
    )

    assert record["source_frames"] == 3
    assert record["output_frames"] == 5
    assert record["input_fps"] == 30.0
    assert record["output_fps"] == 30.0
    assert record["backend"] == "pytorch"
    assert record["precision"] == "fp32"
    assert record["batch_size"] == 1
    assert record["model"] == "AMT-S"
    for field in (
        "load_time_s",
        "warmup_time_s",
        "inference_time_s",
        "decode_time_s",
        "encode_time_s",
        "h2d_time_s",
        "d2h_time_s",
        "total_time_s",
        "inference_fps",
        "realtime_factor",
        "peak_vram_mib",
        "gpu_utilization_percent",
    ):
        assert field in record


def test_timed_engine_exposes_numeric_transfer_timings() -> None:
    from video_upscaler import amt_benchmark

    class TimedFakeEngine:
        device = "cpu"

        def __init__(self) -> None:
            self.timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}

        def timing_snapshot(self) -> dict[str, float]:
            return dict(self.timings)

        def interpolate(self, frames, niters):
            self.timings["h2d_time_s"] += 0.125
            self.timings["d2h_time_s"] += 0.25
            return iter(frames)

    timing = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
    engine = TimedFakeEngine()
    wrapped = _timed_engine(engine, "pytorch", timing)

    list(wrapped.interpolate([np.zeros((2, 2, 3), dtype=np.uint8)], 1))

    assert timing["h2d_time_s"] == 0.125
    assert timing["d2h_time_s"] == 0.25


def test_timed_engine_counts_engines_that_reset_each_window() -> None:
    class ResettingFakeEngine:
        device = "cpu"

        def __init__(self) -> None:
            self.timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}

        def reset_timing(self) -> None:
            self.timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}

        def timing_snapshot(self) -> dict[str, float]:
            return dict(self.timings)

        def interpolate(self, frames, niters):
            self.reset_timing()
            self.timings["h2d_time_s"] = 0.1
            self.timings["d2h_time_s"] = 0.2
            return iter(frames)

    timing = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
    wrapped = _timed_engine(ResettingFakeEngine(), "pytorch", timing)

    list(wrapped.interpolate([np.zeros((2, 2, 3), dtype=np.uint8)], 1))
    list(wrapped.interpolate([np.zeros((2, 2, 3), dtype=np.uint8)], 1))

    assert timing["h2d_time_s"] == 0.2
    assert timing["d2h_time_s"] == 0.4


def test_scheduler_timing_adapter_delegates_backend_transfer_metrics() -> None:
    from video_upscaler import amt_benchmark

    class TimedBackend:
        device = "cuda"
        name = "fake"
        precision = "fp16"

        def __init__(self) -> None:
            self.enabled = False
            self.timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}

        def set_timing_enabled(self, enabled: bool) -> None:
            self.enabled = enabled

        def timing_snapshot(self) -> dict[str, float]:
            return dict(self.timings)

        def prepare_frames(self, frames):
            return frames

        def infer_batch(self, frame_a, frame_b):
            if self.enabled:
                self.timings["h2d_time_s"] += 0.125
            return frame_a

        def transfer_batch_to_host(self, output):
            if self.enabled:
                self.timings["d2h_time_s"] += 0.25
            return output

        def finalize_frames(self, frames):
            return frames

    timing = {"inference_time_s": 0.0, "h2d_time_s": 0.0, "d2h_time_s": 0.0}
    wrapped = amt_benchmark._timed_scheduler_backend(TimedBackend(), timing)

    wrapped.infer_batch("a", "b")
    wrapped.transfer_batch_to_host("output")

    assert timing["h2d_time_s"] == 0.125
    assert timing["d2h_time_s"] == 0.25


def test_scheduler_timing_adapter_includes_async_transfer_inference_interval() -> None:
    from video_upscaler import amt_benchmark

    class AsyncBackend:
        device = "cuda"
        name = "fake"
        precision = "fp16"

        def __init__(self) -> None:
            self.enabled = False
            self.timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
            self.transfer_started = None
            self.transfer_elapsed = 0.0

        def set_timing_enabled(self, enabled: bool) -> None:
            self.enabled = enabled

        def timing_snapshot(self) -> dict[str, float]:
            return dict(self.timings)

        def prepare_frames(self, frames):
            return frames

        def infer_batch(self, frame_a, frame_b):
            assert self.enabled
            self.timings["h2d_time_s"] += 0.1
            return frame_a

        def transfer_batch_to_host(self, output):
            assert self.enabled
            started = time.perf_counter()
            time.sleep(0.01)
            self.transfer_elapsed = time.perf_counter() - started
            self.timings["d2h_time_s"] += 0.2
            return output

        def finalize_frames(self, frames):
            return frames

    timing = {"inference_time_s": 0.0, "h2d_time_s": 0.0, "d2h_time_s": 0.0}
    backend = AsyncBackend()
    wrapped = amt_benchmark._timed_scheduler_backend(backend, timing)

    wrapped.infer_batch("a", "b")
    wrapped.transfer_batch_to_host("output")

    assert timing["inference_time_s"] >= backend.transfer_elapsed
    assert timing["h2d_time_s"] == 0.1
    assert timing["d2h_time_s"] == 0.2


def test_benchmark_enables_transfer_timing_only_around_inference() -> None:
    transitions: list[bool] = []

    class ToggleFakeEngine:
        device = "cuda"

        def set_timing_enabled(self, enabled: bool) -> None:
            transitions.append(enabled)

        def interpolate(self, frames, niters):
            return iter(frames)

    timing = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
    wrapped = _timed_engine(ToggleFakeEngine(), "pytorch", timing)

    list(wrapped.interpolate([np.zeros((2, 2, 3), dtype=np.uint8)], 1))

    assert transitions == [True, False]


def test_encode_finalize_timing_includes_stdin_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import ffmpeg

    class Clock:
        value = 0.0

        def now(self) -> float:
            return self.value

    clock = Clock()

    class FakeStdin:
        def write(self, data: bytes) -> None:
            pass

        def close(self) -> None:
            clock.value += 5.0

    class FakeStderr:
        def read(self) -> bytes:
            return b""

        def close(self) -> None:
            pass

    class FakeProcess:
        stdin = FakeStdin()
        stderr = FakeStderr()

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(ffmpeg, "ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(ffmpeg, "nvenc_available", lambda: False)
    monkeypatch.setattr(ffmpeg.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(ffmpeg.time, "perf_counter", clock.now)

    timing: dict[str, float] = {}
    ffmpeg.encode_video(
        [b"\x00\x00\x00"],
        1,
        1,
        30.0,
        tmp_path / "source.mp4",
        tmp_path / "output.mp4",
        False,
        timing=timing,
    )

    assert timing["encode_finalize_time_s"] >= 5.0


def test_pipeline_uses_direct_encode_hook_and_numeric_timings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import amt_benchmark

    source = tmp_path / "clip.mp4"
    source.touch()
    raw_frames = [bytes(2 * 2 * 3) for _ in range(2)]

    class FakeEngine:
        device = "cpu"

        def __init__(self) -> None:
            self.timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}

        def timing_snapshot(self) -> dict[str, float]:
            return dict(self.timings)

        def interpolate(self, frames, niters):
            self.timings["h2d_time_s"] += 0.1
            self.timings["d2h_time_s"] += 0.2
            return iter(frames)

    class FakeSampler:
        peak_mib = 0
        peak_util = 0

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    def fake_encode(frames, *args, timing=None, **kwargs):
        list(frames)
        timing.update(
            {
                "encode_time_s": 0.25,
                "encode_write_time_s": 0.1,
                "encode_finalize_time_s": 0.15,
            }
        )

    monkeypatch.setattr(amt_benchmark, "AMTInterpEngine", lambda model: FakeEngine())
    monkeypatch.setattr(amt_benchmark, "_NvidiaSmiSampler", FakeSampler)
    monkeypatch.setattr(amt_benchmark, "encode_video", fake_encode)
    monkeypatch.setattr(amt_benchmark, "decode_frames", lambda path: iter(raw_frames))
    monkeypatch.setattr(
        amt_benchmark,
        "probe",
        lambda path: {
            "width": 2,
            "height": 2,
            "fps": 30.0,
            "duration": 0.1,
            "codec_name": "h264",
            "has_audio": False,
            "rotation": 0,
        },
    )
    monkeypatch.setattr(amt_benchmark, "_software_versions", lambda: {})
    monkeypatch.setattr(amt_benchmark, "gpu_name", lambda: "fake-gpu")

    record = run_benchmark(
        Namespace(
            input=str(source),
            model="AMT-S",
            mode="2x",
            backend="pytorch",
            device="cpu",
            warmup=0,
            iterations=1,
            batch_size=1,
            frames=2,
            resolution=None,
            pipeline=True,
        )
    )

    assert record["h2d_time_s"] == 0.2
    assert record["d2h_time_s"] == 0.4
    assert record["encode_time_s"] == 0.25
    assert record["encode_write_time_s"] == 0.1
    assert record["encode_finalize_time_s"] == 0.15
    assert all(
        record[field] >= 0.0
        for field in (
            "h2d_time_s",
            "d2h_time_s",
            "encode_time_s",
            "encode_write_time_s",
            "encode_finalize_time_s",
        )
    )


@pytest.mark.parametrize(("mode", "expected_niters"), [("2x", 1), ("4x", 2), ("8x", 3)])
def test_run_benchmark_passes_mode_niters_to_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, expected_niters: int
) -> None:
    from video_upscaler import amt_benchmark

    source = tmp_path / "clip.mp4"
    source.touch()
    calls: list[int] = []

    class FakeEngine:
        device = "cpu"

        def interpolate(self, frames, niters):
            calls.append(niters)
            return iter(frames)

    monkeypatch.setattr(amt_benchmark, "AMTInterpEngine", lambda model: FakeEngine())
    monkeypatch.setattr(amt_benchmark, "_NvidiaSmiSampler", type("S", (), {
        "peak_mib": 0,
        "peak_util": 0,
        "start": lambda self: None,
        "stop": lambda self: None,
    }))
    monkeypatch.setattr(amt_benchmark, "decode_frames", lambda path: iter([bytes(12)] * 2))
    monkeypatch.setattr(
        amt_benchmark,
        "probe",
        lambda path: {"width": 2, "height": 2, "fps": 30.0, "duration": 0.1},
    )
    monkeypatch.setattr(amt_benchmark, "_software_versions", lambda: {})
    monkeypatch.setattr(amt_benchmark, "gpu_name", lambda: "fake-gpu")

    run_benchmark(
        Namespace(
            input=str(source), model="AMT-S", mode=mode, backend="pytorch", device="cpu",
            warmup=0, iterations=1, batch_size=1, frames=0,
            resolution=None, pipeline=False,
        )
    )

    assert calls == [expected_niters]


def test_fp16_requires_cuda_before_engine_load() -> None:
    with pytest.raises(RuntimeError, match="requires --device cuda"):
        _validate(
            Namespace(
                backend="pytorch-fp16",
                device="cpu",
                warmup=0,
                iterations=1,
                batch_size=1,
                frames=0,
                mode="2x",
                resolution=None,
            )
        )


def test_parse_args_rejects_invalid_resolution() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--input", "clip.mp4", "--resolution", "not-a-resolution"])
    assert _parse_resolution("1280x720") == (1280, 720)


def test_cli_stdout_is_one_json_object(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from video_upscaler import amt_benchmark

    def fake_run(args):
        print("diagnostic", flush=True)
        return {"status": "ok"}

    monkeypatch.setattr(amt_benchmark, "run_benchmark", fake_run)

    assert main(["--input", "clip.mp4"]) == 0
    captured = capsys.readouterr()

    assert json.loads(captured.out) == {"status": "ok"}
    assert captured.err == "diagnostic\n"
