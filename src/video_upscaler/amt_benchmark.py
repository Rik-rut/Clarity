"""Reproducible AMT interpolation benchmark and baseline record."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from video_upscaler.benchmark import _NvidiaSmiSampler, gpu_name
from video_upscaler.ffmpeg import decode_frames, encode_video, probe
from video_upscaler.interp import (
    AMTBackendFactory,
    AMTBackendSelection,
    AMTInterpEngine,
)
from video_upscaler.amt_scheduler import AMTFrameScheduler
from video_upscaler.processor import _interp_window_stream

_NITERS_BY_MODE = {"2x": 1, "4x": 2, "8x": 3}
_PRECISION_BY_BACKEND = {
    "pytorch": "fp32",
    "pytorch-fp16": "fp16",
    "tensorrt-fp16": "fp16",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the AMT benchmark command-line options."""
    parser = argparse.ArgumentParser(
        prog="python -m video_upscaler.amt_benchmark",
        description="Benchmark the AMT-S/AMT-L/AMT-G interpolation pipeline.",
    )
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument(
        "--model",
        choices=["AMT-S", "AMT-L", "AMT-G"],
        default="AMT-S",
        help="AMT model to benchmark (default: AMT-S).",
    )
    parser.add_argument("--mode", choices=sorted(_NITERS_BY_MODE), default="2x")
    parser.add_argument(
        "--backend",
        choices=sorted(_PRECISION_BY_BACKEND),
        default="pytorch",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument(
        "--resolution",
        default=None,
        type=_resolution_argument,
        help="Optional benchmark resolution in WIDTHxHEIGHT form.",
    )
    parser.add_argument("--pipeline", action="store_true")
    return parser.parse_args(argv)


def output_frame_count(source_frames: int, factor: int) -> int:
    """Return AMT's recursive output count for a source frame sequence."""
    if source_frames <= 0:
        return 0
    if factor not in (2, 4, 8):
        raise ValueError("factor must be one of 2, 4, or 8")
    return factor * (source_frames - 1) + 1


def realtime_factor(input_duration_s: float, processing_time_s: float) -> float:
    """Return input duration divided by measured processing time."""
    if input_duration_s <= 0 or processing_time_s <= 0:
        return 0.0
    return input_duration_s / processing_time_s


def _parse_resolution(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (ValueError, AttributeError):
        raise ValueError("resolution must use WIDTHxHEIGHT, for example 1920x1080") from None
    if width <= 0 or height <= 0:
        raise ValueError("resolution dimensions must be positive")
    return width, height


def _resolution_argument(value: str) -> str:
    try:
        _parse_resolution(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _resize_frame(frame: np.ndarray, resolution: tuple[int, int] | None) -> np.ndarray:
    if resolution is None or frame.shape[1::-1] == resolution:
        return frame
    try:
        import cv2
    except ImportError:
        raise RuntimeError("--resolution requires OpenCV, which is not installed") from None
    width, height = resolution
    return np.ascontiguousarray(cv2.resize(frame, (width, height)))


def _read_frames(
    path: Path,
    info: dict,
    limit: int,
    resolution: tuple[int, int] | None,
) -> tuple[list[np.ndarray], float]:
    started = time.perf_counter()
    frames: list[np.ndarray] = []
    for raw in decode_frames(path):
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            info["height"], info["width"], 3
        )
        frames.append(_resize_frame(frame, resolution))
        if limit and len(frames) >= limit:
            break
    return frames, time.perf_counter() - started


def _timed_raw_frames(
    path: Path,
    info: dict,
    limit: int,
    resolution: tuple[int, int] | None,
    timing: dict[str, float],
) -> Iterator[bytes]:
    source = iter(decode_frames(path))
    count = 0
    while not limit or count < limit:
        started = time.perf_counter()
        try:
            raw = next(source)
        except StopIteration:
            return
        timing["decode_time_s"] += time.perf_counter() - started
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            info["height"], info["width"], 3
        )
        yield _resize_frame(frame, resolution).tobytes()
        count += 1


def _build_engine(device: str, precision: str = "fp32", model_key: str = "AMT-S"):
    from video_upscaler import cugan

    previous_device = cugan.PREFERRED_DEVICE
    cugan.PREFERRED_DEVICE = device
    try:
        if precision == "fp32":
            return AMTInterpEngine(model_key)
        return AMTInterpEngine(model_key, precision=precision)
    finally:
        cugan.PREFERRED_DEVICE = previous_device


def _build_amt_backend_factory(args: argparse.Namespace):
    """Build the benchmark's AMT factory from its explicit backend option."""
    if args.backend == "tensorrt-fp16":
        selection = AMTBackendSelection("tensorrt", "fp16", args.batch_size)
        return AMTBackendFactory(args.model, selection)

    precision = "fp16" if args.backend == "pytorch-fp16" else "fp32"

    class _BenchmarkFactory:
        selection = AMTBackendSelection("pytorch", precision, args.batch_size)

        def build(self, frame_shape=None):
            return _build_engine(args.device, precision, args.model)

        def close(self):
            return None

    return _BenchmarkFactory()


def _supports_scheduler(engine) -> bool:
    return all(
        hasattr(engine, method)
        for method in (
            "prepare_frames",
            "infer_batch",
            "transfer_batch_to_host",
            "finalize_frames",
        )
    )


def _timed_scheduler_backend(engine, timing: dict[str, float]):
    """Delegate scheduler calls while measuring the full CUDA batch interval.

    ``infer_batch`` only enqueues async CUDA work; execution completes inside
    ``transfer_batch_to_host`` (which synchronizes). ``inference_time_s`` is
    therefore counted from the batch enqueue to the host-transfer completion,
    while H2D/D2H counters keep the backend's own transfer measurements.
    """
    timing.setdefault("inference_time_s", 0.0)
    timing.setdefault("h2d_time_s", 0.0)
    timing.setdefault("d2h_time_s", 0.0)

    def snapshot() -> dict[str, float]:
        if hasattr(engine, "timing_snapshot"):
            return engine.timing_snapshot()
        return dict(getattr(engine, "timings", {}))

    def enable_timing(enabled: bool) -> None:
        setter = getattr(engine, "set_timing_enabled", None)
        if setter is not None:
            setter(enabled)

    def accumulate(key: str, before: dict[str, float], after: dict[str, float]) -> None:
        measured = after.get(key, 0.0) - before.get(key, 0.0)
        if measured < 0:
            measured = after.get(key, 0.0)
        timing[key] += max(0.0, measured)

    class TimedSchedulerBackend:
        name = getattr(engine, "name", "amt")
        precision = getattr(engine, "precision", "fp32")
        device = getattr(engine, "device", "cpu")
        _infer_started: float | None = None

        def prepare_frames(self, frames):
            return engine.prepare_frames(frames)

        def infer_batch(self, frame_a, frame_b):
            before = snapshot()
            enable_timing(True)
            started = time.perf_counter()
            try:
                result = engine.infer_batch(frame_a, frame_b)
            finally:
                enable_timing(False)
            self._infer_started = started
            accumulate("h2d_time_s", before, snapshot())
            return result

        def transfer_batch_to_host(self, batch_output):
            before = snapshot()
            enable_timing(True)
            started = time.perf_counter()
            try:
                result = engine.transfer_batch_to_host(batch_output)
            finally:
                enable_timing(False)
            ended = time.perf_counter()
            if self._infer_started is not None:
                timing["inference_time_s"] += ended - self._infer_started
                self._infer_started = None
            accumulate("d2h_time_s", before, snapshot())
            return result

        def finalize_frames(self, frames):
            return engine.finalize_frames(frames)

        def close(self):
            return engine.close()

    return TimedSchedulerBackend()


def _scheduled_inference(engine, frames, niters: int, batch_size: int):
    if not _supports_scheduler(engine):
        return _run_inference(engine, frames, niters, "pytorch")
    return list(
        AMTFrameScheduler().interpolate_window(
            frames, niters, engine, batch_size
        )
    )


def _run_inference(
    engine,
    frames: list[np.ndarray],
    niters: int,
    backend: str,
) -> list[np.ndarray]:
    if backend == "pytorch-fp16" and getattr(engine, "device", "cpu") == "cpu":
        raise RuntimeError("pytorch-fp16 requires --device cuda")
    if backend == "pytorch-fp16" and getattr(engine, "device", "cpu") == "cuda":
        import torch

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return list(engine.interpolate(frames, niters))
    return list(engine.interpolate(frames, niters))


def _timed_engine(engine, backend: str, timing: dict[str, float]):
    timing.setdefault("inference_time_s", 0.0)
    timing.setdefault("h2d_time_s", 0.0)
    timing.setdefault("d2h_time_s", 0.0)

    def snapshot() -> dict[str, float]:
        if hasattr(engine, "timing_snapshot"):
            return engine.timing_snapshot()
        return dict(getattr(engine, "timings", {}))

    resets_per_call = hasattr(engine, "reset_timing")

    class TimedEngine:
        device = getattr(engine, "device", "cpu")

        def interpolate(self, frames: list[np.ndarray], niters: int):
            before = snapshot()
            if resets_per_call:
                engine.reset_timing()
            started = time.perf_counter()
            set_timing_enabled = getattr(engine, "set_timing_enabled", None)
            if set_timing_enabled is not None:
                set_timing_enabled(True)
            try:
                output = _run_inference(engine, frames, niters, backend)
            finally:
                if set_timing_enabled is not None:
                    set_timing_enabled(False)
            timing["inference_time_s"] += time.perf_counter() - started
            after = snapshot()
            for key in ("h2d_time_s", "d2h_time_s"):
                measured = after.get(key, 0.0)
                if not resets_per_call:
                    measured -= before.get(key, 0.0)
                timing[key] += max(0.0, measured)
            return output

    return TimedEngine()


def _torch_stats() -> tuple[int, str | None, str | None]:
    try:
        import torch
    except ImportError:
        return 0, None, None
    peak = 0
    if torch.cuda.is_available():
        peak = int(torch.cuda.max_memory_allocated(0) // (1 << 20))
    return peak, getattr(torch, "__version__", None), getattr(torch.version, "cuda", None)


def _package_version(name: str) -> str | None:
    try:
        return str(importlib.import_module(name).__version__)
    except (ImportError, AttributeError, OSError):
        return None


def _software_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "pytorch": _torch_stats()[1],
        "cuda": _torch_stats()[2],
        "tensorrt": _package_version("tensorrt"),
        "triton": _package_version("triton"),
    }


def _validate(args: argparse.Namespace) -> tuple[int, tuple[int, int] | None]:
    if args.warmup < 0 or args.iterations < 1 or args.batch_size < 1 or args.frames < 0:
        raise ValueError("warmup and frames must be >= 0; iterations and batch-size must be >= 1")
    if args.backend == "pytorch-fp16" and args.device != "cuda":
        raise RuntimeError("pytorch-fp16 requires --device cuda")
    return _NITERS_BY_MODE[args.mode], _parse_resolution(args.resolution)


def _record_base(
    args: argparse.Namespace,
    input_path: Path,
    info: dict,
    factor: int,
    frames: int,
    output_resolution: tuple[int, int],
) -> dict:
    output_frames = output_frame_count(frames, factor)
    output_fps = float(info["fps"])
    input_duration = float(info.get("duration") or 0.0)
    if args.frames:
        input_duration = frames / output_fps if output_fps else 0.0
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": str(input_path),
        "mode": args.mode,
        "factor": factor,
        "model": args.model,
        "backend": args.backend,
        "precision": _PRECISION_BY_BACKEND[args.backend],
        "device": args.device,
        "batch_size": args.batch_size,
        "requested_resolution": args.resolution,
        "source_frames": frames,
        "output_frames": output_frames,
        "input_fps": float(info["fps"]),
        "output_fps": output_fps,
        "input_duration_s": input_duration,
        "output_duration_s": output_frames / output_fps if output_fps else 0.0,
        "input_resolution": f"{info['width']}x{info['height']}",
        "output_resolution": f"{output_resolution[0]}x{output_resolution[1]}",
        "source_codec": info.get("codec_name"),
        "has_audio": bool(info.get("has_audio")),
        "software": _software_versions(),
        "gpu": gpu_name(),
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the AMT benchmark and return a JSON-compatible result record."""
    niters, resolution = _validate(args)
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    total_started = time.perf_counter()
    info = probe(input_path)
    load_started = time.perf_counter()
    factory = _build_amt_backend_factory(args)
    shape = resolution or (info["width"], info["height"])
    engine = factory.build((shape[1], shape[0]))
    load_time = time.perf_counter() - load_started
    frames: list[np.ndarray] = []
    decode_time = 0.0
    if not args.pipeline:
        frames, decode_time = _read_frames(input_path, info, args.frames, resolution)
        if not frames:
            raise RuntimeError("No frames decoded from the input video.")

    warmup_time = 0.0
    if args.warmup:
        warmup_frames = [np.zeros((shape[1], shape[0], 3), dtype=np.uint8)] * 2
        warmup_started = time.perf_counter()
        for _ in range(args.warmup):
            _scheduled_inference(
                engine, warmup_frames, niters, factory.selection.batch_size
            )
        warmup_time = time.perf_counter() - warmup_started

    sampler = _NvidiaSmiSampler()
    sampler.start()
    inference_time = 0.0
    encode_time = 0.0
    encode_write_time = 0.0
    encode_finalize_time = 0.0
    h2d_time = 0.0
    d2h_time = 0.0
    generated_frames = 0

    if args.pipeline:
        pipeline_timings = []
        with tempfile.TemporaryDirectory(prefix="clarity_amt_benchmark_") as temp_dir:
            for _ in range(args.iterations):
                timing = {"decode_time_s": 0.0, "inference_time_s": 0.0}
                use_scheduler = _supports_scheduler(engine)
                timed_engine = (
                    _timed_scheduler_backend(engine, timing)
                    if use_scheduler
                    else _timed_engine(engine, args.backend, timing)
                )
                generated = 0

                def output_stream() -> Iterator[np.ndarray]:
                    nonlocal generated
                    raw_frames = _timed_raw_frames(
                        input_path, info, args.frames, resolution, timing
                    )
                    for frame in _interp_window_stream(
                        raw_frames,
                        timed_engine,
                        niters,
                        max(1, args.frames or 120),
                        (resolution or (info["width"], info["height"]))[::-1],
                        scheduler=AMTFrameScheduler() if use_scheduler else None,
                        batch_size=factory.selection.batch_size,
                    ):
                        generated += 1
                        yield frame

                output_path = Path(temp_dir) / f"iteration-{len(pipeline_timings)}.mp4"
                encode_timing: dict[str, float] = {}
                encode_started = time.perf_counter()
                encode_video(
                    output_stream(),
                    resolution[0] if resolution else info["width"],
                    resolution[1] if resolution else info["height"],
                    info["fps"],
                    input_path,
                    output_path,
                    bool(info.get("has_audio")),
                    rotation=int(info.get("rotation") or 0),
                    timing=encode_timing,
                )
                pipeline_elapsed = time.perf_counter() - encode_started
                timing.update(encode_timing)
                timing["pipeline_time_s"] = pipeline_elapsed
                timing["generated_frames"] = generated
                pipeline_timings.append(timing)
                generated_frames = generated
        decode_time = sum(item["decode_time_s"] for item in pipeline_timings) / len(pipeline_timings)
        inference_time = sum(item["inference_time_s"] for item in pipeline_timings) / len(pipeline_timings)
        h2d_time = sum(item["h2d_time_s"] for item in pipeline_timings) / len(pipeline_timings)
        d2h_time = sum(item["d2h_time_s"] for item in pipeline_timings) / len(pipeline_timings)
        encode_time = sum(item["encode_time_s"] for item in pipeline_timings) / len(pipeline_timings)
        encode_write_time = sum(item.get("encode_write_time_s", 0.0) for item in pipeline_timings) / len(pipeline_timings)
        encode_finalize_time = sum(item.get("encode_finalize_time_s", 0.0) for item in pipeline_timings) / len(pipeline_timings)
        processing_time = sum(item["pipeline_time_s"] for item in pipeline_timings) / len(pipeline_timings)
    else:
        results = []
        timing = {"inference_time_s": 0.0, "h2d_time_s": 0.0, "d2h_time_s": 0.0}
        use_scheduler = _supports_scheduler(engine)
        timed_engine = (
            _timed_scheduler_backend(engine, timing)
            if use_scheduler
            else _timed_engine(engine, args.backend, timing)
        )
        for _ in range(args.iterations):
            started = time.perf_counter()
            if use_scheduler:
                results = _scheduled_inference(
                    timed_engine, frames, niters, factory.selection.batch_size
                )
            else:
                results = list(timed_engine.interpolate(frames, niters))
            inference_time += time.perf_counter() - started
        inference_time /= args.iterations
        h2d_time = timing["h2d_time_s"] / args.iterations
        d2h_time = timing["d2h_time_s"] / args.iterations
        processing_time = inference_time
        generated_frames = len(results)

    sampler.stop()
    close = getattr(factory, "close", None)
    if close is not None:
        close()
    peak_vram, _, _ = _torch_stats()
    if peak_vram == 0:
        peak_vram = sampler.peak_mib

    source_frames = len(frames) if not args.pipeline else (
        generated_frames and (generated_frames - 1) // (args.mode == "2x" and 2 or int(args.mode[:-1])) + 1
    )
    if not source_frames:
        source_frames = int(round(float(info.get("duration") or 0.0) * float(info["fps"])))
        if args.frames:
            source_frames = min(source_frames, args.frames)
    output_resolution = resolution or (info["width"], info["height"])
    record = _record_base(args, input_path, info, int(args.mode[:-1]), source_frames, output_resolution)
    record.update(
        {
            "iterations": args.iterations,
            "load_time_s": round(load_time, 6),
            "warmup_time_s": round(warmup_time, 6),
            "inference_time_s": round(inference_time, 6),
            "decode_time_s": round(decode_time, 6),
            "encode_time_s": round(encode_time, 6),
            "encode_write_time_s": round(encode_write_time, 6),
            "encode_finalize_time_s": round(encode_finalize_time, 6),
            "h2d_time_s": round(h2d_time, 6),
            "d2h_time_s": round(d2h_time, 6),
            "total_time_s": round(time.perf_counter() - total_started, 6),
            "processing_time_s": round(processing_time, 6),
            "generated_frames": generated_frames,
            "inference_fps": round(generated_frames / inference_time, 6) if inference_time else 0.0,
            "realtime_factor": round(realtime_factor(record["input_duration_s"], processing_time), 6),
            "peak_vram_mib": peak_vram,
            "gpu_utilization_percent": sampler.peak_util,
        }
    )
    json.dumps(record)
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        record = run_benchmark(args)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
