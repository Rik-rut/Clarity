"""Repeatable benchmark mode: ``uv run main.py benchmark``.

Runs a fixed workload (the benchmark video from the performance plan by
default) through an engine and records the doc's §11 metrics to
``benchmark-results.json`` in the project root so every optimization is
measured against the same baseline and history is never lost.

Two modes:

- inference-only (default): decode up to ``--frames`` frames, run the
  engine, report inference time / fps / s-per-frame / peak VRAM.
- full pipeline (``--pipeline``): the normal end-to-end run
  (decode -> enhance -> encode), reporting total pipeline time. This is
  the official apples-to-apples number vs the 16m 37s baseline.

Official numbers must use the full video (no ``--frames``) per the
performance plan. Use ``--frames`` only for fast iteration.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from video_upscaler import config
from video_upscaler.backend import backend_label, detect_backend
from video_upscaler.config import BASE_DIR, INPUT_DIR
from video_upscaler.ffmpeg import decode_frames, probe
from video_upscaler.models import model_for_profile, scale_for_model

BENCHMARK_VIDEO = "WWClipPlanning Devontae Smith at WR20 Is a Mistake CUT.mp4"

# Baseline recorded in CLARITY_PERFORMANCE_OPTIMIZATION_PLAN.md (Real-CUGAN,
# 2x_Balanced, torch fp16, full pipeline). Reference for regression reports.
BASELINE_VIDEO = BENCHMARK_VIDEO
BASELINE_PROFILE = "2x_Balanced"
BASELINE_TIME = 16 * 60 + 37  # seconds

RESULT_LOG = BASE_DIR / "benchmark-results.json"

_PRECISION_BY_BACKEND = {
    "torch-cuda": "fp16",
    "torch-cpu": "fp32",
    "ncnn": "fp16",
    "tensorrt": "fp16",
}


def resolve_video(video: str | None, input_dir: Path = INPUT_DIR) -> Path:
    """Locate the benchmark video: explicit path, or the fixed plan video."""
    if video:
        path = Path(video)
        if path.is_file():
            return path
        candidate = input_dir / video
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Benchmark video not found: {video}")
    candidate = input_dir / BENCHMARK_VIDEO
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Benchmark video {BENCHMARK_VIDEO!r} was not found in {input_dir}.\n"
        "Add it (or pass --video <path>)."
    )


def gpu_name() -> str:
    """Return a short GPU label for the record (nvidia-smi, else torch)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "unknown"


def seconds_per_frame(inference_time: float, frames: int) -> float:
    return inference_time / frames if frames else 0.0


def fps(inference_time: float, frames: int) -> float:
    return frames / inference_time if inference_time else 0.0


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def comparison_report(baseline: float, measured: float) -> str:
    """Required report block comparing a measured time to the baseline."""
    improvement = baseline - measured
    speedup = baseline / measured if measured else 0.0
    percent = improvement / baseline * 100 if baseline else 0.0
    return (
        f"\nBaseline time: {format_duration(baseline)}\n"
        f"Optimized time: {format_duration(measured)}\n"
        f"Absolute improvement: {format_duration(improvement)}\n"
        f"Speedup: {speedup:.2f}x\n"
        f"Percentage improvement: {percent:.1f}%"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py benchmark",
        description="Run the Clarity benchmark workload and record results.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "ncnn", "tensorrt"],
        default="auto",
        help="Engine backend (default: auto-detect).",
    )
    parser.add_argument(
        "--profile", default=None,
        help="Profile name (default: 2x_Balanced).",
    )
    parser.add_argument(
        "--video", default=None,
        help="Video path or filename (default: the fixed plan benchmark video).",
    )
    parser.add_argument(
        "--frames", type=int, default=0,
        help="Process only the first N frames (0 = all; iteration only).",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Frames per engine call (default: 1).",
    )
    parser.add_argument(
        "--pipeline", action="store_true",
        help="Also run the full end-to-end pipeline (decode+enhance+encode) "
        "and report total time.",
    )
    return parser.parse_args(argv)


class _NvidiaSmiSampler:
    """Background nvidia-smi sampler for peak VRAM / GPU utilization."""

    def __init__(self) -> None:
        self._peak_mib = 0
        self._peak_util = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 0 and result.stdout.strip():
                    mem, util = result.stdout.strip().splitlines()[0].split(",")
                    self._peak_mib = max(self._peak_mib, int(mem.strip()))
                    self._peak_util = max(self._peak_util, int(util.strip()))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(0.2)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def peak_mib(self) -> int:
        return self._peak_mib

    @property
    def peak_util(self) -> int:
        return self._peak_util


def resolve_backend(requested: str) -> str:
    """Map the CLI backend choice to a concrete backend key."""
    if requested == "auto":
        return detect_backend()
    if requested == "torch":
        from video_upscaler.backend import _torch_cuda_available

        return "torch-cuda" if _torch_cuda_available() else "torch-cpu"
    if requested == "tensorrt":
        import importlib

        try:
            tensorrt_backend = importlib.import_module(
                "video_upscaler.tensorrt_backend"
            )
        except ImportError:
            raise RuntimeError(
                "TensorRT is not installed (add it with "
                "`uv sync --extra tensorrt` and extract the NVIDIA zip "
                "into tools/TensorRT/)."
            ) from None
        if not tensorrt_backend.tensorrt_available():
            raise RuntimeError(
                "TensorRT install not found. Extract the NVIDIA TensorRT "
                "Windows zip into tools/TensorRT/ (or set "
                "CLARITY_TENSORRT_DIR)."
            )
        return "tensorrt"
    return requested


def _frames_to_arrays(video: Path, limit: int, width: int, height: int) -> list[np.ndarray]:
    """Decode frames into a bounded list of RGB arrays."""
    frames: list[np.ndarray] = []
    for raw in decode_frames(video):
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3))
        if limit and len(frames) >= limit:
            break
    return frames


def _synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _peak_vram_torch() -> int:
    """Peak VRAM in MiB as tracked by the torch allocator, or 0."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return torch.cuda.max_memory_allocated(0) // (1 << 20)
    except ImportError:
        return 0


def _infer_once(
    engine,
    frames: list[np.ndarray],
    batch: int,
    progress_cb: Callable[[int, int], None] | None = None,
) -> float:
    """Run inference over frames with the requested batching, return seconds.

    ``progress_cb(done, total)`` is called as frames complete so long runs
    show live progress instead of silence.
    """
    total = len(frames)
    done = 0

    def _report() -> None:
        if progress_cb is not None:
            progress_cb(done, total)

    started = time.perf_counter()
    if batch == 1:
        if getattr(engine, "chunked", False):
            chunk_size = config.NCNN_CHUNK
            for i in range(0, total, chunk_size):
                chunk = frames[i : i + chunk_size]

                def _chunk_progress(fraction: float) -> None:
                    done = min(i + int(fraction * len(chunk)), total)
                    _report()

                engine.enhance_chunk(chunk, on_progress=_chunk_progress)
                done = min(i + chunk_size, total)
                _report()
        else:
            for frame in frames:
                engine.enhance(frame)
                done += 1
                _report()
    else:
        method = getattr(engine, "enhance_batch", None)
        if method is None:
            raise RuntimeError(
                f"Backend {type(engine).__name__} does not support batch "
                f"{batch} yet."
            )
        for i in range(0, total, batch):
            method(frames[i : i + batch])
            done = min(i + batch, total)
            _report()
    _synchronize()
    return time.perf_counter() - started


def load_records() -> list[dict]:
    if not RESULT_LOG.is_file():
        return []
    try:
        data = json.loads(RESULT_LOG.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def append_record(record: dict) -> None:
    records = load_records()
    records.append(record)
    tmp = RESULT_LOG.with_name(RESULT_LOG.name + ".tmp")
    tmp.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(RESULT_LOG)


def run_benchmark(args: argparse.Namespace) -> None:
    from video_upscaler.models import default_profile
    from video_upscaler.processor import build_engine, process_videos

    video = resolve_video(args.video)
    profile = args.profile or default_profile()
    model = model_for_profile(profile)
    scale = scale_for_model(model)
    info = probe(video)
    out_width = info["width"] * scale
    out_height = info["height"] * scale

    backend = resolve_backend(args.backend)
    engine = build_engine(profile, backend)

    frames = _frames_to_arrays(video, args.frames, info["width"], info["height"])
    frame_count = len(frames)
    if frame_count == 0:
        raise RuntimeError("No frames decoded from the benchmark video.")

    # Warmup: absorb cuDNN/TRT autotune and one-shot setup.
    warmup = frames[: min(3, frame_count)]
    _infer_once(engine, warmup, args.batch)

    sampler = _NvidiaSmiSampler()
    sampler.start()
    if backend.startswith("torch"):
        import torch

        torch.cuda.reset_peak_memory_stats(0)

    inference_started = time.perf_counter()
    last_percent = -1

    def _progress(done: int, total: int) -> None:
        nonlocal last_percent
        if total <= 0:
            return
        percent = int(done * 100 / total)
        if percent >= last_percent + 5 or done >= total:
            elapsed = time.perf_counter() - inference_started
            rate = elapsed / done if done else 0.0
            print(
                f"Processed {done}/{total} frames ({percent}%) — "
                f"{rate:.2f} s/frame",
                flush=True,
            )
            last_percent = percent

    inference_time = _infer_once(engine, frames, args.batch, _progress)
    sampler.stop()
    peak_vram = _peak_vram_torch() or sampler.peak_mib

    pipeline_time = None
    if args.pipeline:
        pipeline_started = time.perf_counter()
        process_videos(
            [video], profile,
            lambda file_index, file_count, percent: None,
        )
        pipeline_time = time.perf_counter() - pipeline_started

    measured = pipeline_time if args.pipeline else inference_time
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": video.name,
        "backend": backend,
        "backend_label": backend_label(backend),
        "profile": profile,
        "model": model,
        "precision": _PRECISION_BY_BACKEND.get(backend, "unknown"),
        "gpu": gpu_name(),
        "input_resolution": f"{info['width']}x{info['height']}",
        "output_resolution": f"{out_width}x{out_height}",
        "frames": frame_count,
        "inference_time": round(inference_time, 3),
        "pipeline_time": round(pipeline_time, 3) if pipeline_time is not None else None,
        "seconds_per_frame": round(seconds_per_frame(inference_time, frame_count), 4),
        "fps": round(fps(inference_time, frame_count), 4),
        "peak_vram_mib": peak_vram,
        "gpu_util_peak": sampler.peak_util,
    }
    append_record(record)

    print(f"\nBenchmark: {video.name}")
    print(f"Backend: {backend_label(backend)}")
    print(f"Model: {model} ({profile})")
    print(f"Precision: {record['precision']}")
    print(f"GPU: {record['gpu']}")
    print(f"Input: {record['input_resolution']}  Output: {record['output_resolution']}")
    print(f"Frames: {frame_count}")
    print(f"Inference: {format_duration(inference_time)}")
    if pipeline_time is not None:
        print(f"Pipeline: {format_duration(pipeline_time)}")
    print(f"FPS: {record['fps']}  Seconds/frame: {record['seconds_per_frame']}")
    print(f"Peak VRAM: {peak_vram} MiB  Peak GPU util: {sampler.peak_util}%")
    if video.name == BASELINE_VIDEO and profile == BASELINE_PROFILE:
        print(comparison_report(BASELINE_TIME, measured))
    print(f"\nResult appended to {RESULT_LOG}")


def run_cli(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.frames < 0:
        raise SystemExit("--frames must be >= 0 (0 = all frames).")
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1.")
    try:
        run_benchmark(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
