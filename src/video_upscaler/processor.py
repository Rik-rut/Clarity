"""Per-video pipeline coordination and batch continue-on-failure."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Generator, Iterable, Iterator

import numpy as np

from video_upscaler.backend import detect_backend
from video_upscaler import config
from video_upscaler.config import MAX_OUTPUT_WIDTH_WARNING
from video_upscaler.ffmpeg import decode_frames, encode_video, probe
from video_upscaler.models import (
    model_for_profile,
    scale_for_model,
    niters_for_factor,
)
from video_upscaler.amt_scheduler import AMTFrameScheduler

def _unique_tagged_path(output_dir: Path, original_name: str, tag: str) -> Path:
    """Build ``<stem>_<tag><ext>``, appending ``_N`` before the extension
    when the target exists (never overwrites)."""
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix
    candidate = output_dir / f"{stem}_{tag}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}_{tag}_{counter}{suffix}"
        counter += 1
    return candidate


def upscale_output_tag(profile: str) -> str:
    """Return the filename tag for a profile, e.g. ``2x``."""
    return f"{scale_for_model(model_for_profile(profile))}x"


def slowed_output_tag(factor: int, model_key: str) -> str:
    """Return the filename tag for slow motion, e.g. ``slowed2x_amt-s``."""
    return f"slowed{factor}x_{model_key.lower()}"


def dedup_output_tag(factor: int, model_type: str) -> str:
    """Return the filename tag for MultiPassDedup, e.g. ``dedup2x_gmfss``."""
    return f"dedup{factor}x_{model_type.lower()}"


def unique_output_path(
    output_dir: Path, original_name: str, profile: str
) -> Path:
    """Build a tagged output path, never overwriting an existing file."""
    return _unique_tagged_path(
        output_dir, original_name, upscale_output_tag(profile)
    )


def unique_interp_output_path(
    output_dir: Path, original_name: str, factor: int, model_key: str
) -> Path:
    """Build a slow-motion tagged output path, never overwriting existing."""
    return _unique_tagged_path(
        output_dir, original_name, slowed_output_tag(factor, model_key)
    )


def unique_dedup_output_path(
    output_dir: Path, original_name: str, factor: int, model: str
) -> Path:
    """Build a dedup-tagged output path, never overwriting an existing file."""
    return _unique_tagged_path(
        output_dir, original_name, dedup_output_tag(factor, model)
    )



def effective_backend(backend: str, profile: str) -> str:
    """Apply profile-aware backend fallbacks (auto mode only).

    TensorRT supports 2x Real-CUGAN models only; 3x/4x profiles fall back
    to the torch engine in auto mode. An explicitly forced backend
    (CLARITY_BACKEND=tensorrt) is never silently overridden.
    """
    from video_upscaler.models import scale_for_model

    if config.BACKEND_PREF != "auto":
        return backend
    if backend == "tensorrt" and scale_for_model(model_for_profile(profile)) != 2:
        return "torch-cuda"
    return backend


def _make_amt_backend_factory(model_key: str, selection=None):
    from video_upscaler.interp import AMTBackendFactory, select_amt_backend

    selection = selection or select_amt_backend(model_key)
    return AMTBackendFactory(model_key, selection)


def build_engine(profile: str, backend: str):
    """Construct the right engine for profile + backend."""
    from video_upscaler.models import model_for_profile, ncnn_args_for_profile

    backend = effective_backend(backend, profile)
    model_name = model_for_profile(profile)

    if backend == "tensorrt":
        from video_upscaler.tensorrt_backend import RealCUGANTensorRTEngine

        return RealCUGANTensorRTEngine(model_name)

    if backend == "ncnn":
        from video_upscaler.ncnn import NCNNEngine

        return NCNNEngine(profile)

    from video_upscaler.cugan import RealCUGANEngine

    return RealCUGANEngine(model_name)


def _enhance_with_progress(
    raw_frames: Generator[bytes, None, None],
    engine,
    width: int,
    height: int,
    total_frames: float,
    file_index: int,
    file_count: int,
    progress_cb: Callable[[int, int, int], None],
) -> Generator:
    """Enhance frames, reporting progress every ~2%."""
    done = 0
    last_percent = -1

    def _report(percent: int) -> None:
        nonlocal last_percent
        if percent >= last_percent + 2:
            progress_cb(file_index, file_count, percent)
            last_percent = percent

    def _to_frame(raw: bytes) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)

    if getattr(engine, "chunked", False):
        from video_upscaler.config import NCNN_CHUNK

        buffer: list[np.ndarray] = []
        for raw in raw_frames:
            done += 1
            buffer.append(_to_frame(raw))
            if len(buffer) >= NCNN_CHUNK:
                for frame in engine.enhance_chunk(buffer):
                    yield frame
                buffer = []
                if total_frames > 0:
                    _report(min(100, int(done * 100 / total_frames)))
        if buffer:
            for frame in engine.enhance_chunk(buffer):
                yield frame
        _report(100)
        return

    for raw in raw_frames:
        done += 1
        if total_frames > 0:
            _report(min(100, int(done * 100 / total_frames)))
        yield engine.enhance(_to_frame(raw))


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact "1m 23s" / "45s" string."""
    seconds = int(round(seconds))
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def process_videos(
    videos: list[Path],
    profile: str,
    progress_cb: Callable[[int, int, int], None],
) -> dict:
    """Process videos sequentially; a failing file never stops the batch.

    Returns {"success": [Path], "failed": [(name, reason)], "times": [...]}
    where "times" holds per-video wall-clock seconds in the same order as
    "success" (and a None marker for failed videos).
    """
    results: dict = {"success": [], "failed": [], "times": []}

    backend = detect_backend()
    engine = build_engine(profile, backend)
    scale = scale_for_model(model_for_profile(profile))

    file_count = len(videos)
    for index, video in enumerate(videos, start=1):
        started = time.perf_counter()
        try:
            info = probe(video)
            out_width = info["width"] * scale
            out_height = info["height"] * scale
            if out_width > MAX_OUTPUT_WIDTH_WARNING:
                print(
                    f"Warning: output width {out_width}px exceeds the "
                    f"{MAX_OUTPUT_WIDTH_WARNING}px threshold — processing may "
                    "be very slow or produce an unusable file."
                )
            out_path = unique_output_path(config.OUTPUT_DIR, video.name, profile)
            total_frames = info["duration"] * info["fps"]
            raw_frames = decode_frames(video)
            enhanced = _enhance_with_progress(
                raw_frames,
                engine,
                info["width"],
                info["height"],
                total_frames,
                index,
                file_count,
                progress_cb,
            )
            encode_video(
                enhanced,
                out_width,
                out_height,
                info["fps"],
                video,
                out_path,
                info["has_audio"],
                rotation=info["rotation"],
                use_nvenc=config.USE_NVENC,
            )
            progress_cb(index, file_count, 100)
            results["success"].append(out_path)
            results["times"].append(time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - batch must continue
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            results["failed"].append((video.name, reason))
            results["times"].append(None)

    return results


def _interp_window_stream(
    raw_iter: Iterable[bytes],
    engine: "AMTInterpEngine",
    niters: int,
    seg: int,
    shape: tuple[int, int],
    scheduler: AMTFrameScheduler | None = None,
    batch_size: int = 1,
) -> Iterator[np.ndarray]:
    """Stream-interpolate decoded source frames in windows of ``seg`` frames.

    Each window overlaps the previous by one source frame (carried over) so the
    output is seamless. The last output frame of every full window is dropped
    and re-emitted by the next window's carried frame, so no frame is ever
    duplicated or lost regardless of where the video ends. Memory stays bounded
    to roughly one window of frames.
    """
    h, w = shape
    carry = None
    window_idx = 0
    while True:
        window = [carry] if carry is not None else []
        count = 0
        for raw in raw_iter:
            window.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3))
            count += 1
            if count >= seg:
                break
        if not window:
            break
        partial = count < seg
        window_idx += 1
        print(
            f"  window {window_idx}: {len(window)} source frames"
            + (" (final)" if partial else "")
        )
        if scheduler is None:
            out = list(engine.interpolate(window, niters))
        else:
            out = list(
                scheduler.interpolate_window(window, niters, engine, batch_size)
            )
        if partial:
            # Final window: emit everything; its last source frame is the true
            # end of the clip (no next window will re-emit it). Clear the carry
            # so the post-loop recovery window does not duplicate it.
            yield from out
            carry = None
            break
        # Full window: drop the boundary source frame; the carried frame in the
        # next window re-emits it exactly once.
        yield from out[:-1]
        carry = window[-1]

    # Stream ended exactly on a full-window boundary: the dropped boundary was
    # never re-emitted, so emit it as a 1-frame recovery window.
    if carry is not None:
        if scheduler is None:
            yield from engine.interpolate([carry], niters)
        else:
            yield from scheduler.interpolate_window(
                [carry], niters, engine, batch_size
            )


def process_interpolate(
    videos: list[Path],
    model_key: str,
    factor: int,
    progress_cb: Callable[[int, int, int], None],
) -> dict:
    """Interpolate videos (slow motion) sequentially; a failing file never
    stops the batch.

    Returns the same shape as ``process_videos``. The output keeps the input
    frame rate, so a 2x/4x/8x factor yields a half/quarter/eighth-speed clip
    of proportionally more frames.

    Frames are decoded and interpolated in streaming windows (with 1-frame
    overlap for seamless stitching) so memory stays bounded to roughly one
    window regardless of video length. ``progress_cb`` receives a fine-grained
    overall percent (0-100).
    """
    results: dict = {"success": [], "failed": [], "times": []}
    factory = _make_amt_backend_factory(model_key)
    selection = factory.selection
    if selection.fallback_reason:
        print(f"AMT backend fallback: {selection.fallback_reason}")
    engine = None
    if selection.backend == "pytorch":
        engine = factory.build()
    scheduler = AMTFrameScheduler()
    niters = niters_for_factor(factor)
    warmed_backends: set[int] = set()

    file_count = len(videos)
    for index, video in enumerate(videos, start=1):
        started = time.perf_counter()
        try:
            info = probe(video)
            width, height = info["width"], info["height"]
            if width <= 0 or height <= 0:
                raise RuntimeError(f"Could not determine dimensions for {video.name}.")

            # TensorRT resources are shape-specific and belong inside this
            # try block so one invalid profile only fails this video.
            if selection.backend == "tensorrt":
                try:
                    engine = factory.build((height, width))
                except Exception as exc:  # noqa: BLE001 - auto fallback is explicit
                    if selection.explicit_tensorrt:
                        raise
                    reason = str(exc).strip().splitlines()[0] or type(exc).__name__
                    fallback_reason = (
                        "AMT TensorRT setup failed; falling back to PyTorch: "
                        f"{reason}"
                    )
                    print(fallback_reason)
                    close = getattr(factory, "close", None)
                    if close is not None:
                        close()
                    fallback_selection = replace(
                        selection,
                        backend="pytorch",
                        precision=config.AMT_PRECISION,
                        fallback_reason=fallback_reason,
                        explicit_tensorrt=False,
                    )
                    factory = _make_amt_backend_factory(model_key, fallback_selection)
                    selection = factory.selection
                    engine = factory.build()

            warmup = getattr(engine, "warmup", None)
            if warmup is not None and id(engine) not in warmed_backends:
                warmup((height, width), selection.batch_size)
                warmed_backends.add(id(engine))

            out_path = unique_interp_output_path(
                config.OUTPUT_DIR, video.name, factor, model_key
            )

            # Estimate total output frames for progress (probe is approximate).
            est_source = max(1, int(round(info["duration"] * info["fps"])))
            est_out = max(1, est_source * (2 ** niters) - (2 ** niters - 1))
            if factor >= 4 and info["duration"] >= 60:
                print(
                    f"Note: {factor}x interpolation of a ~{info['duration']:.0f}s "
                    "video is compute-intensive and can take a long time. Consider "
                    "2x or a shorter clip if you need a quick result."
                )

            seg = config.AMT_SEGMENT_FRAMES or est_source
            produced = 0
            last_percent = -1

            def _stream():
                nonlocal produced, last_percent
                raw_iter = decode_frames(video)
                for frame in _interp_window_stream(
                    raw_iter,
                    engine,
                    niters,
                    seg,
                    (height, width),
                    scheduler=scheduler,
                    batch_size=selection.batch_size,
                ):
                    yield yield_frame(frame)

            def yield_frame(frame):
                nonlocal produced, last_percent
                produced += 1
                percent = min(100, int(produced * 100 / est_out))
                if percent >= last_percent + 1:
                    if progress_cb:
                        progress_cb(index, file_count, percent)
                    last_percent = percent
                return frame

            encode_video(
                _stream(),
                width,
                height,
                info["fps"],
                video,
                out_path,
                info["has_audio"],
                rotation=info["rotation"],
                use_nvenc=config.USE_NVENC,
            )
            progress_cb(index, file_count, 100)
            results["success"].append(out_path)
            results["times"].append(time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - batch must continue
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            results["failed"].append((video.name, reason))
            results["times"].append(None)

    close = getattr(factory, "close", None)
    if close is not None:
        close()
    return results


def process_dedup(
    videos: list[Path],
    model: str,
    npass: str | int,
    factor: int,
    progress_cb: Callable[[int, int, int], None] | None = None,
) -> dict:
    """Process anime frame interpolation using MultiPassDedup.

    A failing file never stops the batch. Returns the standard
    {"success": [...], "failed": [...], "times": [...]} dict.
    """
    from video_upscaler.dedup import run_dedup_infer
    from video_upscaler.dedup_backend import parse_npass, validate_model_type

    model_type = validate_model_type(model)
    npass_int = parse_npass(npass)
    results: dict = {"success": [], "failed": [], "times": []}
    file_count = len(videos)

    for index, video in enumerate(videos, start=1):
        started = time.perf_counter()
        try:
            out_path = unique_dedup_output_path(
                config.OUTPUT_DIR, video.name, factor, model_type
            )
            if progress_cb:
                progress_cb(index, file_count, 0)
            run_dedup_infer(
                video_in=video,
                video_out=out_path,
                model_type=model_type,
                npass=npass_int,
                factor=factor,
                scale=config.DEDUP_SCALE_DEFAULT,
                enable_scdet=config.DEDUP_SCDET_DEFAULT,
                scdet_threshold=config.DEDUP_SCDET_THRESHOLD,
                hwaccel=config.USE_NVENC,
                progress_cb=progress_cb,
            )
            if progress_cb:
                progress_cb(index, file_count, 100)
            results["success"].append(out_path)
            results["times"].append(time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - batch must continue
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            results["failed"].append((video.name, reason))
            results["times"].append(None)

    return results

