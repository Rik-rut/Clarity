"""Clarity job pipeline for MatAnyone2 video matting.

Frames stream through ffmpeg pipes (never fully materialized in RAM), a
single inference pass feeds raw spool files for the requested outputs
(alpha matte, green-screen composite, transparent ProRes 4444), and each
output encodes through the shared ``encode_video`` helper so FPS/audio/
rotation handling stays identical to every other Clarity tab.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from video_upscaler import config
from video_upscaler.ffmpeg import decode_frames, encode_video, probe, prores_available
from video_upscaler.processor import _unique_tagged_path

GREEN_SCREEN_RGB = np.array([120, 255, 155], dtype=np.float32) / 255.0

_CANCEL_MESSAGE = "job cancelled by user"

# Output selector values accepted from the job params.
OUTPUT_MATTE = "matte"
OUTPUT_GREENSCREEN = "greenscreen"
# Legacy value written by earlier builds; treated as green screen.
_OUTPUT_LEGACY_FOREGROUND = "foreground"
OUTPUT_TRANSPARENT = "transparent"


def _normalize_outputs(raw: object) -> set[str]:
    requested = raw if isinstance(raw, (list, tuple)) else [OUTPUT_MATTE, _OUTPUT_LEGACY_FOREGROUND]
    selected: set[str] = set()
    for value in requested:
        if value == _OUTPUT_LEGACY_FOREGROUND:
            selected.add(OUTPUT_GREENSCREEN)
        elif value in (OUTPUT_MATTE, OUTPUT_GREENSCREEN, OUTPUT_TRANSPARENT):
            selected.add(str(value))
    return selected


def _is_cancellation(exc: BaseException) -> bool:
    return _CANCEL_MESSAGE in str(exc).lower()


def build_session(selection):
    """Indirection point so tests can stub session construction."""
    from video_upscaler.matanyone2.pytorch_backend import build_session as build

    return build(selection)


def _frame_to_tensor(img_rgb_u8: np.ndarray):
    from video_upscaler.matanyone2.pytorch_backend import frame_to_tensor

    return frame_to_tensor(img_rgb_u8)


def _raw_reader(path: Path, width: int, height: int, channels: int = 3) -> Callable[..., np.ndarray]:
    frame_bytes = width * height * channels

    def reader():
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(frame_bytes)
                if not chunk:
                    break
                yield np.frombuffer(chunk, np.uint8).reshape(height, width, channels)

    return reader()


def _maybe_resize(img: np.ndarray, pw: int, ph: int) -> np.ndarray:
    if img.shape[1] == pw and img.shape[0] == ph:
        return img
    return cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)


def process_matanyone2(
    video_paths: list[Path],
    params: dict,
    progress_callback: Callable[[int, int, int], None],
    stage_cb: Callable[[str], None] | None = None,
) -> dict:
    results: dict = {"success": [], "failed": [], "times": []}
    total_files = len(video_paths)

    for index, src in enumerate(video_paths, start=1):
        started = time.perf_counter()
        try:
            outputs = _process_one(
                src, params, progress_callback, index, total_files, stage_cb
            )
        except Exception as exc:
            if _is_cancellation(exc):
                raise
            results["failed"].append((src.name, str(exc)))
            continue
        results["success"].extend(outputs)
        results["times"].append(
            {
                "file": src.name,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
    return results


def _process_one(
    src: Path,
    params: dict,
    progress_callback: Callable[[int, int, int], None],
    index: int,
    total_files: int,
    stage_cb: Callable[[str], None] | None,
) -> list[Path]:
    from video_upscaler.matanyone2.backend import select_backend
    from video_upscaler.matanyone2.mask import (
        decode_mask_b64,
        preprocess_mask,
        processing_size,
        save_mask,
        validate_mask,
    )

    def report_stage(stage: str) -> None:
        if stage_cb is not None:
            stage_cb(stage)

    info = probe(src)
    width, height, fps = int(info["width"]), int(info["height"]), float(info["fps"])
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"Could not read usable dimensions/fps from {src.name}")

    mask_png = params.get("mask_png")
    if not mask_png:
        raise RuntimeError("MatAnyone2 requires a first-frame mask.")
    mask = decode_mask_b64(mask_png)
    # The browser paints on the frame preview, which may be smaller than the
    # native video (preview caps the long side at 1280px). Nearest-neighbour
    # resample back to native geometry so validation and processing agree.
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    validate_mask(mask, width, height)

    max_size = int(params.get("max_size", -1))
    pw, ph = processing_size(width, height, max_size)
    proc_mask = preprocess_mask(
        mask,
        dilate=int(params.get("dilate", 10)),
        erode=int(params.get("erode", 10)),
        target_size=(pw, ph),
    )

    workdir = Path(tempfile.mkdtemp(prefix=f"ma2_{src.stem}_"))
    outputs_wanted = _normalize_outputs(params.get("outputs"))
    if OUTPUT_TRANSPARENT in outputs_wanted:
        prores_available()
    matte_raw = workdir / "matte.raw"
    gs_raw = workdir / "greenscreen.raw"
    rgba_raw = workdir / "rgba.raw"

    selection = select_backend(
        str(params.get("backend", "auto")), str(params.get("precision", "fp16"))
    )

    session = None
    failed = False
    estimated_total = max(1, round(float(info["duration"]) * fps)) if info["duration"] else 1
    try:
        save_mask(proc_mask.astype(np.uint8), workdir / "mask.png")

        report_stage("Loading model")
        session = build_session(selection)

        frames = decode_frames(src)
        first_bytes = next(frames)
        first_img = _maybe_resize(
            np.frombuffer(first_bytes, np.uint8).reshape(height, width, 3), pw, ph
        )

        report_stage("Warming up")
        session.start(
            _frame_to_tensor(first_img),
            _mask_tensor(proc_mask),
            warmup=int(params.get("warmup", 10)),
        )

        matte_out = (
            _unique_tagged_path(config.OUTPUT_DIR, src.name, "matte")
            if OUTPUT_MATTE in outputs_wanted
            else None
        )
        gs_out = (
            _unique_tagged_path(config.OUTPUT_DIR, src.name, "greenscreen")
            if OUTPUT_GREENSCREEN in outputs_wanted
            else None
        )
        transparent_out = (
            _unique_tagged_path(config.OUTPUT_DIR, src.name, "transparent").with_suffix(".mov")
            if OUTPUT_TRANSPARENT in outputs_wanted
            else None
        )

        processed = 0
        with open(matte_raw, "wb") as mf, open(gs_raw, "wb") as gf, open(rgba_raw, "wb") as rf:
            def emit(img: np.ndarray, prob_map: np.ndarray) -> None:
                pha = np.clip(prob_map, 0.0, 1.0)[..., None]
                img_f = img.astype(np.float32) / 255.0
                if matte_out is not None:
                    gray = np.repeat((pha * 255.0).round().astype(np.uint8), 3, axis=2)
                    mf.write(gray.tobytes())
                if gs_out is not None:
                    comp = (
                        img_f * pha
                        + GREEN_SCREEN_RGB.reshape(1, 1, 3) * (1.0 - pha)
                    )
                    # comp is float 0..1 — scale back to 8-bit or the video
                    # encodes as black (regression: values collapsed to 0/1).
                    gf.write(
                        (comp.clip(0.0, 1.0) * 255.0).round().astype(np.uint8).tobytes()
                    )
                if transparent_out is not None:
                    rgba = np.empty((*pha.shape[:2], 4), np.uint8)
                    rgba[..., :3] = img.round().clip(0, 255).astype(np.uint8)
                    rgba[..., 3] = (pha[..., 0] * 255.0).round().astype(np.uint8)
                    rf.write(rgba.tobytes())

            emit(first_img, session.first_prob_np)
            processed += 1
            progress_callback(index, total_files, _percent(processed, estimated_total))

            for chunk in frames:
                img = _maybe_resize(
                    np.frombuffer(chunk, np.uint8).reshape(height, width, 3), pw, ph
                )
                prob_map = session.step(_frame_to_tensor(img))
                emit(img, prob_map)
                processed += 1
                if processed % 10 == 0 or processed == estimated_total:
                    progress_callback(
                        index, total_files, _percent(processed, estimated_total)
                    )

        report_stage("Encoding outputs")
        progress_callback(index, total_files, 88)
        outputs: list[Path] = []
        if matte_out is not None:
            encode_video(
                _raw_reader(matte_raw, pw, ph), pw, ph, fps, src, matte_out,
                use_audio=False, rotation=int(info.get("rotation", 0)),
                use_nvenc=False,
            )
            outputs.append(matte_out)
        if gs_out is not None:
            encode_video(
                _raw_reader(gs_raw, pw, ph), pw, ph, fps, src, gs_out,
                use_audio=bool(info.get("has_audio")),
                rotation=int(info.get("rotation", 0)),
                use_nvenc=False,
            )
            outputs.append(gs_out)
        if transparent_out is not None:
            encode_video(
                _raw_reader(rgba_raw, pw, ph, channels=4), pw, ph, fps, src,
                transparent_out,
                use_audio=bool(info.get("has_audio")),
                rotation=int(info.get("rotation", 0)),
                use_nvenc=False, alpha=True,
            )
            outputs.append(transparent_out)
        progress_callback(index, total_files, 100)
        return outputs
    except Exception:
        failed = True
        raise
    finally:
        if session is not None:
            session.close()
        if failed:
            print(f"[ma2] kept debug artifacts: {workdir}", flush=True)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def _mask_tensor(proc_mask: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(proc_mask)).float()


def _percent(processed: int, estimated_total: int) -> int:
    fraction = min(1.0, processed / max(1, estimated_total))
    return 5 + int(fraction * 80)
