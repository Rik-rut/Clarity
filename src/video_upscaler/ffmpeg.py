"""FFmpeg/ffprobe subprocess helpers: check, probe, frame decode, encode.

All file paths are passed as argv list entries (never shell-interpreted).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Generator

from video_upscaler.config import ffmpeg_path

_FRAME_CHUNK = 1 << 20  # bounded 1 MiB pipe reads

_NVENC_AVAILABLE: bool | None = None
_PRORES_AVAILABLE: bool | None = None


def prores_available() -> bool:
    """Return True if this ffmpeg build has the prores_ks encoder."""
    global _PRORES_AVAILABLE
    if _PRORES_AVAILABLE is not None:
        return _PRORES_AVAILABLE
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        _PRORES_AVAILABLE = False
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        _PRORES_AVAILABLE = "prores_ks" in (result.stdout + result.stderr)
    except (subprocess.SubprocessError, OSError):
        _PRORES_AVAILABLE = False
    return _PRORES_AVAILABLE


def _ffprobe_path() -> str | None:
    """Return the ffprobe binary path (next to ffmpeg, else PATH)."""
    env_ffmpeg = os.environ.get("CLARITY_FFMPEG")
    if env_ffmpeg:
        candidate = Path(env_ffmpeg).with_name(
            "ffprobe.exe" if os.name == "nt" else "ffprobe"
        )
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffprobe")


def nvenc_available() -> bool:
    """Return True if this ffmpeg build has an NVENC h.264 encoder."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        _NVENC_AVAILABLE = False
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        _NVENC_AVAILABLE = "h264_nvenc" in (result.stdout + result.stderr)
    except (subprocess.SubprocessError, OSError):
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def vp9_available() -> bool:
    """Return True if this ffmpeg build has a libvpx-vp9 encoder (alpha WebM)."""
    global _VP9_AVAILABLE
    if _VP9_AVAILABLE is not None:
        return _VP9_AVAILABLE
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        _VP9_AVAILABLE = False
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        _VP9_AVAILABLE = "libvpx-vp9" in (result.stdout + result.stderr)
    except (subprocess.SubprocessError, OSError):
        _VP9_AVAILABLE = False
    return _VP9_AVAILABLE


def check_ffmpeg() -> str | None:
    """Return None if ffmpeg + ffprobe are found, else the spec error message."""
    if ffmpeg_path() and _ffprobe_path():
        return None
    return "FFmpeg was not found.\n\nPlease install FFmpeg and ensure it is available in PATH."


def _rational_to_float(value: str) -> float:
    """Convert an ffprobe rational ("30000/1001") to a float."""
    try:
        numerator, denominator = value.split("/")
        if float(denominator) == 0:
            return 0.0
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: Path) -> dict:
    """Return metadata for a video: width, height, fps, has_audio, duration.

    Raises RuntimeError with the ffprobe stderr on failure.
    """
    ffprobe = _ffprobe_path()
    if not ffprobe:
        raise RuntimeError(
            "FFprobe was not found.\n\nPlease install FFmpeg and ensure it is available in PATH."
        )

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,side_data_list",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to read video metadata for {path.name}: {result.stderr.strip()}"
        )

    data = json.loads(result.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    fps = _rational_to_float(stream.get("avg_frame_rate") or "0/0")
    if fps <= 0:
        fps = _rational_to_float(stream.get("r_frame_rate") or "0/0")
    if fps <= 0:
        fps = 30.0

    rotation = 0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = side["rotation"]
            break

    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "codec_name": stream.get("codec_name"),
        "has_audio": _probe_audio(ffprobe, path),
        "duration": float(fmt.get("duration") or 0.0),
        "rotation": int(rotation),
    }


def _probe_audio(ffprobe: str, path: Path) -> bool:
    """Return True if the file contains at least one audio stream."""
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    data = json.loads(result.stdout or "{}")
    return bool(data.get("streams"))


def decode_frames(path: Path) -> Generator[bytes, None, None]:
    """Yield raw RGB24 frames from ``path`` via an ffmpeg rawvideo pipe."""
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg was not found.\n\nPlease install FFmpeg and ensure it is available in PATH."
        )
    info = probe(path)
    frame_size = info["width"] * info["height"] * 3
    if frame_size <= 0:
        raise RuntimeError(f"Could not determine frame size for {path.name}.")

    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        buffer = b""
        while True:
            needed = frame_size - len(buffer)
            chunk = process.stdout.read(min(_FRAME_CHUNK, needed))
            if not chunk:
                break
            buffer += chunk
            while len(buffer) >= frame_size:
                yield buffer[:frame_size]
                buffer = buffer[frame_size:]
    finally:
        if process.poll() is None:
            process.kill()
        process.stdout.close()
        process.stderr.close()
        process.wait()


def encode_video(
    frames: Generator,
    frame_w: int,
    frame_h: int,
    fps: float,
    src_path: Path,
    out_path: Path,
    use_audio: bool,
    rotation: int = 0,
    use_nvenc: bool = True,
    timing: dict[str, float] | None = None,
    alpha: bool = False,
) -> None:
    """Encode enhanced frames to ``out_path`` with audio copy.

    ``use_nvenc=False`` forces CPU x264 — callers whose inference hammers
    CUDA (e.g. Real-CUGAN) pass this to avoid NVENC/CUDA contention, which
    on consumer GPUs can slow per-frame inference several-fold.

    ``alpha=True`` encodes RGBA input as a transparent Apple ProRes 4444
    QuickTime (.mov); requires an ffmpeg build with prores_ks. ProRes is
    the professional editing standard for alpha video (After Effects,
    Premiere, Resolve all read it natively) and every mainstream ffmpeg
    build ships it, unlike VP9 which drops alpha on many distributions.
    """
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg was not found.\n\nPlease install FFmpeg and ensure it is available in PATH."
        )
    if alpha and not prores_available():
        raise RuntimeError(
            "Transparent video export needs FFmpeg with the prores_ks "
            "encoder (included in standard FFmpeg builds).\n\n"
            "Please update FFmpeg or uncheck Transparent Video."
        )

    in_pix_fmt = "rgba" if alpha else "rgb24"
    command = [
        ffmpeg,
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        in_pix_fmt,
        "-s",
        f"{frame_w}x{frame_h}",
        "-r",
        f"{fps:.6f}".rstrip("0").rstrip("."),
        "-i",
        "-",
    ]
    if use_audio:
        command += ["-i", str(src_path)]
    command += ["-map", "0:v:0"]
    if use_audio:
        command += ["-map", "1:a:0?"]
    if alpha:
        # Transparent ProRes 4444: yuva444p10le keeps full-res alpha and is
        # read natively by every video editor. Audio stays stream-copied —
        # the QuickTime container accepts the usual source codecs.
        command += [
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
        ]
    elif use_nvenc and nvenc_available():
        # Hardware encode: far faster than CPU x264, quality tuned via CQ.
        # yuv420p is the universally decodable H.264 chroma format — browsers
        # and media players struggle with 4:4:4, especially at 4K.
        command += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-rc",
            "vbr",
            "-cq",
            "20",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        # libx264 defaults to yuv444p when fed rgb24, which browsers cannot
        # reliably decode (4K renders as corrupted stripes). Force 4:2:0.
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    if use_audio:
        command += ["-c:a", "copy"]
    if rotation:
        command += ["-metadata:s:v:0", f"rotate={rotation}"]
    command += ["-movflags", "+faststart", str(out_path)]

    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )
    write_time = 0.0
    finalize_started: float | None = None
    stdin_closed = False
    try:
        for frame in frames:
            data = frame.tobytes() if hasattr(frame, "tobytes") else frame
            write_started = time.perf_counter()
            process.stdin.write(data)
            write_time += time.perf_counter() - write_started
        finalize_started = time.perf_counter()
        process.stdin.close()
        stdin_closed = True
    except BrokenPipeError:
        finalize_started = time.perf_counter()
        process.stdin.close()
        stdin_closed = True
    finally:
        if finalize_started is None:
            finalize_started = time.perf_counter()
        if not stdin_closed:
            # Exception mid-loop: close stdin so ffmpeg sees EOF and exits —
            # otherwise the stderr drain below deadlocks forever and masks
            # the original error.
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        process.stderr.close()
    returncode = process.wait()
    finalize_time = time.perf_counter() - finalize_started
    if timing is not None:
        timing["encode_write_time_s"] = timing.get("encode_write_time_s", 0.0) + write_time
        timing["encode_finalize_time_s"] = timing.get("encode_finalize_time_s", 0.0) + finalize_time
        timing["encode_time_s"] = (
            timing.get("encode_time_s", 0.0) + write_time + finalize_time
        )
    if returncode != 0:
        raise RuntimeError(
            f"FFmpeg encoding failed for {out_path.name}: {stderr.strip()}"
        )
