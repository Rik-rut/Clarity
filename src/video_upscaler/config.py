"""Central configuration: directories, extensions, defaults, env overrides.

All paths are derived from the project root (BASE_DIR) unless overridden
via the documented CLARITY_* environment variables.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Project root: <root>/src/video_upscaler/config.py -> parents[2] is the root
BASE_DIR = Path(__file__).resolve().parents[2]


def _env_dir(name: str, default: Path) -> Path:
    """Return an env-overridable directory path."""
    value = os.environ.get(name)
    return Path(value) if value else default


INPUT_DIR = _env_dir("CLARITY_INPUT_DIR", BASE_DIR / "input")
OUTPUT_DIR = _env_dir("CLARITY_OUTPUT_DIR", BASE_DIR / "output")
MODELS_DIR = _env_dir("CLARITY_MODELS_DIR", BASE_DIR / "models")

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}

# "auto" | "cuda" | "cpu" — env CLARITY_DEVICE may force a device
PREFERRED_DEVICE = os.environ.get("CLARITY_DEVICE", "auto").lower()

# Warn before producing output wider than this (env CLARITY_MAX_OUTPUT_WIDTH)
MAX_OUTPUT_WIDTH_WARNING = int(os.environ.get("CLARITY_MAX_OUTPUT_WIDTH", "7680"))

# Real-CUGAN tiling: 0 = whole frame, 1 = split the long edge in half,
# >=2 = split both dimensions into N tiles (lower VRAM usage).
CUGAN_TILE_MODE = int(os.environ.get("CLARITY_CUGAN_TILE", "1"))

# Real-CUGAN cache mode: 0 = normal, 2 = fast_rough, 3 = gap_sync
# (env CLARITY_CUGAN_CACHE — vendor cache_mode argument).
CUGAN_CACHE_MODE = int(os.environ.get("CLARITY_CUGAN_CACHE", "0"))

# NVENC hardware encode: set CLARITY_NVENC=1 to enable (default off —
# NVENC contends with CUDA inference on the torch path).
USE_NVENC = os.environ.get("CLARITY_NVENC", "0").lower() not in ("0", "false", "off")

TOOLS_DIR = _env_dir("CLARITY_TOOLS_DIR", BASE_DIR / "tools")

# Hugging Face mirror for the official AMT pretrained checkpoints
# (https://huggingface.co/lalala125/AMT). Checkpoint filenames are appended.
AMT_CKPT_BASE = os.environ.get(
    "CLARITY_AMT_CKPT_BASE",
    "https://huggingface.co/lalala125/AMT/resolve/main",
)

# "auto" | "torch" | "ncnn" — env CLARITY_BACKEND may force the backend
BACKEND_PREF = os.environ.get("CLARITY_BACKEND", "auto").lower()

# ncnn Vulkan: number of frames processed per tool invocation
NCNN_CHUNK = int(os.environ.get("CLARITY_NCNN_CHUNK", "120"))

# Triton fast path: set CLARITY_TRITON=0 to disable custom kernels
TRITON_ENABLED = os.environ.get("CLARITY_TRITON", "1").lower() not in ("0", "false", "off")

# AMT frame interpolation: number of source frames processed per segment.
# Interpolation must hold many intermediate frames in memory, so long videos
# are split into segments (with 1-frame overlap) to bound peak memory.
# 0 disables segmentation and processes the whole video at once.
AMT_SEGMENT_FRAMES = int(os.environ.get("CLARITY_AMT_SEGMENT", "120"))


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "off")


# AMT backend selection is separate from CLARITY_BACKEND so Real-CUGAN keeps
# its existing behavior. When CLARITY_AMT_BACKEND is absent, the CLI/backend
# preference is consulted by the AMT selector.
AMT_BACKEND_PREF = os.environ.get("CLARITY_AMT_BACKEND", "auto").lower()
AMT_BACKEND_EXPLICIT = "CLARITY_AMT_BACKEND" in os.environ
AMT_PRECISION = os.environ.get("CLARITY_AMT_PRECISION", "fp32").lower()
AMT_BATCH = os.environ.get("CLARITY_AMT_BATCH", "auto").lower()
AMT_ENGINE_BUILD = _env_bool("CLARITY_AMT_ENGINE_BUILD", "1")
AMT_ENGINE_CACHE = _env_bool("CLARITY_AMT_ENGINE_CACHE", "1")
AMT_TRT_WORKSPACE_GIB = float(os.environ.get("CLARITY_AMT_TRT_WORKSPACE_GIB", "1"))
AMT_ONNX_OPSET = int(os.environ.get("CLARITY_AMT_ONNX_OPSET", "17"))

# MultiPassDedup configuration
DEDUP_MODELS_DIR = _env_dir("CLARITY_DEDUP_MODELS_DIR", MODELS_DIR / "multipassdedup")
DEDUP_MODEL_DEFAULT = os.environ.get("CLARITY_DEDUP_MODEL", "gmfss").lower()
DEDUP_NPASS_DEFAULT = os.environ.get("CLARITY_DEDUP_NPASS", "auto").lower()
DEDUP_SCALE_DEFAULT = float(os.environ.get("CLARITY_DEDUP_SCALE", "1.0"))
DEDUP_SCDET_DEFAULT = _env_bool("CLARITY_DEDUP_SCDET", "1")
DEDUP_SCDET_THRESHOLD = float(os.environ.get("CLARITY_DEDUP_SCDET_THRESHOLD", "0.3"))
DEDUP_BACKEND_PREF = os.environ.get("CLARITY_DEDUP_BACKEND", "torch").lower()


def ffmpeg_path() -> str | None:
    """Return the FFmpeg binary path, or None if not found."""
    env_path = os.environ.get("CLARITY_FFMPEG")
    if env_path:
        return env_path
    return shutil.which("ffmpeg")


def ensure_directories() -> None:
    """Create input/, output/, and models/ directories if they do not exist."""
    for directory in (INPUT_DIR, OUTPUT_DIR, MODELS_DIR, DEDUP_MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

