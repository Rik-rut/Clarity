"""ncnn Vulkan fallback engine (non-NVIDIA hardware).

Runs the portable realcugan-ncnn-vulkan executable (Tencent ncnn + Vulkan
— works on Intel, AMD and NVIDIA without CUDA/PyTorch). Frames are batched
through the tool as PNG files in chunks (one subprocess per chunk); progress
is parsed from the tool's stdout percentage lines. Tool + models download
with consent from the official GitHub releases (Windows, Ubuntu, and macOS
packages).
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from video_upscaler.config import TOOLS_DIR

NCNN_DIR = TOOLS_DIR / "ncnn"

_TOOLS = {
    "realcugan": {
        "exe_base": "realcugan-ncnn-vulkan",
        "repo": "nihui/realcugan-ncnn-vulkan",
    },
}


def _os_asset_token() -> str:
    """GitHub release asset token for this OS (upstream: -windows/-ubuntu/-macos)."""
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "ubuntu"

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)%")

_TOOL_PARAM_PREFIXES = {
    "realcugan": ("up2x-", "up3x-", "up4x-"),
}


def _exe_name(tool: str) -> str:
    """Platform-specific executable filename for an ncnn tool."""
    base = _TOOLS[tool]["exe_base"]
    return f"{base}.exe" if os.name == "nt" else base


def ncnn_exe(tool: str) -> Path:
    """Path of the ncnn tool executable.

    Official release zips extract into a versioned subdirectory; fall back
    to a recursive search so either layout works (both naming variants are
    accepted so a folder staged from another OS still resolves).
    """
    preferred = TOOLS_DIR / "ncnn" / _exe_name(tool)
    if preferred.is_file():
        return preferred
    base = _TOOLS[tool]["exe_base"]
    root = TOOLS_DIR / "ncnn"
    candidates: list[Path] = []
    for pattern in (_exe_name(tool), base):
        if root.is_dir():
            candidates.extend(sorted(p for p in root.rglob(pattern) if p.is_file()))
    return candidates[0] if candidates else preferred


def ncnn_model_dir(tool: str) -> Path:
    """Find the model folder for a tool.

    The realcugan tool's ``-m`` must point at a model folder named
    ``models-se`` / ``models-nose`` / ``models-pro`` (it validates the name
    and appends the scale/noise filename itself). ``models-se`` holds the
    full "latest" SE model set (every scale x noise level) and matches the
    torch engine's ``up*_latest`` weights, so it is preferred.
    """
    root = TOOLS_DIR / "ncnn"
    if not root.is_dir():
        raise FileNotFoundError(
            f"No ncnn model directory (.param files) found under {root}"
        )
    candidates = sorted({p.parent for p in root.rglob("*.param")})
    if not candidates:
        raise FileNotFoundError(
            f"No ncnn model directory (.param files) found under {root}"
        )
    for candidate in candidates:
        if candidate.name == "models-se" and next(
            candidate.glob("*.param"), None
        ) is not None:
            return candidate
    prefixes = _TOOL_PARAM_PREFIXES[tool]
    for candidate in candidates:
        for prefix in prefixes:
            if next(candidate.glob(f"{prefix}*.param"), None) is not None:
                return candidate
    return candidates[0]


def check_ncnn(tool: str) -> str | None:
    """Return None if the tool + models exist, else a clear error message."""
    missing = []
    if not ncnn_exe(tool).is_file():
        missing.append(str(ncnn_exe(tool)))
    try:
        ncnn_model_dir(tool)
    except FileNotFoundError:
        missing.append(f"model files (.param) under {NCNN_DIR}")
    if not missing:
        return None
    return (
        "ncnn Vulkan fallback is not installed:\n\n"
        + "\n".join(f"  {item}" for item in missing)
        + "\n\nRun the app again and answer [Y] to download it automatically."
    )


def _latest_zip_url(repo: str) -> str:
    """Resolve the newest release .zip asset for this OS via the GitHub API."""
    token = _os_asset_token()
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    request = urllib.request.Request(api, headers={"User-Agent": "clarity-upscaler"})
    with urllib.request.urlopen(request, timeout=60) as response:
        import json

        payload = json.loads(response.read().decode("utf-8"))
    for asset in payload.get("assets", []):
        url = asset.get("browser_download_url", "")
        if token in url.lower() and url.lower().endswith(".zip"):
            return url
    raise RuntimeError(f"No {token} release asset found for {repo}.")


def _make_executables(tool: str) -> None:
    """chmod +x the extracted binary (and any bundled helpers) on POSIX."""
    if os.name == "nt":
        return
    base = _TOOLS[tool]["exe_base"]
    root = TOOLS_DIR / "ncnn"
    if not root.is_dir():
        return
    for candidate in root.rglob(base):
        if candidate.is_file():
            mode = candidate.stat().st_mode
            candidate.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def download_ncnn(tool: str) -> list[Path]:
    """Download + extract the portable tool package into tools/ncnn/."""
    repo = _TOOLS[tool]["repo"]
    url = _latest_zip_url(repo)
    NCNN_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="clarity_ncnn_"))
    zip_path = temp_dir / "tool.zip"
    try:
        print(f"Downloading {tool} ncnn Vulkan tool (~40 MB)...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"Failed to download {tool} ncnn tool:\n{exc}\n\n"
                f"Download it manually from:\n{url}\n"
                f"and extract it into:\n{NCNN_DIR}"
            ) from exc
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(NCNN_DIR)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Downloaded {tool} archive is corrupt:\n{exc}") from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    _make_executables(tool)
    extracted = [ncnn_exe(tool)] if ncnn_exe(tool).is_file() else []
    if not extracted:
        raise RuntimeError(f"Extracted archive for {tool} contains no executable.")
    print(f"Installed {tool} ncnn Vulkan tool into {NCNN_DIR}")
    return extracted


def _build_command(tool: str, args: dict, in_dir: Path, out_dir: Path) -> list[str]:
    """Build the CLI invocation for one chunk."""
    command = [
        str(ncnn_exe(tool)),
        "-i", str(in_dir),
        "-o", str(out_dir),
        "-m", str(ncnn_model_dir(tool)),
        "-f", "png",
    ]
    if "s" in args:
        command += ["-s", str(args["s"])]
    if "n" in args:
        command += ["-n", str(args["n"])]
    if tool == "realcugan":
        # syncgap 0 matches the torch engine's cache_mode=0 (quality parity).
        command += ["-c", "0"]
    return command


class NCNNEngine:
    """ncnn Vulkan upscaling engine for one profile (subprocess batches)."""

    chunked = True

    def __init__(self, profile: str) -> None:
        from video_upscaler.models import ncnn_args_for_profile

        self._tool = "realcugan"
        self._args = ncnn_args_for_profile(profile)

    def enhance_chunk(
        self,
        frames: list[np.ndarray],
        on_progress=None,
    ) -> list[np.ndarray]:
        """Enhance a chunk of RGB frames via one tool invocation.

        ``on_progress(fraction)`` is called with 0.0-1.0 while the tool
        reports percentages; a final 1.0 call always happens on success.
        """
        import cv2

        if not frames:
            return []
        work_dir = Path(tempfile.mkdtemp(prefix="clarity_ncnn_"))
        try:
            in_dir = work_dir / "in"
            out_dir = work_dir / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            for index, frame in enumerate(frames):
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(in_dir / f"{index:08d}.png"), bgr)

            command = _build_command(self._tool, self._args, in_dir, out_dir)
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=None
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ncnn Vulkan tool failed ({self._tool}): "
                    f"{(result.stderr or '').strip()}"
                )
            if on_progress is not None:
                for match in _PERCENT_RE.finditer(result.stdout):
                    on_progress(float(match.group(1)) / 100.0)

            enhanced = []
            for path in sorted(out_dir.glob("*.png")):
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise RuntimeError(f"ncnn tool produced an unreadable file: {path.name}")
                enhanced.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if len(enhanced) != len(frames):
                raise RuntimeError(
                    f"ncnn tool produced {len(enhanced)} frames for "
                    f"{len(frames)} inputs."
                )
            if on_progress is not None:
                on_progress(1.0)
            return enhanced
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Single-frame convenience wrapper (spawns one invocation)."""
        return self.enhance_chunk([frame])[0]
