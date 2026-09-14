r"""Filesystem layout for the desktop data directory.

The layout is owned by Python: the shell passes a single ``CLARITY_DATA_DIR``
environment variable and everything else (``input/``, ``output/``, ``models/``,
``tools/``, ``.cache/``, ``logs/``, ``env/``, ``python/``, ``setup.json``)
derives from it. Keeping one variable instead of seven means the desktop shell
and the app can never disagree about where user media lives.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from video_upscaler.desktop import state as setup_state

MEDIA_DIRS: tuple[str, ...] = ("input", "output", "models", "tools", ".cache", "logs")


@dataclass(frozen=True)
class SetupContext:
    """Resolved locations and tools needed to provision a desktop install."""

    data_dir: Path
    resources_dir: Path
    static_dir: Path
    python_executable: Path

    @property
    def uv_exe(self) -> Path:
        return resolve_uv_exe(self.resources_dir)

    @property
    def env_dir(self) -> Path:
        return setup_state.venv_dir(self.data_dir)

    @property
    def venv_python(self) -> Path:
        return setup_state.venv_python(self.data_dir)

    @property
    def python_install_dir(self) -> Path:
        return self.data_dir / "python"

    @property
    def app_dir(self) -> Path:
        """Directory holding ``pyproject.toml``, ``main.py`` and ``src/``."""
        return self.resources_dir

    @property
    def main_py(self) -> Path:
        return self.resources_dir / "main.py"

    @property
    def package_src(self) -> Path:
        return self.resources_dir / "src"

    @property
    def setup_file(self) -> Path:
        return setup_state.setup_file(self.data_dir)

    @property
    def marker_file(self) -> Path:
        return setup_state.marker_file(self.data_dir)

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def media_dir(self, name: str) -> Path:
        return self.data_dir / name

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for name in MEDIA_DIRS:
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)

    def child_env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Environment for provisioning subprocesses (uv, the venv's python)."""
        env = dict(os.environ)
        env["CLARITY_DATA_DIR"] = str(self.data_dir)
        env["CLARITY_DESKTOP_MODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # Bundled ffmpeg/ffprobe must win over anything on the host PATH.
        env["PATH"] = _prepend_path(str(self.resources_dir), env.get("PATH", ""))
        for name in ("CLARITY_FFMPEG", "CLARITY_FFPROBE"):
            binary = self.resources_dir / name.removeprefix("CLARITY_")
            if binary.is_file():
                env[name] = str(binary)
        env["PYTHONPATH"] = _prepend_path(
            str(self.package_src), env.get("PYTHONPATH", "")
        )
        # Keep uv's managed runtimes inside the data dir, never the host cache.
        env["UV_PYTHON_INSTALL_DIR"] = str(self.python_install_dir)
        if extra:
            env.update(extra)
        return env


def _prepend_path(entry: str, existing: str) -> str:
    if not existing:
        return entry
    separator = ";" if os.name == "nt" else ":"
    return f"{entry}{separator}{existing}"


def resolve_uv_exe(resources_dir: Path) -> Path:
    """Bundled ``uv`` first, then the host PATH, then a bare name as last resort."""
    for candidate in (resources_dir / ("uv.exe" if os.name == "nt" else "uv"),):
        if candidate.is_file():
            return candidate
    found = shutil.which("uv")
    if found:
        return Path(found)
    return Path("uv.exe" if os.name == "nt" else "uv")


def find_python_executable(directory: Path) -> Optional[Path]:
    """Locate a uv-managed CPython inside an install directory."""
    if not directory.is_dir():
        return None
    for candidate in (
        directory / "python.exe",
        directory / "bin" / "python",
        directory / "Scripts" / "python.exe",
    ):
        if candidate.is_file():
            return candidate
    for entry in sorted(directory.iterdir()):
        # `.temp` and `.lock` are uv's own staging artefacts, not runtimes.
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        for candidate in (
            entry / "python.exe",
            entry / "install" / "python.exe",
            entry / "Scripts" / "python.exe",
            entry / "bin" / "python",
        ):
            if candidate.is_file():
                return candidate
    return None


def canonical_executable(path: Path) -> Path:
    """Resolve a uv-managed interpreter to its real, non-reparse path.

    ``uv python install --install-dir`` keeps the runtime in a versioned folder
    (``cpython-3.11.15-windows-x86_64-none``) and adds a versionless alias
    (``cpython-3.11-windows-x86_64-none``) that is a directory symlink. Sorted
    scanning picks the alias first, because ``-`` sorts before ``.``, and handing
    that to ``uv venv`` makes uv create the venv interpreter link *through* a
    reparse point — which Windows refuses with ERROR_ACCESS_DENIED (os error 5).
    """
    try:
        resolved = Path(os.path.realpath(str(path)))
    except OSError:
        return path
    return resolved if resolved.is_file() else path


def venv_home(env_dir: Path) -> Optional[Path]:
    """The base interpreter directory recorded in ``pyvenv.cfg``, if any."""
    cfg = env_dir / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    try:
        lines = cfg.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "home":
            value = value.strip()
            return Path(value) if value else None
    return None


def is_inside(path: Path, parent: Path) -> bool:
    """Case-insensitive containment on Windows, exact elsewhere."""
    target = os.path.normcase(str(path))
    prefix = os.path.normcase(str(parent))
    return target.startswith(prefix + os.sep)


def venv_is_ours(env_dir: Path, managed_root: Path) -> bool:
    """True when ``env_dir`` holds a venv with its own interpreter, based on a
    runtime that lives inside Clarity's data directory.

    Both halves matter. An environment whose ``pyvenv.cfg`` points at uv's
    roaming ``%APPDATA%\\uv\\python`` is not self-contained, and one without
    ``Scripts/python.exe`` is only half built.
    """
    has_python = any(
        candidate.is_file()
        for candidate in (env_dir / "Scripts" / "python.exe", env_dir / "bin" / "python")
    )
    if not has_python:
        return False
    home = venv_home(env_dir)
    return home is not None and is_inside(home, managed_root)


def default_static_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "web" / "static"


def default_resources_dir() -> Path:
    """Directory containing the staged app (``src/`` + ``pyproject.toml``)."""
    candidates = [
        Path(__file__).resolve().parents[3],          # <resources>/src/video_upscaler/desktop
        Path.cwd(),
        Path.cwd().parent,
    ]
    if getattr(sys, "frozen", False):
        candidates.insert(0, Path(sys.executable).resolve().parent)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    return candidates[0]


def build_context(
    data_dir: Optional[Path] = None,
    resources_dir: Optional[Path] = None,
    static_dir: Optional[Path] = None,
) -> SetupContext:
    return SetupContext(
        data_dir=Path(data_dir or os.environ.get("CLARITY_DATA_DIR") or setup_state.default_data_dir()),
        resources_dir=Path(resources_dir or default_resources_dir()),
        static_dir=Path(static_dir or default_static_dir()),
        python_executable=canonical_executable(Path(sys.executable)),
    )
