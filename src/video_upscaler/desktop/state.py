"""Setup state: ``setup.json`` plus the ``.setup_complete`` handoff marker.

The marker is the single contract shared with the Tauri shell: both sides
agree the environment is ready when it exists *and* the provisioned venv
interpreter is on disk. State is written atomically so an interrupted
download never leaves a half-written file behind.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

STATE_VERSION = 1
SETUP_FILE_NAME = "setup.json"
MARKER_FILE_NAME = ".setup_complete"

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# Ordered provisioning steps owned by the wizard (the Python runtime itself is
# installed by the shell beforehand, since nothing can execute without it).
STEPS: tuple[str, ...] = ("gpu", "runtime", "models", "verify", "complete")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp() -> str:
    """Current UTC timestamp in the format stored in ``setup.json``."""
    return _now()


def default_data_dir() -> Path:
    r"""Fallback data root: ``%LOCALAPPDATA%\Clarity`` on Windows.

    The desktop shell always passes ``CLARITY_DATA_DIR`` — the folder it was
    installed into — so this only matters for the CLI, tests and a read-only
    install. There is deliberately no remembered-location file: one rule,
    computed the same way by the installer and the app, cannot disagree with
    itself.
    """
    override = os.environ.get("CLARITY_DATA_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "Clarity"
        return Path.home() / "AppData" / "Local" / "Clarity"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "clarity"
    return Path.home() / ".local" / "share" / "clarity"


@dataclass
class SetupState:
    """Mutable provisioning state, mirrored to ``setup.json``."""

    version: int = STATE_VERSION
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    torch_variant: str = ""
    gpu_names: List[str] = field(default_factory=list)
    models_tier: str = ""
    tensorrt: bool = False
    backend: str = ""
    completed_at: str = ""
    steps: Dict[str, str] = field(default_factory=lambda: {s: PENDING for s in STEPS})
    errors: Dict[str, str] = field(default_factory=dict)

    def status(self, step: str) -> str:
        return self.steps.get(step, PENDING)

    def is_done(self, step: str) -> bool:
        return self.status(step) == DONE

    def mark(self, step: str, status: str, error: Optional[str] = None) -> None:
        self.steps[step] = status
        if status == FAILED and error:
            self.errors[step] = error
        elif step in self.errors:
            del self.errors[step]
        self.updated_at = _now()

    def reset_from(self, step: str) -> List[str]:
        """Clear ``step`` and everything after it. Returns the reset step names."""
        try:
            start = STEPS.index(step)
        except ValueError:
            return []
        reset = list(STEPS[start:])
        for name in reset:
            self.steps[name] = PENDING
            self.errors.pop(name, None)
        if "complete" in reset:
            self.completed_at = ""
        self.updated_at = _now()
        return reset

    @property
    def complete(self) -> bool:
        return self.status("complete") == DONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "torch_variant": self.torch_variant,
            "gpu_names": list(self.gpu_names),
            "models_tier": self.models_tier,
            "tensorrt": self.tensorrt,
            "backend": self.backend,
            "completed_at": self.completed_at,
            "steps": dict(self.steps),
            "errors": dict(self.errors),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SetupState":
        state = cls()
        state.version = int(payload.get("version", STATE_VERSION))
        state.created_at = str(payload.get("created_at", state.created_at))
        state.updated_at = str(payload.get("updated_at", state.updated_at))
        state.torch_variant = str(payload.get("torch_variant", ""))
        gpu_names = payload.get("gpu_names") or []
        state.gpu_names = [str(n) for n in gpu_names] if isinstance(gpu_names, list) else []
        state.models_tier = str(payload.get("models_tier", ""))
        state.tensorrt = bool(payload.get("tensorrt", False))
        state.backend = str(payload.get("backend", "") or "")
        state.completed_at = str(payload.get("completed_at", ""))
        steps = payload.get("steps") or {}
        if isinstance(steps, dict):
            for name in STEPS:
                value = steps.get(name)
                if value in (PENDING, RUNNING, DONE, FAILED):
                    state.steps[name] = str(value)
        errors = payload.get("errors") or {}
        if isinstance(errors, dict):
            state.errors = {str(k): str(v) for k, v in errors.items()}
        return state


def setup_file(data_dir: Path) -> Path:
    return data_dir / SETUP_FILE_NAME


def marker_file(data_dir: Path) -> Path:
    return data_dir / MARKER_FILE_NAME


def venv_dir(data_dir: Path) -> Path:
    return data_dir / "env"


def venv_python(data_dir: Path) -> Path:
    """Interpreter inside the provisioned venv (Windows layout first)."""
    windows = venv_dir(data_dir) / "Scripts" / "python.exe"
    unix = venv_dir(data_dir) / "bin" / "python"
    if windows.is_file():
        return windows
    if unix.is_file():
        return unix
    return windows if os.name == "nt" else unix


def load_state(data_dir: Path) -> SetupState:
    path = setup_file(data_dir)
    if not path.is_file():
        return SetupState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return SetupState()
    if not isinstance(payload, dict):
        return SetupState()
    return SetupState.from_dict(payload)


def save_state(data_dir: Path, state: SetupState) -> None:
    """Write ``setup.json`` atomically (temp file + replace)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(data_dir),
        prefix=".setup-",
        suffix=".tmp",
        delete=False,
    )
    tmp = Path(handle.name)
    try:
        with handle:
            json.dump(state.to_dict(), handle, indent=2, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, setup_file(data_dir))
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def is_setup_complete(data_dir: Path) -> bool:
    """Marker present *and* the provisioned interpreter exists on disk."""
    return marker_file(data_dir).is_file() and venv_python(data_dir).is_file()


def write_marker(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    marker_file(data_dir).write_text("1", encoding="utf-8")
