"""Environment provisioning driven by ``uv`` plus an import smoke test.

Every step is a subprocess: the wizard never imports the packages it is
installing, so a broken wheel can take down a step but never the server that
reports it.

What actually goes wrong on Windows
----------------------------------
``uv venv`` writes ``env/Scripts/python.exe`` by moving a temporary file over the
target. Windows answers ``ERROR_ACCESS_DENIED`` when that target is an image
already in use, which reproduces the failure this module guards against::

    $ uv venv --allow-existing --python <base> env
    error: Failed to create virtual environment
      Caused by: Failed to create Python executable link
      Caused by: Failed to persist temporary file to .../env/Scripts/python.exe:
                 failed to persist temporary file: Access is denied. (os error 5)

The holder is usually a leftover Clarity server started from that exact
interpreter, or an antivirus that opened the file moments after it appeared. Separately, a
uv invocation can pick a base interpreter from outside the data directory, and
``uv pip install`` will then build an environment of its own — which is how a run
can report "failed" while a working but unaccounted env exists.

Design responses, each tied to one of those:

* an environment that can already import the stack is **reused**, never re-linked
  over and never deleted to make a point about lineage;
* one that cannot be reused is deleted only if the delete succeeds, otherwise the
  user is told which process to close instead of being shown ``os error 5``;
* a genuine access-denied is retried once after a delay, which covers the
  second-scale antivirus hold;
* uv is pinned to the managed interpreter and the created env's base is checked,
  so an interpreter nobody chose cannot sneak into the data folder.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from video_upscaler.desktop.context import (
    SetupContext,
    is_inside,
    venv_home,
    venv_is_ours,
)
from video_upscaler.desktop.progress import ProgressTracker

PYTORCH_CU126_INDEX = "https://download.pytorch.org/whl/cu126"
VERIFY_IMPORTS = "torch, video_upscaler, cv2, fastapi"
_ERROR_TAIL_LINES = 12
# Windows reports a locked or blocked executable as "Access is denied. (os error 5)".
_ACCESS_DENIED_MARKERS = ("access is denied", "os error 5", "permission denied")
_RETRY_DELAY_SECONDS = 1.5


class SetupError(RuntimeError):
    """A provisioning step failed; the message is safe to show in the wizard."""


def _hide_window_options() -> Dict[str, object]:
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    return {
        "startupinfo": startupinfo,
        "creationflags": subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
    }


def _mask_secrets(text: str) -> str:
    masked = text
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CLARITY_HF_TOKEN"):
        value = os.environ.get(name)
        if value:
            masked = masked.replace(value, "***")
    return masked


def run_streamed(
    command: Sequence[str],
    tracker: ProgressTracker,
    *,
    context: SetupContext,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stage: str = "step",
) -> str:
    """Run ``command``, streaming combined output into ``tracker``.

    Returns the captured output. Raises :class:`SetupError` with the failing
    command and the last output lines when the exit code is non-zero.
    """
    argv = [str(part) for part in command]
    tracker.line(f"$ {' '.join(argv)}")
    collected: List[str] = []

    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env if env is not None else context.child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **_hide_window_options(),
        )
    except OSError as exc:
        tracker.finish(f"Failed to start {argv[0]}: {exc}")
        raise SetupError(f"Failed to start {argv[0]}: {exc}") from exc

    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        line = _mask_secrets(line)
        collected.append(line)
        tracker.line(line)

    returncode = process.wait()
    output = "\n".join(collected)
    if returncode != 0:
        tail = " | ".join(collected[-_ERROR_TAIL_LINES:]) or "no output"
        message = f"{stage} failed (exit {returncode}): {tail}"
        tracker.finish(message)
        raise SetupError(message)
    return output


def _venv_interpreter(env_dir: Path) -> Path:
    """The interpreter inside a venv, Windows layout first.

    Mirrors :func:`video_upscaler.desktop.state.venv_python` but takes the venv
    directory itself, because the probe judges environments that may not be the
    one this data root currently uses.
    """
    windows = env_dir / "Scripts" / "python.exe"
    if windows.is_file():
        return windows
    unix = env_dir / "bin" / "python"
    if unix.is_file():
        return unix
    return windows


def _is_access_denied(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _ACCESS_DENIED_MARKERS)


def _venv_imports_the_stack(env_dir: Path) -> bool:
    """Ask an existing environment whether it still works.

    Only the interpreter itself can answer. ``pyvenv.cfg`` records which base
    interpreter the env was made from, and an env whose base has been reinstalled
    or moved stops being able to import anything, so this is the check that
    separates "unusual lineage" from "broken".
    """
    python = _venv_interpreter(env_dir)
    if not python.is_file():
        return False
    try:
        completed = subprocess.run(
            [str(python), "-c", f"import {VERIFY_IMPORTS}"],
            capture_output=True,
            text=True,
            timeout=240,
            **_hide_window_options(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[desktop] probing {python} failed: {exc}")
        return False
    return completed.returncode == 0


def _require_venv_interpreter(context: SetupContext) -> None:
    """Fail closed rather than letting uv invent an environment of its own.

    This is the check that was missing: after ``uv venv`` failed, the next step
    (``uv pip install --python <that venv>``) silently built an environment from
    an interpreter elsewhere, so a step reported "failed" while an unaccounted
    env sat in the data folder.
    """
    if not _venv_interpreter(context.env_dir).is_file():
        raise SetupError(
            f"The virtual environment at {context.env_dir} has no interpreter. "
            "Close any other Clarity window and press Retry."
        )
    if not (context.env_dir / "pyvenv.cfg").is_file():
        raise SetupError(
            f"{context.env_dir} exists but is not a virtual environment "
            "(no pyvenv.cfg). Move or delete that folder and press Retry."
        )


def _require_managed_creation(context: SetupContext) -> None:
    """A freshly created env must come from the interpreter in the data folder.

    An env that already existed and merely has unusual lineage is handled by the
    adopt path in :func:`create_venv`; something we created a second ago has no
    excuse, and a surprise base here means uv resolved a different interpreter
    than the one we handed it.
    """
    _require_venv_interpreter(context)

    home = venv_home(context.env_dir)
    if home is None:
        return
    if not is_inside(home, context.python_install_dir):
        raise SetupError(
            f"Clarity created its environment on top of {home}, outside the managed "
            f"runtime folder ({context.python_install_dir}). Close Clarity, delete "
            f"{context.env_dir}, and start again."
        )


def _remove_stale_venv(context: SetupContext, reason: str, tracker: ProgressTracker) -> None:
    """Delete an env that cannot be reused, so creation starts from a clean slate.

    Only the generated environment is touched — never media, models or logs — and
    only after :func:`_venv_imports_the_stack` proved it unusable, because
    ``uv venv --allow-existing`` rewrites ``Scripts/python.exe`` in place and
    Windows refuses that with ERROR_ACCESS_DENIED while the interpreter runs.
    """
    tracker.line(f"{reason} Recreating it now; installed packages will be re-downloaded.")
    try:
        shutil.rmtree(context.env_dir)
    except OSError as exc:
        raise SetupError(
            f"Clarity needs to replace {context.env_dir} but Windows refused: "
            f"{getattr(exc, 'strerror', None) or exc}. Another copy of Clarity, or "
            "its server, is still running from that folder. Close it (or end "
            "python.exe in Task Manager) and press Retry."
        ) from exc


def create_venv(
    context: SetupContext,
    tracker: ProgressTracker,
    probe=None,
) -> None:
    """Create the isolated venv in the data directory, reusing anything that works.

    Never re-link over a live interpreter: the error Windows returns for that
    ("Failed to persist temporary file … Access is denied") is unrecoverable from
    the user's point of view, and it is what a leftover studio server from an
    earlier attempt causes. So an existing env is judged by whether it can import
    the stack, and only genuinely broken ones are replaced.
    """
    if probe is None:
        probe = _venv_imports_the_stack
    tracker.phase("venv")
    context.ensure_directories()

    if context.env_dir.is_dir():
        if venv_is_ours(context.env_dir, context.python_install_dir):
            tracker.line("Reusing the virtual environment that is already here.")
            return
        if probe(context.env_dir):
            tracker.line(
                "Reusing the environment from an earlier run: its base interpreter "
                "lives outside the data folder, but everything Clarity needs is "
                "installed and working."
            )
            return
        home = venv_home(context.env_dir)
        _remove_stale_venv(
            context,
            f"The existing environment is unusable (interpreter home: {home or 'missing'}).",
            tracker,
        )

    command = [
        str(context.uv_exe),
        "venv",
        "--allow-existing",
        # Never fall back to a host or roaming interpreter.
        "--python-preference",
        "only-managed",
        "--python",
        str(context.python_executable),
        str(context.env_dir),
    ]
    tracker.line("Creating the isolated virtual environment…")

    for attempt in (1, 2):
        try:
            run_streamed(command, tracker, context=context, stage="venv")
            break
        except SetupError as exc:
            if _is_access_denied(str(exc)):
                if attempt == 1:
                    tracker.line(
                        "Windows refused to write the venv interpreter — retrying in a moment…"
                    )
                    time.sleep(_RETRY_DELAY_SECONDS)
                    continue
                raise SetupError(
                    f"{exc} Clarity could not create the interpreter inside "
                    f"{context.env_dir}. Close any other Clarity window (a studio server left "
                    "running from a previous attempt holds that file), allow python.exe if your "
                    "antivirus prompts, then press Retry."
                ) from exc
            raise

    _require_managed_creation(context)


def install_dependencies(
    context: SetupContext,
    tracker: ProgressTracker,
    torch_variant: str,
    tensorrt: bool = False,
) -> None:
    """Install the project and its dependencies into the provisioned venv."""
    # uv will create an environment for itself if the interpreter is missing;
    # that is how a failed venv step used to end up with a roaming interpreter.
    _require_venv_interpreter(context)

    base = [
        str(context.uv_exe),
        "pip",
        "install",
        "--python",
        str(context.venv_python),
    ]
    if torch_variant == "cu126":
        base += ["--extra-index-url", PYTORCH_CU126_INDEX]

    tracker.phase("dependencies")
    if tensorrt:
        tracker.line(
            f"Installing Clarity, PyTorch ({torch_variant}) and TensorRT — the long step…"
        )
        try:
            run_streamed(
                base + ["-e", f"{context.app_dir}[tensorrt]"],
                tracker,
                context=context,
                stage="dependency install",
            )
            return
        except SetupError as exc:
            tracker.line(f"TensorRT could not be installed ({exc}). Continuing with CUDA.")
    else:
        tracker.line(f"Installing Clarity and PyTorch ({torch_variant}) — this is the long step…")

    run_streamed(
        base + ["-e", str(context.app_dir)],
        tracker,
        context=context,
        stage="dependency install",
    )


def install_runtime(context: SetupContext, tracker: ProgressTracker, state) -> None:
    """Venv plus dependencies, honouring the TensorRT decision in ``state``."""
    create_venv(context, tracker)
    install_dependencies(
        context, tracker, state.torch_variant or "cpu", tensorrt=bool(state.tensorrt)
    )


def verify_runtime(
    context: SetupContext,
    tracker: ProgressTracker,
) -> Dict[str, object]:
    """Import the heavy stack in the new venv and report CUDA availability."""
    python = context.venv_python
    if not python.is_file():
        raise SetupError(f"Provisioned interpreter is missing: {python}")

    code = (
        f"import {VERIFY_IMPORTS}\n"
        "import torch\n"
        "from video_upscaler.backend import detect_backend\n"
        "print('imports ok', torch.__version__, 'cuda', torch.cuda.is_available())\n"
        "print('backend', detect_backend())\n"
    )
    tracker.line("Verifying the provisioned environment…")
    output = run_streamed(
        [str(python), "-c", code],
        tracker,
        context=context,
        cwd=str(context.data_dir),
        stage="verification",
    )
    return {
        "ok": True,
        "output": output.strip(),
        "cuda": "cuda True" in output,
        "torch_version": _first_version(output),
        "backend": _first_field(output, "backend"),
    }


def _first_field(output: str, key: str) -> str:
    """Value printed as ``key value`` by the verification probe, else ''."""
    for line in output.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0] == key:
            return parts[1]
    return ""


def _first_version(output: str) -> str:
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "imports":
            return parts[2]
    return ""


@dataclass
class EnvironmentInfo:
    """What the wizard reports about the provisioned environment."""

    venv_python: str
    exists: bool
    detail: str = ""

    @classmethod
    def probe(cls, context: SetupContext) -> "EnvironmentInfo":
        python = context.venv_python
        return cls(
            venv_python=str(python),
            exists=python.is_file(),
            detail="" if python.is_file() else "provisioned by setup",
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "venv_python": self.venv_python,
            "exists": self.exists,
            "detail": self.detail,
        }
