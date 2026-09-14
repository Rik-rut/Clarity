"""Headless provisioning driver — the only implementation of first-run setup.

Two callers, one contract:

* the NSIS installer (``hooks.nsh``) runs it at install time;
* the Tauri boot shell runs it when an installation is incomplete (repair).

It prints one machine-readable line per update::

    PROGRESS <percent>|<phase>|<message>

and writes ``.setup_complete`` only after every step succeeded, so both callers
can trust the marker. Exits 0 on success, 1 on failure after an ``ERROR <msg>``
line.

Runs on the uv-managed interpreter with nothing but the standard library and
this package — no venv is needed to start, because creating it is the job.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

if __package__ in (None, ""):  # invoked as a script by NSIS or the shell
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_upscaler.desktop import models as model_steps
from video_upscaler.desktop import runtime
from video_upscaler.desktop import state as setup_state
from video_upscaler.desktop.context import SetupContext, build_context
from video_upscaler.desktop.gpu import detect_gpu
from video_upscaler.desktop.progress import ProgressTracker

EXIT_OK = 0
EXIT_FAILED = 1
LOCK_NAME = ".provisioning.lock"
STEPS: tuple[str, ...] = ("gpu", "runtime", "models", "verify")


class StepFailure(RuntimeError):
    """A step failed; the message is safe to show the user."""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="clarity-provision")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--resources-dir", type=Path, default=None)
    parser.add_argument("--tier", choices=model_steps.TIERS, default=model_steps.DEFAULT_TIER)
    parser.add_argument("--tensorrt", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore setup.json and redo every step",
    )
    return parser.parse_args(argv)


class LinePrinter(ProgressTracker):
    """ProgressTracker that also emits the wire format both callers parse."""

    def _report(self, message: str = "") -> None:
        snapshot = self.snapshot()
        print(
            f"PROGRESS {snapshot['percent']}|{snapshot['phase']}|{message or snapshot.get('message', '')}",
            flush=True,
        )

    def start(self, step: str, message: Optional[str] = None) -> None:
        super().start(step, message)
        self._report(message or "")

    def phase(self, name: str) -> None:
        super().phase(name)
        self._report()

    def line(self, text: str) -> None:
        super().line(text)
        if text:
            self._report(text)

    def finish(self, error: Optional[str] = None) -> None:
        super().finish(error)
        self._report(error or "")


def _pid_alive(pid: int) -> bool:
    """Liveness check. ``os.kill(pid, 0)`` is NOT safe on Windows: for any
    signal other than CTRL_C_EVENT/CTRL_BREAK_EVENT Python calls
    TerminateProcess, which would kill the installer."""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return True  # assume alive: refusing to start beats double-provisioning
    return str(pid) in (completed.stdout or "")


def acquire_lock(data_dir: Path) -> Optional[int]:
    """Take the provisioning lock; returns a live holder's PID, else None."""
    data_dir.mkdir(parents=True, exist_ok=True)
    lock = data_dir / LOCK_NAME
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = -1
        if _pid_alive(pid):
            return pid
    try:
        lock.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        print(f"ERROR Could not take the provisioning lock: {exc}", flush=True)
        return os.getpid()
    return None


def release_lock(data_dir: Path) -> None:
    try:
        (data_dir / LOCK_NAME).unlink(missing_ok=True)
    except OSError:
        pass


def _step(
    context: SetupContext,
    state: setup_state.SetupState,
    tracker: ProgressTracker,
    name: str,
    work: Callable[[], None],
) -> None:
    """Run one step, persisting its status before and after.

    The state file is written first so a killed run resumes instead of starting
    over — a partially downloaded model set must not be fetched twice.
    """
    state.mark(name, setup_state.RUNNING)
    setup_state.save_state(context.data_dir, state)
    tracker.start(name)
    try:
        work()
    except Exception as exc:  # noqa: BLE001 - reported to the caller, then failed
        message = str(exc) or exc.__class__.__name__
        state.mark(name, setup_state.FAILED, message)
        setup_state.save_state(context.data_dir, state)
        tracker.finish(message)
        raise StepFailure(f"{name} failed: {message}") from exc
    state.mark(name, setup_state.DONE)
    setup_state.save_state(context.data_dir, state)


def run_steps(
    context: SetupContext,
    state: setup_state.SetupState,
    tracker: ProgressTracker,
    args: argparse.Namespace,
) -> None:
    if args.force:
        state.reset_from(STEPS[0])

    if not state.is_done("gpu"):
        _step(context, state, tracker, "gpu", lambda: _step_gpu(state, args))
    if not state.is_done("runtime"):
        _step(
            context, state, tracker, "runtime",
            lambda: runtime.install_runtime(context, tracker, state),
        )
    if not state.is_done("models"):
        _step(
            context, state, tracker, "models",
            lambda: model_steps.install_models(context, tracker, args.tier),
        )
    if not state.is_done("verify"):
        _step(context, state, tracker, "verify", lambda: _step_verify(context, state, tracker))

    state.models_tier = args.tier
    setup_state.save_state(context.data_dir, state)


def _step_gpu(state: setup_state.SetupState, args: argparse.Namespace) -> None:
    info = detect_gpu()
    state.torch_variant = info.torch_variant
    state.gpu_names = list(info.names)
    state.tensorrt = args.tensorrt == "yes" or (args.tensorrt == "auto" and info.has_nvidia)
    if info.error:
        print(f"PROGRESS 0|gpu|{info.error}", flush=True)


def _step_verify(
    context: SetupContext,
    state: setup_state.SetupState,
    tracker: ProgressTracker,
) -> None:
    report = runtime.verify_runtime(context, tracker)
    state.backend = str(report.get("backend", ""))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    context = build_context(data_dir=args.data_dir, resources_dir=args.resources_dir)
    context.ensure_directories()
    state = setup_state.load_state(context.data_dir)
    tracker = LinePrinter()

    holder = acquire_lock(context.data_dir)
    if holder is not None:
        print(
            f"ERROR Another Clarity process (PID {holder}) is already provisioning "
            f"{context.data_dir}. Close it and try again.",
            flush=True,
        )
        return EXIT_FAILED

    try:
        run_steps(context, state, tracker, args)
    except StepFailure as exc:
        print(f"ERROR {exc}", flush=True)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR {exc.__class__.__name__}: {exc}", flush=True)
        return EXIT_FAILED
    finally:
        release_lock(context.data_dir)

    # The marker is the only thing either caller trusts, so it is written last.
    state.mark("complete", setup_state.DONE)
    state.completed_at = setup_state.timestamp()
    setup_state.save_state(context.data_dir, state)
    setup_state.write_marker(context.data_dir)
    tracker.start("complete", "Clarity is ready")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
