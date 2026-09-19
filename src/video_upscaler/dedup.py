"""Interactive prompts, plan building, and execution orchestration for MultiPassDedup."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import questionary

from video_upscaler import config
from video_upscaler.dedup_backend import (
    DEDUP_MODEL_NAMES,
    DEDUP_MODELS,
    check_dedup_weights,
    detect_dedup_device,
    parse_npass,
    validate_model_type,
)


def _interactive() -> bool:
    """Return True when stdin and stdout are interactive TTYs."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_choice(title: str, options: dict[str, str]) -> str:
    """Arrow-key single select; returns the selected option key."""
    if not _interactive():
        return next(iter(options))
    try:
        chosen = questionary.select(title, choices=list(options.values())).ask()
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(0)
    if chosen is None:
        raise SystemExit(0)
    return next(key for key, label in options.items() if label == chosen)


def _prompt_confirm(title: str) -> bool:
    """Yes/no confirmation; False when non-interactive."""
    if not _interactive():
        return False
    try:
        return bool(questionary.confirm(title, default=False).ask())
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(0)


def _cadence_options() -> dict[str, str]:
    return {
        "auto": "Auto-detect (recommended) — automatic cadence detection",
        "2": "On Twos — animation on 2s (e.g. 12 fps on 24 fps)",
        "3": "On Threes — animation on 3s (e.g. 8 fps on 24 fps)",
    }


def _model_options() -> dict[str, str]:
    return {
        "gmfss": "GMFSS (Fortuna) | Best quality (default)",
        "rife": "RIFE (Practical-RIFE) | Faster",
    }


def _factor_options() -> dict[str, str]:
    return {
        "2": "2x — double frame rate (e.g. 24 -> 48 fps)",
        "4": "4x — quadruple frame rate (e.g. 24 -> 96 fps)",
        "8": "8x — octuple frame rate (e.g. 24 -> 192 fps)",
    }


def _dedup_entries_for(model_type: str) -> list[dict]:
    """Manifest entries required by one MultiPassDedup model type."""
    from video_upscaler import modelhub

    entries = modelhub.entries(group="dedup")
    if model_type == "gmfss":
        return [e for e in entries if str(e["dest"]).startswith("train_log_pg104/")]
    if model_type == "rife":
        return [e for e in entries if str(e["dest"]) == "rife48.pkl"]
    raise ValueError(f"Unknown MultiPassDedup model: {model_type}")


def ensure_dedup_weights(model_type: str, auto_download: bool = False) -> None:
    """Verify weights exist, offering a hub download on first use.

    Only entries whose destination file is absent are downloaded, so a
    partially-installed model gets repaired instead of crashing inference.
    Non-interactive runs raise SystemExit(1) when weights are missing unless
    auto_download=True.
    """
    model_type = validate_model_type(model_type)
    missing = check_dedup_weights(model_type)
    if not missing:
        return

    from video_upscaler import modelhub
    needed = modelhub.missing_entries(_dedup_entries_for(model_type))

    if auto_download and needed:
        for entry in needed:
            modelhub.install_entry(entry)
        return

    if _interactive():
        size_mb = sum(int(e["size"]) for e in needed) / (1 << 20)
        label = DEDUP_MODEL_NAMES.get(model_type, model_type.upper())
        if _prompt_confirm(
            f"MultiPassDedup weights for {label} are not installed. "
            f"Download them (~{size_mb:.0f} MB) into models/multipassdedup/?"
        ):
            try:
                for entry in needed:
                    modelhub.install_entry(entry)
                return
            except modelhub.HubError as exc:
                print()
                print(str(exc))
                raise SystemExit(1) from exc

    print()
    print(missing)
    raise SystemExit(1)


def build_dedup_plan() -> dict:
    """Prompt for MultiPassDedup parameters and construct an execution plan."""
    if _interactive():
        cadence_key = _prompt_choice("Duplicate cadence:", _cadence_options())
        model_key = _prompt_choice("Interpolation model:", _model_options())
        factor_key = _prompt_choice("Interpolation multiplier (frame rate):", _factor_options())
    else:
        cadence_key = config.DEDUP_NPASS_DEFAULT
        model_key = config.DEDUP_MODEL_DEFAULT
        factor_key = "2"

    model_type = validate_model_type(model_key)
    npass = parse_npass(cadence_key)
    factor = int(factor_key)

    ensure_dedup_weights(model_type)

    device = detect_dedup_device()
    device_label = "CUDA (torch)" if device == "cuda" else "CPU (torch)"
    model_label = DEDUP_MODEL_NAMES.get(model_type, model_type.upper())
    cadence_label = f"Auto (np=0)" if npass == 0 else f"On {'Twos' if npass == 2 else 'Threes'} (np={npass})"

    return {
        "action_label": "Interpolate (MultiPassDedup)",
        "engine_label": f"MultiPassDedup ({model_type.upper()})",
        "header_lines": [
            f"Model: {model_label}",
            f"Cadence: {cadence_label}",
            f"Factor: {factor}x",
            f"Device: {device_label}",
            f"Scale: {config.DEDUP_SCALE_DEFAULT}",
            f"Scene Detect: {'on' if config.DEDUP_SCDET_DEFAULT else 'off'}",
        ],
        "summary_lines": [
            f"\nAction:\n  Interpolate (MultiPassDedup)",
            f"\nModel:\n  {model_label}",
            f"\nCadence:\n  {cadence_label}",
            f"\nFactor:\n  {factor}x",
            f"\nDevice:\n  {device_label}",
        ],
        "model": model_type,
        "npass": npass,
        "factor": factor,
    }


def _worker_env() -> dict:
    """Environment for the worker: vendored models + video_upscaler importable."""
    env = os.environ.copy()
    pythonpath_parts = [
        str(config.BASE_DIR / "src" / "video_upscaler" / "multipass_dedup"),
        str(config.BASE_DIR / "src"),
    ]
    if "PYTHONPATH" in env:
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["PYTHONUNBUFFERED"] = "1"
    return env


class _WorkerRequestError(RuntimeError):
    """The worker refused one request but is still healthy and reusable."""


class _DedupWorker:
    """One long-lived ``worker.py`` process shared by every dedup render.

    Keeping the model resident removes the per-video load (torch/cupy import,
    checkpoint loads, CUDA context, softsplat kernel compile, cuDNN autotune)
    that previously ran again for every file and every render.
    """

    def __init__(self) -> None:
        self._io_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._log_handle = None
        self._loaded: tuple[str, float] | None = None
        self._log_path = config.CACHE_DIR / "dedup_worker.log"

    # -- state ---------------------------------------------------------

    def is_loaded(self, model_type: str, scale: float) -> bool:
        with self._state_lock:
            return self._loaded == (model_type, float(scale))

    def _mark_loaded(self, model_type: str, scale: float) -> None:
        with self._state_lock:
            self._loaded = (model_type, float(scale))

    # -- process lifecycle ---------------------------------------------

    def _start(self) -> subprocess.Popen:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        self._close_process()

        script = (
            config.BASE_DIR / "src" / "video_upscaler" / "multipass_dedup" / "worker.py"
        )
        if not script.is_file():
            raise FileNotFoundError(f"MultiPassDedup worker not found at {script}")

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(self._log_path, "ab", buffering=0)
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(script),
                    "--weights",
                    str(config.DEDUP_MODELS_DIR.resolve()),
                ],
                cwd=str(script.parent),
                env=_worker_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_handle,
                text=True,
                bufsize=1,
            )
        except Exception:
            self._close_process()
            raise
        with self._state_lock:
            self._loaded = None
        return self._process

    def _close_process(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
            except OSError:
                pass
            for stream in (process.stdin, process.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
        handle, self._log_handle = self._log_handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def _diagnostics(self, message: str) -> str:
        try:
            tail = self._log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except OSError:
            tail = ""
        return f"{message}\n{tail}".strip()

    # -- requests ------------------------------------------------------

    def _exchange(self, payload: dict, on_progress=None) -> dict:
        """Send one command and read protocol events until it terminates.

        Caller must hold ``_io_lock``. A transport failure or a cancel tears
        the worker down so a broken or busy process is never reused; a
        per-request ``error`` event leaves the (healthy) worker alive.
        """
        try:
            process = self._start()
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
        except Exception:
            self._close_process()
            raise

        while True:
            try:
                line = process.stdout.readline()
            except Exception:
                self._close_process()
                raise
            if not line:
                self._close_process()
                raise RuntimeError(
                    self._diagnostics("MultiPassDedup worker exited unexpectedly")
                )
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("event")
            if kind == "progress":
                if on_progress is not None:
                    try:
                        on_progress(
                            int(event.get("done") or 0), int(event.get("total") or 0)
                        )
                    except Exception:
                        # Cancellation (or any callback failure) must stop the
                        # busy worker rather than leave it mid-render.
                        self._close_process()
                        raise
            elif kind == "ready":
                return event
            elif kind == "done":
                return event
            elif kind == "error":
                raise _WorkerRequestError(
                    str(event.get("message") or "worker error")
                )

    def run(
        self,
        *,
        video: str,
        output: str,
        model_type: str,
        npass: int,
        times: int,
        scale: float,
        scdet: bool,
        threshold: float,
        hwaccel: bool,
        on_progress=None,
    ) -> None:
        payload = {
            "cmd": "run",
            "video": video,
            "output": output,
            "model_type": model_type,
            "npass": npass,
            "times": times,
            "scale": scale,
            "scdet": scdet,
            "threshold": threshold,
            "hwaccel": hwaccel,
        }
        with self._io_lock:
            self._exchange(payload, on_progress=on_progress)
        self._mark_loaded(model_type, scale)

    def preload(self, model_type: str, scale: float) -> bool:
        """Load the model without running a video; never queues behind a run."""
        if self.is_loaded(model_type, scale):
            return True
        if not self._io_lock.acquire(blocking=False):
            return False
        try:
            if self.is_loaded(model_type, scale):
                return True
            self._exchange(
                {"cmd": "load", "model_type": model_type, "scale": scale}
            )
            self._mark_loaded(model_type, scale)
            return True
        except Exception:
            return False
        finally:
            self._io_lock.release()

    def stop(self) -> None:
        self._close_process()
        with self._state_lock:
            self._loaded = None


_worker: _DedupWorker | None = None
_worker_guard = threading.Lock()


def _get_worker() -> _DedupWorker:
    global _worker
    with _worker_guard:
        if _worker is None:
            _worker = _DedupWorker()
        return _worker


def preload_dedup_model(model_type: str, scale: float | None = None) -> bool:
    """Warm the persistent worker so the next render skips the model load.

    Called in the background when the Interpolate tab opens. Returns False
    when the weights are not installed yet or another run owns the worker.
    """
    model_type = validate_model_type(model_type)
    if check_dedup_weights(model_type):
        return False
    worker = _get_worker()
    return worker.preload(
        model_type,
        float(config.DEDUP_SCALE_DEFAULT if scale is None else scale),
    )


def stop_dedup_worker() -> None:
    """Kill the resident worker (Reset App, shutdown, tests)."""
    global _worker
    with _worker_guard:
        worker, _worker = _worker, None
    if worker is not None:
        worker.stop()


atexit.register(stop_dedup_worker)


def run_dedup_infer(
    video_in: Path,
    video_out: Path,
    model_type: str,
    npass: int,
    factor: int,
    scale: float = 1.0,
    enable_scdet: bool = True,
    scdet_threshold: float = 0.3,
    hwaccel: bool = False,
    progress_cb=None,
    stage_cb=None,
) -> None:
    """Execute MultiPassDedup inference on a video through the persistent worker.

    ``progress_cb`` receives a fraction in [0, 1] for the current video.
    """
    model_type = validate_model_type(model_type)
    ensure_dedup_weights(model_type, auto_download=True)
    video_in = Path(video_in).resolve()
    video_out = Path(video_out)
    if not video_in.is_file():
        raise FileNotFoundError(f"can't find the file {video_in}")

    worker = _get_worker()
    if stage_cb is not None and not worker.is_loaded(model_type, scale):
        stage_cb(f"Loading the MultiPassDedup model ({model_type.upper()})…")

    interpolating = False

    def on_frame_progress(done: int, total: int) -> None:
        nonlocal interpolating
        if not interpolating:
            interpolating = True
            if stage_cb is not None:
                stage_cb(f"Interpolating {video_in.name}…")
        if progress_cb is not None:
            fraction = (done / total) if total else 0.0
            progress_cb(min(max(fraction, 0.0), 1.0))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        temp_out = Path(tmpdir) / video_in.name
        worker.run(
            video=str(video_in),
            output=str(temp_out.resolve()),
            model_type=model_type,
            npass=int(npass),
            times=int(factor),
            scale=float(scale),
            scdet=bool(enable_scdet),
            threshold=float(scdet_threshold),
            hwaccel=bool(hwaccel),
            on_progress=on_frame_progress,
        )
        if not temp_out.exists():
            raise RuntimeError("MultiPassDedup did not generate an output video.")
        video_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_out), str(video_out))
