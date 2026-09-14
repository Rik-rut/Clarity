"""Stdlib HTTP server for the first-run desktop wizard.

The wizard is a page of the product, not a separate tool: it speaks JSON over
loopback HTTP and polls for progress, so it renders identically in the Tauri
webview and in a plain browser. Only the standard library may be used here —
this process starts before torch, opencv or fastapi exist.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from video_upscaler.desktop import models as model_steps
from video_upscaler.desktop import runtime as runtime_steps
from video_upscaler.desktop import state as setup_state
from video_upscaler.desktop.context import SetupContext
from video_upscaler.desktop.gpu import GpuInfo, detect_gpu
from video_upscaler.desktop.progress import ProgressTracker

# Provisioning steps the wizard can run, in order.
STEP_OPERATIONS: Tuple[str, ...] = ("gpu", "runtime", "models", "verify", "complete")


class WizardService:
    """Setup steps, their state, and a single-operation-at-a-time guard.

    Two locks, deliberately: ``_op_lock`` is held by the worker thread for the
    whole duration of a step (so a second POST is rejected with 409), while
    ``_state_lock`` only ever covers a read or write of ``state``. Using one
    lock for both would deadlock the worker against its own guard.
    """

    def __init__(self, context: SetupContext) -> None:
        self.context = context
        self.tracker = ProgressTracker()
        self._state_lock = threading.Lock()
        self._op_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self.state = setup_state.load_state(context.data_dir)

    # -- reads -------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self._state_lock:
            payload = self.state.to_dict()
        return {
            "complete": setup_state.is_setup_complete(self.context.data_dir),
            "state": payload,
            "next_step": self.next_step(),
            "running": self.busy(),
            "environment": {
                "data_dir": str(self.context.data_dir),
                "venv_python": str(self.context.venv_python),
                "resources_dir": str(self.context.resources_dir),
                "uv_exe": str(self.context.uv_exe),
            },
        }

    def progress(self) -> Dict[str, Any]:
        return self.tracker.snapshot()

    def next_step(self) -> str:
        with self._state_lock:
            for step in STEP_OPERATIONS:
                if self.state.status(step) != setup_state.DONE:
                    return step
        return "complete"

    def busy(self) -> bool:
        worker = self._worker
        return bool(worker and worker.is_alive())

    def current_torch_variant(self) -> str:
        with self._state_lock:
            return self.state.torch_variant

    # -- writes ------------------------------------------------------------
    def begin(self, step: str) -> None:
        """Start ``step`` on a worker thread. Raises ValueError/RuntimeError."""
        if step not in STEP_OPERATIONS:
            raise ValueError(f"Unknown setup step: {step}")
        if not self._op_lock.acquire(blocking=False):
            raise RuntimeError("Another setup operation is already running")
        self.tracker.start(step)
        with self._state_lock:
            self.state.mark(step, setup_state.RUNNING)
            setup_state.save_state(self.context.data_dir, self.state)
        self._worker = threading.Thread(
            target=self._execute, args=(step,), name=f"clarity-setup-{step}", daemon=True
        )
        self._worker.start()

    def _execute(self, step: str) -> None:
        try:
            getattr(self, f"_step_{step}")()
        except Exception as exc:  # reported to the wizard, never re-raised
            message = str(exc).strip() or exc.__class__.__name__
            self.tracker.finish(message)
            with self._state_lock:
                self.state.mark(step, setup_state.FAILED, message)
                setup_state.save_state(self.context.data_dir, self.state)
        finally:
            self._op_lock.release()

    def retry(self, step: str) -> List[str]:
        """Clear ``step`` and everything after it so the chain can re-run."""
        with self._state_lock:
            reset = self.state.reset_from(step)
            setup_state.save_state(self.context.data_dir, self.state)
        self.tracker.rewind(step)
        marker = self.context.marker_file
        if marker.is_file():
            try:
                marker.unlink()
            except OSError:
                pass
        return reset

    def set_tier(self, tier: str) -> None:
        """Record the requested model tier for the models step."""
        if tier not in model_steps.TIERS:
            raise ValueError(
                f"Invalid model tier {tier!r}. Choose from: {', '.join(model_steps.TIERS)}."
            )
        with self._state_lock:
            self.state.models_tier = tier
            setup_state.save_state(self.context.data_dir, self.state)

    def _mark_done(self, step: str) -> None:
        with self._state_lock:
            self.state.mark(step, setup_state.DONE)
            setup_state.save_state(self.context.data_dir, self.state)

    # -- steps -------------------------------------------------------------
    def _step_gpu(self) -> None:
        info: GpuInfo = detect_gpu()
        self.tracker.line(
            "NVIDIA GPU detected: " + ", ".join(info.names)
            if info.has_nvidia
            else "No NVIDIA GPU found — installing the CPU build."
        )
        with self._state_lock:
            self.state.gpu_names = info.names
            self.state.torch_variant = info.torch_variant
            self.state.mark("gpu", setup_state.DONE)
            setup_state.save_state(self.context.data_dir, self.state)
        self.tracker.finish()

    def _step_runtime(self) -> None:
        variant = self.current_torch_variant()
        if not variant:
            info = detect_gpu()
            variant = info.torch_variant
            with self._state_lock:
                self.state.torch_variant = variant
                self.state.gpu_names = info.names
                setup_state.save_state(self.context.data_dir, self.state)
        self.context.ensure_directories()
        runtime_steps.create_venv(self.context, self.tracker)
        runtime_steps.install_dependencies(self.context, self.tracker, variant)
        self._mark_done("runtime")
        self.tracker.finish()

    def _step_models(self) -> None:
        with self._state_lock:
            tier = self.state.models_tier or model_steps.DEFAULT_TIER
        model_steps.install_models(self.context, self.tracker, tier)
        with self._state_lock:
            self.state.models_tier = tier
            self.state.mark("models", setup_state.DONE)
            setup_state.save_state(self.context.data_dir, self.state)
        self.tracker.finish()

    def _step_verify(self) -> None:
        result = runtime_steps.verify_runtime(self.context, self.tracker)
        self._mark_done("verify")
        self.tracker.line(str(result.get("output", "")))
        self.tracker.finish()

    def _step_complete(self) -> None:
        with self._state_lock:
            if self.state.status("verify") != setup_state.DONE:
                raise RuntimeError("Verify the environment before finishing setup.")
            self.state.mark("complete", setup_state.DONE)
            self.state.completed_at = setup_state.timestamp()
            setup_state.save_state(self.context.data_dir, self.state)
        setup_state.write_marker(self.context.data_dir)
        self.tracker.line("Environment ready. Handing off to Clarity Studio…")
        self.tracker.finish()


class WizardRequestHandler(BaseHTTPRequestHandler):
    """Routes for the wizard: static assets plus the JSON setup API."""

    server_version = "ClaritySetup/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def service(self) -> WizardService:
        return self.server.service  # type: ignore[attr-defined]

    # -- plumbing ----------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        if status == int(HTTPStatus.NO_CONTENT):
            self.end_headers()
            return
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, payload: Any, status: int = int(HTTPStatus.OK)) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"success": False, "error": message}, status)

    def _body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        try:
            parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # -- verbs -------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/setup", "/setup.html"):
            return self._serve_static("setup.html")
        if path.startswith("/static/"):
            return self._serve_static(unquote(path[len("/static/") :]))
        if path == "/favicon.ico":
            return self._send(int(HTTPStatus.NO_CONTENT), b"", "image/x-icon")
        if path == "/api/setup/status":
            return self._json(self.service.status())
        if path == "/api/setup/progress":
            return self._json(self.service.progress())
        self._error(int(HTTPStatus.NOT_FOUND), f"Not found: {path}")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._body()

        if path == "/api/setup/retry":
            return self._retry(body)
        if path == "/api/setup/start":
            return self._start(self.service.next_step(), {})

        routes: Dict[str, Tuple[str, Callable[[Dict[str, Any]], None]]] = {
            "/api/setup/detect-gpu": ("gpu", _noop),
            "/api/setup/runtime": ("runtime", _noop),
            "/api/setup/models": ("models", self._remember_tier),
            "/api/setup/verify": ("verify", _noop),
            "/api/setup/complete": ("complete", _noop),
        }
        if path not in routes:
            return self._error(int(HTTPStatus.NOT_FOUND), f"Not found: {path}")

        step, prepare = routes[path]
        return self._start(step, body, prepare)

    def _start(
        self,
        step: str,
        body: Dict[str, Any],
        prepare: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        try:
            if prepare:
                prepare(body)
            self.service.begin(step)
        except ValueError as exc:
            return self._error(int(HTTPStatus.BAD_REQUEST), str(exc))
        except RuntimeError as exc:
            return self._error(int(HTTPStatus.CONFLICT), str(exc))
        self._json({"success": True, "started": step, "status": self.service.status()})

    def _remember_tier(self, body: Dict[str, Any]) -> None:
        self.service.set_tier(str(body.get("tier") or model_steps.DEFAULT_TIER))

    def _retry(self, body: Dict[str, Any]) -> None:
        step = str(body.get("step") or self.service.next_step())
        if step not in STEP_OPERATIONS:
            return self._error(int(HTTPStatus.BAD_REQUEST), f"Unknown step: {step}")
        reset = self.service.retry(step)
        self._json({"success": True, "reset": reset, "status": self.service.status()})

    # -- static ------------------------------------------------------------
    def _serve_static(self, name: str) -> None:
        static_root = self.service.context.static_dir.resolve()
        candidate = (static_root / name).resolve()
        if not candidate.is_relative_to(static_root) or not candidate.is_file():
            return self._error(int(HTTPStatus.NOT_FOUND), f"Missing asset: {name}")
        content_type, _ = mimetypes.guess_type(str(candidate))
        if content_type is None:
            content_type = "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type = f"{content_type}; charset=utf-8"
        self._send(int(HTTPStatus.OK), candidate.read_bytes(), content_type)


def _noop(body: Dict[str, Any]) -> None:
    return None


def make_server(
    host: str,
    port: int,
    context: SetupContext,
    service: Optional[WizardService] = None,
) -> ThreadingHTTPServer:
    """Bind the wizard server. ``port`` 0 lets the OS choose (used by tests)."""
    httpd = ThreadingHTTPServer((host, port), WizardRequestHandler)
    httpd.daemon_threads = True
    httpd.service = service or WizardService(context)  # type: ignore[attr-defined]
    return httpd


def serve(
    context: SetupContext, host: str = "127.0.0.1", port: int = 0
) -> Tuple[ThreadingHTTPServer, int]:
    """Start the wizard on a daemon thread; returns the server and bound port."""
    httpd = make_server(host, port, context)
    thread = threading.Thread(target=httpd.serve_forever, name="clarity-wizard", daemon=True)
    thread.start()
    return httpd, int(httpd.server_address[1])
