"""Tests for the desktop first-run wizard: state, progress, and the HTTP server.

Every provisioning subprocess is monkeypatched — these tests exercise routing,
state transitions and progress math over real loopback sockets without
downloading a single wheel.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from video_upscaler.desktop import server as wizard
from video_upscaler.desktop import state as setup_state
from video_upscaler.desktop.bootstrap import already_provisioned, find_free_port, parse_args
from video_upscaler.desktop.context import (
    SetupContext,
    canonical_executable,
    find_python_executable,
)
from video_upscaler.desktop.gpu import GpuInfo, parse_nvidia_smi_output
from video_upscaler.desktop.models import TIERS, install_models
from video_upscaler.desktop.progress import ProgressTracker, parse_uv_progress_line
from video_upscaler.desktop.runtime import SetupError


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def context(tmp_path: Path) -> SetupContext:
    static = tmp_path / "static"
    static.mkdir()
    (static / "setup.html").write_text("<html>CLARITY wizard</html>", encoding="utf-8")
    (static / "js").mkdir()
    (static / "js" / "setup.js").write_text("// wizard engine", encoding="utf-8")
    (static / "secret.txt").write_text("do not read", encoding="utf-8")

    app = tmp_path / "app"
    (app / "src").mkdir(parents=True)
    (app / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    return SetupContext(
        data_dir=tmp_path / "data",
        resources_dir=app,
        static_dir=static,
        python_executable=Path(sys.executable),
    )


@pytest.fixture()
def base(context: SetupContext) -> SetupContext:
    context.data_dir.mkdir(parents=True, exist_ok=True)
    return context


@pytest.fixture()
def server(base: SetupContext):
    httpd = wizard.make_server("127.0.0.1", 0, base)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def managed_home(base: SetupContext) -> Path:
    """The directory uv records as ``home`` for a managed runtime.

    It is a child of the install dir: uv keeps the versionless name there as a
    symlink alias, and ``pyvenv.cfg`` resolves through to the versioned folder.
    """
    return base.python_install_dir / "cpython-3.11.15-windows-x86_64-none"


def write_fake_env(env_dir: Path, home: Path) -> None:
    """Materialise the tree ``uv venv`` would leave behind."""
    scripts = env_dir / ("Scripts" if os.name == "nt" else "bin")
    binary = "python.exe" if os.name == "nt" else "python"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / binary).write_bytes(b"")
    (env_dir / "pyvenv.cfg").write_text(
        f"home = {home}\nversion_info = 3.11\n", encoding="utf-8"
    )


def fake_uv(base: SetupContext, calls: list[list[str]]):
    """Stand in for uv: record argv and create a venv with a managed home."""

    def run(command, tracker, **kwargs):
        argv = [str(part) for part in command]
        calls.append(argv)
        if "venv" in argv:
            write_fake_env(Path(argv[-1]), managed_home(base))
        return "ok"

    return run


def get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read().decode()


def post(url: str, payload: dict | None = None) -> tuple[int, dict]:
    body = json.dumps(payload or {}).encode()
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, {"raw": raw}


def get_json(url: str) -> dict:
    return json.loads(get(url)[2])


def settle(server: str, timeout: float = 10.0) -> dict:
    """Poll until the running step finishes, then return the last snapshot."""
    snapshot: dict = {}
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        snapshot = get_json(f"{server}/api/setup/progress")
        if not snapshot.get("running"):
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"setup step did not finish in {timeout}s: {snapshot}")


def wait_running(server: str, timeout: float = 10.0) -> dict:
    """Poll until a step is reported as running."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        snapshot = get_json(f"{server}/api/setup/progress")
        if snapshot.get("running"):
            return snapshot
        time.sleep(0.02)
    raise AssertionError("setup step never started")


# --------------------------------------------------------------------------- #
# uv progress parsing (ported shell behaviour)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "line, expected",
    [
        ("Downloading torch-2.5.1+cu126.whl (832.1MB)", None),
        ("Downloading torch (1.2GB) 45.3% [==================] 24.5MB/s", (45.3, "24.5MB/s")),
        ("Downloading numpy (12.3MiB) 12% (18.2 MB/s)", (12.0, "18.2 MB/s")),
        ("Resolved 41 packages 100%", (100.0, "")),
        ("Built clarity @ file:///app 0%", (0.0, "")),
        ("Prepared 3 packages in 1.20s", None),
        ("", None),
        ("   ", None),
        ("Progress 150% of nothing", None),
        ("Progress -4% nope", None),
        ("A 7% B 8%", (7.0, "")),
    ],
)
def test_parse_uv_progress_line(line: str, expected: tuple[float, str] | None) -> None:
    assert parse_uv_progress_line(line) == expected


def test_tracker_is_monotonic_within_a_step() -> None:
    tracker = ProgressTracker()
    tracker.start("runtime")
    tracker.line("Downloading torch (1.2GB) 40% [===] 24.5MB/s")
    first = tracker.snapshot()["percent"]
    tracker.line("Downloading torch (1.2GB) 10% [===] 24.5MB/s")
    assert tracker.snapshot()["percent"] == first
    tracker.line("Downloading torch (1.2GB) 90% [===] 24.5MB/s")
    assert tracker.snapshot()["percent"] > first
    assert tracker.snapshot()["speed"] == "24.5MB/s"


def test_tracker_finish_completes_the_step_window() -> None:
    tracker = ProgressTracker()
    tracker.start("runtime")
    tracker.finish()
    snapshot = tracker.snapshot()
    assert snapshot["percent"] == 70.0
    assert snapshot["running"] is False
    assert snapshot["error"] is None


def test_tracker_records_errors() -> None:
    tracker = ProgressTracker()
    tracker.start("verify")
    tracker.finish("boom")
    snapshot = tracker.snapshot()
    assert snapshot["error"] == "boom"
    assert snapshot["running"] is False


def test_tracker_rewind_allows_progress_to_move_back() -> None:
    tracker = ProgressTracker()
    tracker.start("runtime")
    tracker.finish()
    assert tracker.snapshot()["percent"] == 70.0
    tracker.rewind("runtime")
    assert tracker.snapshot()["percent"] == 5.0
    tracker.line("Retrying torch 5% [>] 1.0MB/s")
    assert tracker.snapshot()["percent"] > 5.0


# --------------------------------------------------------------------------- #
# gpu parsing
# --------------------------------------------------------------------------- #
def test_parse_nvidia_smi_output_strips_uuid_and_keeps_all_gpus() -> None:
    output = (
        "GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-abc-123)\n"
        "GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-def-456)\n"
    )
    assert parse_nvidia_smi_output(output) == [
        "NVIDIA GeForce RTX 3080",
        "NVIDIA GeForce RTX 4090",
    ]


def test_parse_nvidia_smi_output_rejects_noise() -> None:
    assert parse_nvidia_smi_output("CUDA initialization error") == []
    assert parse_nvidia_smi_output("") == []


def test_gpu_variant_selection() -> None:
    assert GpuInfo(has_nvidia=True, names=["RTX"]).torch_variant == "cu126"
    assert GpuInfo().torch_variant == "cpu"


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
def test_state_roundtrip_is_atomic(tmp_path: Path) -> None:
    state = setup_state.SetupState()
    state.torch_variant = "cu126"
    state.gpu_names = ["RTX 3080"]
    state.mark("gpu", setup_state.DONE)
    setup_state.save_state(tmp_path, state)

    reloaded = setup_state.load_state(tmp_path)
    assert reloaded.torch_variant == "cu126"
    assert reloaded.gpu_names == ["RTX 3080"]
    assert reloaded.is_done("gpu")
    assert not reloaded.is_done("runtime")
    assert list(tmp_path.glob(".setup-*.tmp")) == []


def test_load_state_tolerates_corruption(tmp_path: Path) -> None:
    (tmp_path / setup_state.SETUP_FILE_NAME).write_text("{not json", encoding="utf-8")
    assert setup_state.load_state(tmp_path).steps == {s: "pending" for s in setup_state.STEPS}


def test_reset_from_cascades(tmp_path: Path) -> None:
    state = setup_state.SetupState()
    for step in setup_state.STEPS:
        state.mark(step, setup_state.DONE)
    state.completed_at = "2026-01-01T00:00:00+00:00"

    reset = state.reset_from("models")
    assert reset == ["models", "verify", "complete"]
    assert state.is_done("runtime")
    assert not state.is_done("models")
    assert state.completed_at == ""


def test_is_setup_complete_needs_marker_and_interpreter(tmp_path: Path) -> None:
    assert not setup_state.is_setup_complete(tmp_path)
    setup_state.write_marker(tmp_path)
    assert not setup_state.is_setup_complete(tmp_path)  # venv python still missing
    python = setup_state.venv_python(tmp_path)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")
    assert setup_state.is_setup_complete(tmp_path)


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
def test_child_env_points_the_app_at_the_data_dir(base: SetupContext) -> None:
    env = base.child_env()
    assert env["CLARITY_DATA_DIR"] == str(base.data_dir)
    assert env["CLARITY_DESKTOP_MODE"] == "1"
    assert str(base.resources_dir) in env["PATH"]
    assert str(base.package_src) in env["PYTHONPATH"]


def test_ensure_directories_creates_media_layout(base: SetupContext) -> None:
    base.ensure_directories()
    for name in ("input", "output", "models", "tools", ".cache", "logs"):
        assert (base.data_dir / name).is_dir()


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_install_models_rejects_unknown_tier(base: SetupContext) -> None:
    with pytest.raises(ValueError):
        install_models(base, ProgressTracker(), "ludicrous")


def test_install_models_requires_a_provisioned_interpreter(base: SetupContext) -> None:
    with pytest.raises(FileNotFoundError):
        install_models(base, ProgressTracker(), "essential")


def test_model_tiers_match_the_app_cli() -> None:
    assert TIERS == ("essential", "all")


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
def test_status_reports_incomplete_environment(server: str) -> None:
    status = get_json(f"{server}/api/setup/status")
    assert status["complete"] is False
    assert status["next_step"] == "gpu"
    assert status["state"]["steps"]["runtime"] == "pending"


def test_wizard_page_and_assets_are_served(server: str) -> None:
    status, content_type, body = get(f"{server}/")
    assert status == 200
    assert "text/html" in content_type
    assert "CLARITY wizard" in body

    status, content_type, body = get(f"{server}/static/js/setup.js")
    assert status == 200
    assert "javascript" in content_type
    assert "wizard engine" in body


def test_static_traversal_is_refused(server: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(f"{server}/static/../secret.txt")
    assert excinfo.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(f"{server}/static/%2e%2e%2fsecret.txt")
    assert excinfo.value.code == 404


def test_unknown_routes_are_404(server: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(f"{server}/api/nonsense")
    assert excinfo.value.code == 404


def test_detect_gpu_step_updates_state(server: str, base: SetupContext, monkeypatch) -> None:
    monkeypatch.setattr(
        wizard, "detect_gpu", lambda: GpuInfo(has_nvidia=True, names=["RTX 3080"])
    )
    code, payload = post(f"{server}/api/setup/detect-gpu")
    assert code == 200
    assert payload["started"] == "gpu"
    snapshot = settle(server)
    assert snapshot["running"] is False
    assert snapshot["error"] is None

    status = get_json(f"{server}/api/setup/status")
    assert status["state"]["torch_variant"] == "cu126"
    assert status["state"]["gpu_names"] == ["RTX 3080"]
    assert status["state"]["steps"]["gpu"] == "done"
    assert status["next_step"] == "runtime"


def test_runtime_step_runs_venv_then_dependencies(
    server: str, base: SetupContext, monkeypatch
) -> None:
    calls: list[list[str]] = []

    def fake_streamed(command, tracker, **kwargs):
        fake_uv(base, calls)(command, tracker, **kwargs)
        tracker.line("Resolved 41 packages 100%")

    monkeypatch.setattr(wizard.runtime_steps, "run_streamed", fake_streamed)
    monkeypatch.setattr(
        wizard, "detect_gpu", lambda: GpuInfo(has_nvidia=True, names=["RTX 3080"])
    )

    post(f"{server}/api/setup/detect-gpu")
    settle(server)
    post(f"{server}/api/setup/runtime")
    settle(server)

    assert "venv" in calls[0][1]
    assert "only-managed" in calls[0], "uv must never fall back to a host interpreter"
    assert "pip" in calls[1][1]
    assert "--extra-index-url" in calls[1]  # detected NVIDIA -> CUDA wheel index
    assert "cu126" in " ".join(calls[1])

    status = get_json(f"{server}/api/setup/status")
    assert status["state"]["steps"]["runtime"] == "done"
    assert status["state"]["torch_variant"] == "cu126"
    assert status["next_step"] == "models"


def test_cpu_variant_skips_the_cuda_index(
    server: str, base: SetupContext, monkeypatch
) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(wizard.runtime_steps, "run_streamed", fake_uv(base, calls))
    monkeypatch.setattr(wizard, "detect_gpu", lambda: GpuInfo())

    post(f"{server}/api/setup/detect-gpu")
    settle(server)
    post(f"{server}/api/setup/runtime")
    settle(server)

    assert get_json(f"{server}/api/setup/status")["state"]["torch_variant"] == "cpu"
    assert "--extra-index-url" not in calls[1]


def test_venv_step_reuses_an_environment_that_is_already_ours(
    server: str, base: SetupContext, monkeypatch
) -> None:
    """Retrying must not relink an interpreter that is already in place.

    ``uv venv --allow-existing`` rewrites Scripts/python.exe, which fails with
    ERROR_ACCESS_DENIED when that file is running or being scanned.
    """
    calls: list[list[str]] = []
    write_fake_env(base.env_dir, managed_home(base))
    monkeypatch.setattr(wizard.runtime_steps, "run_streamed", fake_uv(base, calls))
    monkeypatch.setattr(wizard, "detect_gpu", lambda: GpuInfo())

    post(f"{server}/api/setup/detect-gpu")
    settle(server)
    post(f"{server}/api/setup/runtime")
    settle(server)

    assert [call for call in calls if "venv" in call] == [], "uv venv must not run again"
    assert get_json(f"{server}/api/setup/status")["state"]["steps"]["runtime"] == "done"


def test_an_environment_built_from_a_foreign_interpreter_is_recreated(
    server: str, base: SetupContext, monkeypatch
) -> None:
    """A venv we did not provision is not silently accepted.

    The bug this covers: a failed venv step left an environment whose
    ``pyvenv.cfg`` pointed at uv's roaming ``%APPDATA%\\uv\\python``, so the
    "provisioned" runtime depended on files outside the data directory.
    """
    calls: list[list[str]] = []
    foreign = base.data_dir.parent / "roaming" / "uv" / "python"
    write_fake_env(base.env_dir, foreign)
    sentinel = base.env_dir / ("Scripts" if os.name == "nt" else "bin") / "stale.txt"
    sentinel.write_text("installed by the wrong interpreter", encoding="utf-8")

    monkeypatch.setattr(wizard.runtime_steps, "run_streamed", fake_uv(base, calls))
    monkeypatch.setattr(wizard, "detect_gpu", lambda: GpuInfo())

    post(f"{server}/api/setup/detect-gpu")
    settle(server)
    post(f"{server}/api/setup/runtime")
    settle(server)

    assert [call for call in calls if "venv" in call], "the foreign env must be rebuilt"
    assert not sentinel.exists(), "the stale environment must be removed"
    cfg = (base.env_dir / "pyvenv.cfg").read_text(encoding="utf-8")
    assert str(base.python_install_dir) in cfg


def test_failed_step_is_reported_and_retryable(
    server: str, base: SetupContext, monkeypatch
) -> None:
    def explode(command, tracker, **kwargs):
        raise SetupError("network unreachable")

    monkeypatch.setattr(wizard.runtime_steps, "run_streamed", explode)
    post(f"{server}/api/setup/runtime")
    snapshot = settle(server)
    assert "network unreachable" in snapshot["error"]

    status = get_json(f"{server}/api/setup/status")
    assert status["state"]["steps"]["runtime"] == "failed"
    assert status["state"]["errors"]["runtime"] == "network unreachable"

    code, payload = post(f"{server}/api/setup/retry", {"step": "runtime"})
    assert code == 200
    assert "runtime" in payload["reset"]
    assert get_json(f"{server}/api/setup/status")["state"]["steps"]["runtime"] == "pending"


def test_second_operation_is_rejected(
    server: str, base: SetupContext, monkeypatch
) -> None:
    gate = threading.Event()

    def blocking(command, tracker, **kwargs):
        gate.wait(5)
        return "ok"

    monkeypatch.setattr(wizard.runtime_steps, "run_streamed", blocking)
    post(f"{server}/api/setup/runtime")
    wait_running(server)
    code, payload = post(f"{server}/api/setup/verify")
    assert code == 409
    assert "already running" in payload["error"]
    gate.set()
    settle(server)


def test_complete_requires_verification(server: str, base: SetupContext) -> None:
    post(f"{server}/api/setup/complete")
    settle(server)
    status = get_json(f"{server}/api/setup/status")
    assert status["state"]["steps"]["complete"] == "failed"
    assert not setup_state.is_setup_complete(base.data_dir)


def test_verified_setup_completes_and_writes_the_marker(
    server: str, base: SetupContext, monkeypatch
) -> None:
    monkeypatch.setattr(
        wizard.runtime_steps,
        "verify_runtime",
        lambda ctx, tracker: {"ok": True, "output": "imports ok 2.5.1 cuda False", "cuda": False},
    )
    python = base.venv_python
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")

    post(f"{server}/api/setup/verify")
    settle(server)
    post(f"{server}/api/setup/complete")
    settle(server)

    status = get_json(f"{server}/api/setup/status")
    assert status["complete"] is True
    assert status["state"]["steps"]["complete"] == "done"
    assert status["state"]["completed_at"]
    assert setup_state.is_setup_complete(base.data_dir)


def test_invalid_model_tier_is_a_bad_request(server: str) -> None:
    code, payload = post(f"{server}/api/setup/models", {"tier": "everything"})
    assert code == 400
    assert "tier" in payload["error"].lower()


def test_model_step_passes_the_requested_tier(
    server: str, base: SetupContext, monkeypatch
) -> None:
    seen: dict = {}

    def fake_install(ctx, tracker, tier="essential"):
        seen["tier"] = tier

    monkeypatch.setattr(wizard.model_steps, "install_models", fake_install)
    post(f"{server}/api/setup/models", {"tier": "all"})
    settle(server)
    assert seen["tier"] == "all"
    assert get_json(f"{server}/api/setup/status")["state"]["models_tier"] == "all"


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_short_circuits_a_provisioned_environment(base: SetupContext) -> None:
    assert not already_provisioned(base)
    python = base.venv_python
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")
    setup_state.write_marker(base.data_dir)
    assert already_provisioned(base)


def test_find_free_port_returns_a_bindable_port() -> None:
    import socket

    port = find_free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))


def test_bootstrap_arguments_defaults() -> None:
    args = parse_args([])
    assert args.port == 0
    assert args.host == "127.0.0.1"
    assert args.exit_after_complete is True
