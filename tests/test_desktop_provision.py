"""Contract tests for the headless provisioner.

Both callers — the NSIS installer and the Tauri boot shell — parse the same
`PROGRESS <percent>|<phase>|<message>` lines and rely on the same exit codes, so
those two things are what these tests pin down.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from video_upscaler.desktop import provision
from video_upscaler.desktop import runtime
from video_upscaler.desktop import state as setup_state
from video_upscaler.desktop.context import (
    SetupContext,
    build_context,
    canonical_executable,
    find_python_executable,
)
from video_upscaler.desktop.gpu import GpuInfo
from video_upscaler.desktop.progress import ProgressTracker
from video_upscaler.desktop.runtime import SetupError


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "Clarity"
    root.mkdir()
    monkeypatch.setenv("CLARITY_DATA_DIR", str(root))
    return root


def write_fake_env(env_dir: Path, home: Path) -> None:
    scripts = env_dir / ("Scripts" if os.name == "nt" else "bin")
    binary = "python.exe" if os.name == "nt" else "python"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / binary).write_bytes(b"")
    (env_dir / "pyvenv.cfg").write_text(f"home = {home}\nversion_info = 3.11\n", encoding="utf-8")


@pytest.fixture()
def base(data_dir: Path, tmp_path: Path) -> SetupContext:
    resources = tmp_path / "app"
    resources.mkdir()
    return build_context(data_dir=data_dir, resources_dir=resources)


def test_parse_args_defaults(data_dir: Path) -> None:
    args = provision.parse_args(["--data-dir", str(data_dir)])
    assert args.data_dir == data_dir
    assert args.tier == "essential"
    assert args.tensorrt == "auto"
    assert args.force is False


def test_progress_lines_are_machine_readable(capsys) -> None:
    tracker = provision.LinePrinter()
    tracker.start("runtime", "Installing")
    tracker.phase("dependencies")
    tracker.line("Downloading torch")

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("PROGRESS ")]
    assert lines, "every update must reach the caller"
    percent, phase, message = lines[-1].split("|")
    assert percent.startswith("PROGRESS ")
    assert percent.removeprefix("PROGRESS ").replace(".", "").strip().isdigit()
    assert phase == "dependencies"
    assert message == "Downloading torch"


def test_lock_blocks_a_second_provisioner(data_dir: Path) -> None:
    first = provision.acquire_lock(data_dir)
    assert first is None, "the first caller takes the lock"
    holder = provision.acquire_lock(data_dir)
    assert holder == os.getpid(), "a live holder is reported, not overwritten"
    provision.release_lock(data_dir)
    assert provision.acquire_lock(data_dir) is None


def test_a_stale_lock_is_taken_over(data_dir: Path) -> None:
    (data_dir / provision.LOCK_NAME).write_text("999999999", encoding="utf-8")
    assert provision.acquire_lock(data_dir) is None


def test_marker_is_written_only_on_success(data_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(provision, "detect_gpu", lambda: GpuInfo())
    calls: list[str] = []

    def boom(*_args, **_kwargs):
        calls.append("runtime")
        raise SetupError("venv failed (exit 2): Access is denied. (os error 5)")

    monkeypatch.setattr(provision.runtime, "create_venv", boom)
    args = provision.parse_args(["--data-dir", str(data_dir)])

    assert provision.main(["--data-dir", str(data_dir)]) == provision.EXIT_FAILED
    assert not setup_state.is_setup_complete(data_dir), "a failed run must not look complete"
    state = setup_state.load_state(data_dir)
    assert state.status("runtime") == "failed"
    assert "Access is denied" in state.errors["runtime"]


def _fake_steps(monkeypatch, backend: str = "torch-cuda") -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(provision, "detect_gpu", lambda: GpuInfo(has_nvidia=True, names=["RTX 3050"]))
    monkeypatch.setattr(
        provision.runtime, "install_runtime",
        lambda *a, **k: calls.append("runtime"),
    )
    monkeypatch.setattr(
        provision.model_steps, "install_models",
        lambda *a, **k: calls.append("models"),
    )
    monkeypatch.setattr(
        provision.runtime, "verify_runtime",
        lambda *a, **k: (calls.append("verify"), {"backend": backend})[1],
    )
    return calls


def test_a_successful_run_writes_the_marker_last(data_dir: Path, monkeypatch) -> None:
    _fake_steps(monkeypatch, backend="tensorrt")
    write_fake_env(data_dir / "env", data_dir / "python")

    assert provision.main(["--data-dir", str(data_dir)]) == provision.EXIT_OK

    assert setup_state.is_setup_complete(data_dir)
    state = setup_state.load_state(data_dir)
    assert state.status("verify") == "done"
    assert state.status("complete") == "done"
    assert state.torch_variant == "cu126"
    assert state.tensorrt is True, "auto means yes when NVIDIA is present"
    assert state.backend == "tensorrt"
    assert not (data_dir / provision.LOCK_NAME).exists(), "the lock must not survive"


def test_tensorrt_no_is_respected_on_an_nvidia_machine(data_dir: Path, monkeypatch) -> None:
    _fake_steps(monkeypatch)
    write_fake_env(data_dir / "env", data_dir / "python")

    provision.main(["--data-dir", str(data_dir), "--tensorrt", "no"])

    assert setup_state.load_state(data_dir).tensorrt is False


def test_finished_steps_are_not_repeated(data_dir: Path, monkeypatch) -> None:
    """A killed download must resume, not start over."""
    calls = _fake_steps(monkeypatch)
    write_fake_env(data_dir / "env", data_dir / "python")
    state = setup_state.load_state(data_dir)
    state.mark("gpu", setup_state.DONE)
    state.mark("runtime", setup_state.DONE)
    setup_state.save_state(data_dir, state)

    assert provision.main(["--data-dir", str(data_dir)]) == provision.EXIT_OK
    assert calls == ["models", "verify"]


def test_force_redoes_every_step(data_dir: Path, monkeypatch) -> None:
    calls = _fake_steps(monkeypatch)
    write_fake_env(data_dir / "env", data_dir / "python")
    state = setup_state.load_state(data_dir)
    for name in ("gpu", "runtime", "models", "verify"):
        state.mark(name, setup_state.DONE)
    setup_state.save_state(data_dir, state)

    provision.main(["--data-dir", str(data_dir), "--force"])

    assert calls == ["runtime", "models", "verify"], "gpu is re-detected but not in `calls`"


def test_an_environment_that_still_works_is_adopted_rather_than_deleted(
    base: SetupContext, monkeypatch
) -> None:
    """Unusual lineage is not breakage, and deleting a working runtime is worse
    than the inconsistency it fixes.

    This is the real repair path for a machine whose environment was built by an
    older version of provisioning: re-running ``uv venv`` there re-links
    Scripts/python.exe, which Windows refuses while anything holds it.
    """
    calls: list[list[str]] = []
    write_fake_env(base.env_dir, base.data_dir.parent / "roaming" / "uv" / "python")
    marker = base.env_dir / "MARKER.txt"
    marker.write_text("packages that took twenty minutes to download", encoding="utf-8")

    def fake_run(command, tracker, **kwargs):
        calls.append([str(part) for part in command])
        return "ok"

    monkeypatch.setattr(runtime, "run_streamed", fake_run)

    tracker = ProgressTracker()
    runtime.create_venv(base, tracker, probe=lambda env_dir: True)

    assert [call for call in calls if "venv" in call] == [], "a working env is not rebuilt"
    assert marker.exists(), "nothing in the environment may be thrown away"
    assert any("Reusing the environment from an earlier run" in line for line in tracker.snapshot()["lines"])


def test_an_unusable_env_that_cannot_be_deleted_names_the_running_process(
    base: SetupContext, monkeypatch
) -> None:
    """A refused delete must say what to close, not leak a raw WinError."""
    write_fake_env(base.env_dir, base.data_dir.parent / "roaming")

    def refuse(_path):
        raise PermissionError(13, "The process cannot access the file because it is being used")

    monkeypatch.setattr(runtime.shutil, "rmtree", refuse)

    with pytest.raises(SetupError) as raised:
        runtime.create_venv(base, ProgressTracker())

    message = str(raised.value)
    assert "still running" in message
    assert "Retry" in message


def test_the_probe_asks_the_interpreter_itself(tmp_path: Path, monkeypatch) -> None:
    seen: list[list[str]] = []

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        seen.append([str(part) for part in command])
        return Completed()

    python = tmp_path / "env" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    assert runtime._venv_imports_the_stack(tmp_path / "env") is True
    assert seen[0][0] == str(python)
    assert runtime.VERIFY_IMPORTS in " ".join(seen[0])


def test_the_probe_is_false_when_there_is_nothing_to_ask(tmp_path: Path) -> None:
    assert runtime._venv_imports_the_stack(tmp_path / "missing") is False


def test_dependency_install_refuses_an_env_without_an_interpreter(base: SetupContext) -> None:
    """uv would happily create an environment of its own; provisioning must not."""
    with pytest.raises(SetupError, match="no interpreter"):
        runtime.install_dependencies(base, ProgressTracker(), "cpu")


def test_windows_access_denied_is_retried_once_and_explains_itself(
    base: SetupContext, monkeypatch
) -> None:
    attempts: list[int] = []

    def deny(command, tracker, **kwargs):
        attempts.append(1)
        raise SetupError(
            "venv failed (exit 2): Failed to persist temporary file to "
            "...\\env\\Scripts\\python.exe: Access is denied. (os error 5)"
        )

    slept: list[float] = []
    monkeypatch.setattr(runtime, "run_streamed", deny)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: slept.append(seconds))

    with pytest.raises(SetupError) as raised:
        runtime.create_venv(base, ProgressTracker())

    assert len(attempts) == 2, "one retry, then stop and explain"
    assert slept, "the retry must wait rather than hammer the same file"
    message = str(raised.value)
    assert "Access is denied" in message
    assert "Close any other Clarity window" in message


def test_managed_interpreter_lookup_skips_uv_staging_dirs(tmp_path: Path) -> None:
    """``uv python install`` leaves ``.temp``/``.lock`` siblings behind."""
    install = tmp_path / "python"
    real = install / "cpython-3.11.15-windows-x86_64-none"
    staging = install / ".temp" / "download"
    for folder in (real, staging):
        folder.mkdir(parents=True)
        (folder / "python.exe").write_bytes(b"")

    assert find_python_executable(install) == real / "python.exe"


def _record_commands(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def streamed(command, tracker, **kwargs):
        calls.append([str(part) for part in command])
        return "ok"

    monkeypatch.setattr(runtime, "run_streamed", streamed)
    return calls


def test_the_tensorrt_extra_is_requested_when_chosen(base, monkeypatch) -> None:
    write_fake_env(base.env_dir, base.python_install_dir)
    calls = _record_commands(monkeypatch)

    runtime.install_dependencies(base, ProgressTracker(), "cu126", tensorrt=True)

    assert any(part.endswith("[tensorrt]") for part in calls[-1])
    assert "--extra-index-url" in calls[-1]


def test_tensorrt_is_not_requested_on_a_machine_without_nvidia(base, monkeypatch) -> None:
    write_fake_env(base.env_dir, base.python_install_dir)
    calls = _record_commands(monkeypatch)

    runtime.install_dependencies(base, ProgressTracker(), "cpu", tensorrt=False)

    assert not any(part.endswith("[tensorrt]") for part in calls[-1])


def test_a_failed_tensorrt_install_falls_back_to_cuda(base, monkeypatch) -> None:
    """TensorRT is an optimisation: a missing wheel must not brick the install."""
    write_fake_env(base.env_dir, base.python_install_dir)
    attempts: list[list[str]] = []

    def streamed(command, tracker, **kwargs):
        argv = [str(part) for part in command]
        attempts.append(argv)
        if any(part.endswith("[tensorrt]") for part in argv):
            raise SetupError("dependency install failed (exit 1): no solution for tensorrt")
        return "ok"

    monkeypatch.setattr(runtime, "run_streamed", streamed)
    runtime.install_dependencies(base, ProgressTracker(), "cu126", tensorrt=True)

    assert len(attempts) == 2
    assert not any(part.endswith("[tensorrt]") for part in attempts[-1])


def test_verify_runtime_reports_the_active_backend(base, monkeypatch) -> None:
    write_fake_env(base.env_dir, base.python_install_dir)
    monkeypatch.setattr(
        runtime, "run_streamed",
        lambda *a, **k: "imports ok 2.7.0 cuda True\nbackend tensorrt\n",
    )

    report = runtime.verify_runtime(base, ProgressTracker())

    assert report["backend"] == "tensorrt"
    assert report["cuda"] is True


def test_canonical_executable_resolves_the_versionless_alias(tmp_path: Path) -> None:
    """uv's alias directory is a symlink; handing it to uv breaks venv linking.

    Sorted scanning finds ``cpython-3.11-...`` before ``cpython-3.11.15-...``
    because ``-`` sorts before ``.``, so the alias is what used to reach uv.
    """
    install = tmp_path / "python"
    real = install / "cpython-3.11.15-windows-x86_64-none"
    real.mkdir(parents=True)
    target = real / "python.exe"
    target.write_bytes(b"")

    alias = install / "cpython-3.11-windows-x86_64-none"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks need developer mode or admin rights")

    assert find_python_executable(install) == alias / "python.exe"
    resolved = canonical_executable(alias / "python.exe")
    assert os.path.normcase(str(resolved)) == os.path.normcase(os.path.realpath(str(target)))
    assert "3.11.15" in resolved.parent.name
