"""Simulated install benchmark (tasks 4.2-4.5): old installer vs aria2 backend.

Both trees run ``main.py --download-models essential`` as a subprocess against
the same throttled local hub (``hub_sim``), in fresh data dirs. The cap is per
connection, so the baseline's duration is bounded from below and the comparison
is deterministic instead of depending on the machine's network.

``slow``-marked; the aria2 cases skip with a clear reason when no aria2c exists.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from hub_sim import build_simulated_hub

# Pre-change commit, recorded at planning time (design.md §8). HEAD is still this
# commit; the aria2 work is uncommitted in the working tree.
BASELINE_COMMIT = "3c1e2742a20d54eca7b45aea70b1d28ec342adfd"

CAP_BPS = 2 * 1024 * 1024
BIG_SIZE = 48 * 1024 * 1024
SMALL_SIZE = 512 * 1024
SMALL_COUNT = 12
# The cap bounds the baseline from below; 0.5x is loose but catches regressions.
MAX_NEW_RATIO = 0.5
FALLBACK_RATIO = 1.25
FALLBACK_SLACK_S = 3.0

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = (
    "main.py",
    "pyproject.toml",
    "src/video_upscaler/__init__.py",
    "src/video_upscaler/config.py",
    "src/video_upscaler/scanner.py",
    "src/video_upscaler/cli.py",
    "src/video_upscaler/modelhub.py",
    "src/video_upscaler/data/manifest.json",
)
MANIFEST_REL = "src/video_upscaler/data/manifest.json"

pytestmark = pytest.mark.slow


def _extract_commit(dest: Path, commit: str) -> Path:
    """Extract ``commit`` into ``dest/tree`` with ``git archive``."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "tree.tar"
    with open(archive, "wb") as handle:
        subprocess.run(
            ["git", "archive", "--format=tar", commit],
            cwd=str(REPO_ROOT),
            stdout=handle,
            check=True,
        )
    root = dest / "tree"
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(root, filter="tar")
        except TypeError:  # Python < 3.12 has no filter parameter
            tf.extractall(root)
    return root


def _copy_working_tree(dest: Path) -> Path:
    """Copy the files ``--download-models`` actually needs, uncommitted included."""
    root = dest / "tree"
    for rel in RUN_FILES:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, target)
    # The new backend module is untracked in the baseline commit.
    shutil.copyfile(REPO_ROOT / "src/video_upscaler/aria2.py", root / "src/video_upscaler/aria2.py")
    return root


def _write_manifest(tree: Path, manifest: dict) -> None:
    import json

    (tree / MANIFEST_REL).write_text(json.dumps(manifest), encoding="utf-8")


def _run_installer(tree: Path, data_dir: Path, hub_base: str, extra_env: dict[str, str] | None = None) -> tuple[float, str]:
    env = os.environ.copy()
    for name in (
        "CLARITY_DOWNLOADER",
        "CLARITY_ARIA2C",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "CLARITY_HF_TOKEN",
        "CLARITY_MODEL_HUB_BASE",
        "CLARITY_DATA_DIR",
        "CLARITY_MODELS_DIR",
    ):
        env.pop(name, None)
    env.update(
        {
            "CLARITY_DATA_DIR": str(data_dir),
            "CLARITY_MODELS_DIR": str(data_dir / "models"),
            "CLARITY_CACHE_DIR": str(data_dir / ".cache"),
            "CLARITY_MODEL_HUB_BASE": hub_base,
            "PYTHONPATH": str(tree / "src"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if extra_env:
        env.update(extra_env)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "main.py", "--download-models", "essential"],
        cwd=str(tree),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, f"installer failed:\n{proc.stdout}\n{proc.stderr}"
    return elapsed, proc.stdout


def _dest_for(entry: dict, data_dir: Path) -> Path:
    root = data_dir / "models"
    if entry["group"] in ("matanyone", "sam"):
        root = root / entry["group"]
    return root / entry["dest"]


def _verify_installed(data_dir: Path, manifest: dict) -> list[Path]:
    installed = []
    for entry in manifest["files"]:
        dest = _dest_for(entry, data_dir)
        assert dest.is_file(), f"{dest.name} was not installed"
        assert dest.stat().st_size == entry["size"], f"{dest.name} has the wrong size"
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"{dest.name} failed sha256"
        installed.append(dest)
    return installed


def _failing_aria2(tmp_path: Path) -> Path:
    """An executable that always fails, to exercise the fallback path."""
    if os.name == "nt":
        bat = tmp_path / "broken_aria2c.bat"
        bat.write_text("echo aria2 simulated failure\r\nexit /b 22\r\n", encoding="utf-8")
        return bat
    sh = tmp_path / "broken_aria2c.sh"
    sh.write_text("#!/bin/sh\necho aria2 simulated failure\nexit 22\n", encoding="utf-8")
    sh.chmod(0o755)
    return sh


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("speedbench")
    _, manifest, server = build_simulated_hub(
        root, big_size=BIG_SIZE, small_size=SMALL_SIZE, small_count=SMALL_COUNT, cap_bps=CAP_BPS
    )
    hub_base = server.start()
    baseline_tree = _extract_commit(root / "baseline", BASELINE_COMMIT)
    _write_manifest(baseline_tree, manifest)
    working_tree = _copy_working_tree(root / "working")
    _write_manifest(working_tree, manifest)
    yield root, manifest, server, hub_base, baseline_tree, working_tree
    server.stop()


@pytest.fixture(scope="module")
def baseline(bench) -> tuple[float, Path, int]:
    root, manifest, server, hub_base, baseline_tree, _ = bench
    data_dir = root / "baseline-data"
    elapsed, _out = _run_installer(baseline_tree, data_dir, hub_base)
    assert len(_verify_installed(data_dir, manifest)) == len(manifest["files"])
    return elapsed, data_dir, len(server.served)


def real_aria2() -> Path | None:
    """The genuine PATH lookup (``tests/conftest.py`` disables ``find_aria2c``)."""
    found = shutil.which("aria2c")
    return Path(found) if found else None


def test_baseline_installs_every_file_over_one_connection(bench, baseline) -> None:
    """The pre-change installer fetches each file with a single connection."""
    _, manifest, server, _, _, _ = bench
    elapsed, data_dir, served = baseline
    assert served == len(manifest["files"]), (
        "the baseline should make exactly one served request per file"
    )
    assert elapsed > BIG_SIZE / CAP_BPS, "the baseline should be cap-bound by the large file"
    print(f"baseline: {elapsed:.1f}s for {len(manifest['files'])} files, data dir {data_dir}")


def test_aria2_run_is_at_most_half_the_baseline(bench, baseline) -> None:
    if real_aria2() is None:
        pytest.skip("aria2c is not available: install it or set CLARITY_ARIA2C")
    root, manifest, server, hub_base, _, working_tree = bench
    data_dir = root / "new-data"
    elapsed, out = _run_installer(working_tree, data_dir, hub_base)
    installed = _verify_installed(data_dir, manifest)
    assert len(installed) == len(manifest["files"])
    assert "Installed:" in out
    # aria2 fetched the large file with parallel connections, not one stream.
    assert server.served.count("sam/sam_vit_b_01ec64.pth") > 1

    baseline_time, _, _ = baseline
    ratio = elapsed / baseline_time
    print(f"baseline: {baseline_time:.1f}s | aria2: {elapsed:.1f}s | ratio {ratio:.2f}")
    assert ratio <= MAX_NEW_RATIO, (
        f"aria2 run {elapsed:.1f}s vs baseline {baseline_time:.1f}s (ratio {ratio:.2f}, "
        f"limit {MAX_NEW_RATIO})"
    )


def test_forced_aria2_failure_falls_back_without_losing_time(
    bench, baseline, tmp_path: Path
) -> None:
    root, manifest, _server, hub_base, _, working_tree = bench
    data_dir = root / "fallback-data"
    broken = _failing_aria2(tmp_path)
    elapsed, out = _run_installer(
        working_tree, data_dir, hub_base, extra_env={"CLARITY_ARIA2C": str(broken)}
    )
    installed = _verify_installed(data_dir, manifest)
    assert len(installed) == len(manifest["files"])
    assert "Accelerated download unavailable" in out

    baseline_time, _, _ = baseline
    limit = baseline_time * FALLBACK_RATIO + FALLBACK_SLACK_S
    print(f"baseline: {baseline_time:.1f}s | fallback: {elapsed:.1f}s | limit {limit:.1f}s")
    assert elapsed <= limit, f"fallback run {elapsed:.1f}s exceeded {limit:.1f}s"
