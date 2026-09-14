"""F2: progress reporting for on-demand AMT model download.

RED tests (must fail before the F2 implementation):
- install_entry forwards per-chunk (downloaded, total) progress ending at
  (total, total) with monotonic bytes.
- Slow-motion job with a stubbed hub invoking the callback broadcasts a
  job_progress event with the downloading stage before completion.
- No-callback regression: install_entry without progress_cb keeps working.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import PurePosixPath

from video_upscaler import config
from video_upscaler import modelhub
from video_upscaler.web import jobs as jobs_mod
from video_upscaler.web.jobs import JobInfo, JobManager


class _FakeResponse:
    def __init__(self, chunks: list[bytes], total: int) -> None:
        self._chunks = list(chunks)
        self.headers = {"Content-Length": str(total)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_install_entry_reports_monotonic_progress(tmp_path, monkeypatch):
    content = b"0123456789ABCDEF" * 64  # 1024 bytes
    chunks = [content[i : i + 256] for i in range(0, len(content), 256)]
    assert len(chunks) == 4
    entry = {
        "group": "amt",
        "path": "amt/amt-s.pth",
        "dest": "amt-s.pth",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    models_dir = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setenv(
        "CLARITY_MODEL_HUB_BASE", "https://hub.example/CLARITY_MODELS"
    )

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(list(chunks), len(content))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    seen: list[tuple[int, int | None]] = []
    installed = modelhub.install_entry(
        entry, progress_cb=lambda done, total: seen.append((done, total))
    )

    assert installed.read_bytes() == content
    assert seen, "expected at least one progress callback"
    totals = {total for _, total in seen}
    assert totals == {len(content)}
    dones = [done for done, _ in seen]
    assert dones == sorted(dones), "progress bytes must be monotonic"
    assert seen[-1] == (len(content), len(content))


def test_slow_motion_job_broadcasts_download_progress(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)

    def fake_install(entry, *args, **kwargs):
        progress_cb = kwargs.get("progress_cb")
        if progress_cb is not None:
            total = int(entry["size"])
            progress_cb(total // 2, total)
            progress_cb(total, total)
        dest = models_dir / PurePosixPath(entry["dest"]).name
        dest.write_bytes(b"fake-checkpoint")
        return dest

    monkeypatch.setattr("video_upscaler.modelhub.install_entry", fake_install)

    def fake_process(videos, model_key, factor, progress_cb):
        progress_cb(1, 1, 100)
        return {"success": ["clip_slowed2x_amt.mp4"], "failed": [], "times": [0.1]}

    monkeypatch.setattr(jobs_mod, "process_interpolate", fake_process)

    manager = JobManager()
    events: list[dict] = []
    monkeypatch.setattr(manager, "broadcast_sync", events.append)

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"input")
    params = {"model_key": "AMT-G", "factor": 2}
    job = JobInfo(
        job_id="amt1",
        action="Slow-motion",
        video_names=["clip.mp4"],
        params=dict(params),
        output_dir=str(out_dir),
    )
    manager.jobs["amt1"] = job
    manager._run_job_thread("amt1", [src], params, out_dir)

    assert job.status == "completed"
    progress_events = [e for e in events if e.get("type") == "job_progress"]
    assert any(
        "Downloading" in (e.get("job", {}).get("stage") or "")
        for e in progress_events
    ), f"expected a downloading job_progress event, got: {events!r}"


def test_install_entry_without_callback_still_works(tmp_path, monkeypatch):
    hub_dir = tmp_path / "hub"
    source = hub_dir / "amt"
    source.mkdir(parents=True)
    source.joinpath("amt-s.pth").write_bytes(b"xyz")
    monkeypatch.setenv("CLARITY_MODEL_HUB_BASE", str(hub_dir))
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    entry = {
        "group": "amt",
        "path": "amt/amt-s.pth",
        "dest": "amt-s.pth",
        "size": 3,
        "sha256": hashlib.sha256(b"xyz").hexdigest(),
    }
    installed = modelhub.install_entry(entry)
    assert installed.read_bytes() == b"xyz"
