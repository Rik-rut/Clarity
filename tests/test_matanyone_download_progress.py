"""MatAnyone2 first-use downloads report progress instead of freezing.

The MatAnyone2 branch used to install the matting + SAM checkpoints with no
progress, leaving the UI at "Starting MatAnyone2..." for the whole transfer.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from video_upscaler import config
from video_upscaler.web import jobs as jobs_mod
from video_upscaler.web.jobs import JobInfo, JobManager


def test_matanyone_job_reports_download_progress(tmp_path, monkeypatch):
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
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-checkpoint")
        return dest

    monkeypatch.setattr("video_upscaler.modelhub.install_entry", fake_install)

    def fake_process(videos, params, progress_cb, stage_cb):
        stage_cb("Loading MatAnyone2…")
        progress_cb(1, 1, 100)
        return {"success": ["clip_matted.mp4"], "failed": [], "times": [0.1]}

    monkeypatch.setattr(jobs_mod, "process_matanyone2", fake_process)

    manager = JobManager()
    events: list[dict] = []
    monkeypatch.setattr(manager, "broadcast_sync", events.append)

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"input")
    params = {"mask_png": "mask.png"}
    job = JobInfo(
        job_id="ma1",
        action="MatAnyone2",
        video_names=["clip.mp4"],
        params=dict(params),
        output_dir=str(out_dir),
    )
    manager.jobs["ma1"] = job
    manager._run_job_thread("ma1", [src], params, out_dir)

    assert job.status == "completed"
    progress_events = [e for e in events if e.get("type") == "job_progress"]
    download_events = [
        e
        for e in progress_events
        if (e.get("job", {}).get("stage") or "").startswith("Downloading")
    ]
    assert download_events, f"expected download progress events, got: {events!r}"
    assert all(e["job"]["percent"] <= 5 for e in download_events)
    assert any(e["job"]["stage"] == "Loading MatAnyone2…" for e in progress_events)
