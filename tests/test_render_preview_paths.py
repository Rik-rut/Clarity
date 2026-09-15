"""Phase B: render-result previews must stream via data-dir paths.

Regression tests for the 404 on ``GET /api/stream/video?path=output\\<name>.mp4``
after a render finishes:

- Producer (jobs.py): ``job.output_files`` must hold absolute paths so the
  consumer fast path hits.
- Consumer (routes/api.py): relative ``path`` must resolve against
  ``config.DATA_DIR`` (user-media root), not ``config.BASE_DIR``, with a
  containment clamp that rejects ``..`` escapes outside the data root.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from video_upscaler.web.server import create_app

PAYLOAD = b"preview-video-bytes-data-content" * 100


def _native_relative() -> str:
    """Relative ``output/<file>`` using the platform-native separator.

    On Windows this yields backslashes (the reported failure shape);
    elsewhere forward slashes. Both must stream.
    """
    return str(Path("output") / "render_result.mp4")


def test_stream_relative_output_path_resolves_against_data_dir(tmp_path, monkeypatch):
    from video_upscaler import config

    data_dir = tmp_path / "data"
    out_dir = data_dir / "output"
    out_dir.mkdir(parents=True)
    (out_dir / "render_result.mp4").write_bytes(PAYLOAD)
    # Simulate the desktop install where the data root differs from the code dir.
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)

    client = TestClient(create_app())
    resp = client.get("/api/stream/video", params={"path": _native_relative()})

    assert resp.status_code == 200
    assert int(resp.headers["content-length"]) == len(PAYLOAD)


def test_stream_relative_escape_outside_data_dir_rejected(tmp_path, monkeypatch):
    from video_upscaler import config

    data_dir = tmp_path / "data"
    (data_dir / "output").mkdir(parents=True)
    secret = tmp_path / "secret.mp4"
    secret.write_bytes(PAYLOAD)
    assert secret.resolve() != (data_dir / "secret.mp4").resolve()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "OUTPUT_DIR", data_dir / "output")

    client = TestClient(create_app())
    escape = str(Path("..") / "secret.mp4") if os.name != "nt" else "..\\secret.mp4"
    resp = client.get("/api/stream/video", params={"path": escape})

    assert resp.status_code == 403


def test_job_output_files_are_absolute_after_success(tmp_path, monkeypatch):
    from video_upscaler.web import jobs as jobs_mod
    from video_upscaler.web.jobs import JobInfo, JobManager

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Processors historically returned backend-relative paths ("output\\<name>"),
    # which production resolves via Path.resolve() — i.e. anchored at CWD.
    # Pin CWD to the job's out_dir so the containment assertion below is
    # deterministic regardless of the invoker's working directory.
    monkeypatch.chdir(out_dir)
    backend_relative = Path("output") / "render_done.mp4"

    def fake_process(videos, profile, progress_cb, stage_cb=None):
        progress_cb(1, 1, 100)
        return {"success": [backend_relative], "failed": [], "times": [0.1]}

    monkeypatch.setattr(jobs_mod, "process_videos", fake_process)

    manager = JobManager()
    src = tmp_path / "a.mp4"
    src.write_bytes(b"input")
    job = JobInfo(
        job_id="t1",
        action="Upscale",
        video_names=["a.mp4"],
        params={"profile": "2x_Balanced"},
        output_dir=str(out_dir),
    )
    manager.jobs["t1"] = job
    manager._run_job_thread("t1", [src], {"profile": "2x_Balanced"}, out_dir)

    assert job.status == "completed"
    assert job.output_files
    resolved = Path(job.output_files[0])
    assert resolved.is_absolute()
    assert resolved.is_relative_to(out_dir.resolve())
