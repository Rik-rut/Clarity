"""Web-layer additions: first-frame endpoint and MatAnyone2 job dispatch."""

import base64
import threading
import time

import cv2
import numpy as np
import pytest

from video_upscaler import config
from video_upscaler.web.jobs import job_manager


def _require_ffmpeg():
    if not config.ffmpeg_path():
        pytest.skip("ffmpeg required")


def _tiny_video(tmp_path, name="ma2web.mp4"):
    import subprocess

    src = tmp_path / name
    subprocess.run(
        [config.ffmpeg_path(), "-y", "-f", "lavfi",
         "-i", "testsrc=size=16x16:rate=5:duration=0.4",
         "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True,
    )
    return src


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from video_upscaler.web.server import create_app

    app = create_app()
    with TestClient(app) as tc:
        yield tc


def test_frame_endpoint_returns_jpeg(client, tmp_path):
    _require_ffmpeg()
    src = _tiny_video(tmp_path)
    response = client.get("/api/videos/frame", params={"path": str(src)})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert len(response.content) > 0


def test_frame_endpoint_rejects_nonzero_index(client, tmp_path):
    _require_ffmpeg()
    src = _tiny_video(tmp_path)
    r = client.get("/api/videos/frame", params={"path": str(src), "n": "3"})
    assert r.status_code == 400


def test_frame_endpoint_missing_file_404(client):
    r = client.get("/api/videos/frame", params={"path": "Z:/nope.mp4"})
    assert r.status_code == 404


def test_matanyone_job_requires_mask(client, tmp_path):
    _require_ffmpeg()
    src = _tiny_video(tmp_path)
    started = client.post(
        "/api/jobs/start",
        json={"action": "MatAnyone2", "video_names": [str(src)], "params": {}},
    )
    assert started.status_code == 200
    job_id = started.json()["job"]["job_id"]
    deadline = time.time() + 10
    job = None
    while time.time() < deadline:
        current = job_manager.get_job(job_id)
        if current is not None and current.status in ("failed", "completed", "cancelled"):
            job = current.to_dict()
            break
        time.sleep(0.05)
    assert job is not None
    assert job["status"] == "failed"
    assert "mask" in job["error_message"].lower()


def test_unknown_action_still_rejected(client, tmp_path):
    _require_ffmpeg()
    src = _tiny_video(tmp_path)
    started = client.post(
        "/api/jobs/start",
        json={"action": "Nonsense", "video_names": [str(src)], "params": {}},
    )
    assert started.status_code == 200  # accepted, fails in worker like today
    job_id = started.json()["job"]["job_id"]
    deadline = time.time() + 10
    job = None
    while time.time() < deadline:
        current = job_manager.get_job(job_id)
        if current is not None and current.status in ("failed", "completed", "cancelled"):
            job = current.to_dict()
            break
        time.sleep(0.05)
    assert job is not None
    assert job["status"] == "failed"
