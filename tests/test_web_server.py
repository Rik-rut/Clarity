"""Tests for FastAPI web server and video range streaming."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from video_upscaler.web.server import create_app


def test_server_root_endpoint():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code in (200, 404)


def test_video_streaming_range_requests(tmp_path):
    app = create_app()
    client = TestClient(app)

    dummy_vid = tmp_path / "dummy.mp4"
    data = b"preview-video-bytes-data-content" * 100
    dummy_vid.write_bytes(data)
    total_size = len(data)

    resp_full = client.get("/api/stream/video", params={"path": str(dummy_vid.resolve())})
    assert resp_full.status_code == 200
    assert resp_full.headers["accept-ranges"] == "bytes"
    assert int(resp_full.headers["content-length"]) == total_size

    resp_range = client.get(
        "/api/stream/video",
        params={"path": str(dummy_vid.resolve())},
        headers={"Range": "bytes=0-499"},
    )
    assert resp_range.status_code == 206
    assert resp_range.headers["content-range"] == f"bytes 0-499/{total_size}"
    assert resp_range.headers["content-length"] == "500"
    assert len(resp_range.content) == 500
    assert resp_range.content == data[:500]


def test_video_streaming_nonexistent_file():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/stream/video", params={"path": "/non/existent/video.mp4..."})
    assert resp.status_code == 404
