"""Delete-while-streaming retry tests (Phase C).

A just-rendered file keeps an open 206 range-stream handle on Windows, so the
first unlink attempts can fail with WinError 32. The API must retry through
transient locks and only 500 once the lock persists.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from video_upscaler.web.server import create_app


def _locked_error():
    err = PermissionError(13, "Permission denied")
    err.winerror = 32
    return err


def test_delete_retries_through_transient_winerror32(tmp_path):
    app = create_app()
    client = TestClient(app)
    target = tmp_path / "rendered.mp4"
    target.write_bytes(b"video-bytes")

    # Hold the file open the way a 206 range-stream response would.
    held = open(target, "rb")
    try:
        real_unlink = Path.unlink
        attempts = {"n": 0}

        def flaky_unlink(self, *args, **kwargs):
            if self.resolve() == target.resolve():
                attempts["n"] += 1
                if attempts["n"] < 3:
                    if attempts["n"] == 2:
                        held.close()  # preview unload releases the stream handle
                    raise _locked_error()
            return real_unlink(self, *args, **kwargs)

        with patch.object(Path, "unlink", flaky_unlink), patch("time.sleep", return_value=None):
            resp = client.post(
                "/api/videos/delete",
                json={"video_name": target.name, "folder": str(tmp_path)},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["success"] is True
        assert attempts["n"] == 3
        assert not target.exists()
    finally:
        held.close()


def test_delete_reports_path_after_persistent_lock(tmp_path):
    app = create_app()
    client = TestClient(app)
    target = tmp_path / "locked.mp4"
    target.write_bytes(b"video-bytes")

    real_unlink = Path.unlink
    attempts = {"n": 0}

    def always_locked(self, *args, **kwargs):
        if self.resolve() == target.resolve():
            attempts["n"] += 1
            raise _locked_error()
        return real_unlink(self, *args, **kwargs)

    with patch.object(Path, "unlink", always_locked), patch("time.sleep", return_value=None):
        resp = client.post(
            "/api/videos/delete",
            json={"video_name": target.name, "folder": str(tmp_path)},
        )
    assert resp.status_code == 500
    assert attempts["n"] == 3
    assert "Failed to delete file" in resp.json()["detail"]
    assert target.exists()
