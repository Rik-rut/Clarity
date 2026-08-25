"""Tests for REST API endpoints (System, videos, directories, jobs)."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from video_upscaler.web.server import create_app


def test_system_info_endpoint():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/system/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "backend" in data
    assert "backend_label" in data
    assert "profiles" in data
    assert "slow_mo_models" in data
    assert "dedup_models" in data
    assert len(data["profiles"]) > 0



def test_validate_directory():
    app = create_app()
    client = TestClient(app)
    resp = client.post("/api/directories/validate", json={"path": "output/test_valid"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True



def test_video_upload_and_scanned(tmp_path):
    app = create_app()
    client = TestClient(app)
    file_content = b"\x00" * 50 + b"dummy mp4 bytes"
    resp = client.post(
        "/api/videos/upload",
        files={"file": ("test_upload.mp4", file_content, "video/mp4")},
        data={"target_folder": str(tmp_path.resolve())},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["video"]["name"] == "test_upload.mp4"

    resp_scan = client.get("/api/videos/scanned", params={"folder": str(tmp_path.resolve())})
    assert resp_scan.status_code == 200
    scan_data = resp_scan.json()
    names = [v["name"] for v in scan_data["videos"]]
    assert "test_upload.mp4" in names


def test_job_start_and_cancel(tmp_path):
    import time
    app = create_app()
    client = TestClient(app)

    vid = tmp_path / "sample.mp4"
    vid.write_bytes(b"video")

    def slow_process(videos, profile, progress_cb):
        for i in range(10):
            time.sleep(0.05)
            progress_cb(1, 1, i * 10)
        return {"success": [vid], "failed": [], "times": [1.0]}

    with patch("video_upscaler.web.jobs.process_videos", side_effect=slow_process):
        resp = client.post(
            "/api/jobs/start",
            json={
                "action": "Upscale",
                "video_names": ["sample.mp4"],
                "input_dir": str(tmp_path),
                "output_dir": str(tmp_path / "out"),
                "params": {"profile": "2x_Balanced"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        job_id = data["job"]["job_id"]

        # Status check
        status_resp = client.get("/api/jobs/status")
        assert status_resp.status_code == 200

        cancel_resp = client.post(f"/api/jobs/{job_id}/cancel")
        assert cancel_resp.status_code in (200, 404)


def test_video_rename_delete_clear(tmp_path):
    app = create_app()
    client = TestClient(app)

    vid1 = tmp_path / "vid1.mp4"
    vid2 = tmp_path / "vid2.mkv"
    vid1.write_bytes(b"content1")
    vid2.write_bytes(b"content2")

    # Rename
    rename_resp = client.post(
        "/api/videos/rename",
        json={"old_name": "vid1.mp4", "new_name": "renamed_vid1.mp4", "folder": str(tmp_path)},
    )
    assert rename_resp.status_code == 200
    assert rename_resp.json()["video"]["name"] == "renamed_vid1.mp4"
    assert not (tmp_path / "vid1.mp4").exists()
    assert (tmp_path / "renamed_vid1.mp4").exists()

    # Delete single
    del_resp = client.post(
        "/api/videos/delete",
        json={"video_name": "renamed_vid1.mp4", "folder": str(tmp_path)},
    )
    assert del_resp.status_code == 200
    assert not (tmp_path / "renamed_vid1.mp4").exists()
    assert (tmp_path / "vid2.mkv").exists()

    # Clear all
    clear_resp = client.post(
        "/api/videos/clear",
        json={"folder": str(tmp_path)},
    )
    assert clear_resp.status_code == 200
    assert clear_resp.json()["deleted_count"] >= 1
    assert not (tmp_path / "vid2.mkv").exists()


def test_browse_directory_endpoint():
    app = create_app()
    client = TestClient(app)
    with patch("tkinter.filedialog.askdirectory", return_value="C:/MockFolder"):
        resp = client.post("/api/directories/browse", json={"initial_dir": "C:/"})
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert resp.json()["path"] == "C:/MockFolder"


def test_system_reset_endpoint():
    from video_upscaler.memory import free_gpu_memory

    # Direct test
    mem_result = free_gpu_memory()
    assert mem_result["success"] is True

    # Endpoint test
    app = create_app()
    client = TestClient(app)
    resp = client.post("/api/system/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "GPU memory" in data["message"]
    assert "vram" in data


