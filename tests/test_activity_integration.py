"""Integration tests for activity recording in JobManager and API endpoints."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from video_upscaler import config
from video_upscaler.web.activity import ActivityStore
from video_upscaler.web.jobs import JobInfo, JobManager
from video_upscaler.web.server import create_app


def test_restart_persists_history_and_seen_seq(tmp_path: Path) -> None:
    """Task 2.3: Re-instantiating the store from the same directory keeps entries and seq."""
    store1 = ActivityStore(data_dir=tmp_path)
    store1.record({
        "job_id": "job-1",
        "action": "Upscale",
        "status": "completed",
        "video_names": ["a.mp4"],
        "params": {"profile": "1x_Balanced"},
        "output_dir": str(tmp_path),
        "output_files": [str(tmp_path / "a_1x.mp4")],
        "elapsed_seconds": 2.5,
    })
    store1.record({
        "job_id": "job-2",
        "action": "Slow-motion",
        "status": "failed",
        "video_names": ["b.mp4"],
        "params": {"model": "AMT-S", "factor": 2},
        "output_dir": str(tmp_path),
        "output_files": [],
        "error_message": "Failed processing",
        "elapsed_seconds": 1.2,
    })
    store1.mark_seen()
    state1 = store1.get()
    assert state1["unread"] == 0
    assert len(state1["entries"]) == 2

    # "Restart": new store instance on same folder
    store2 = ActivityStore(data_dir=tmp_path)
    state2 = store2.get()
    assert state2["unread"] == 0
    assert len(state2["entries"]) == 2
    assert state2["entries"][0]["job_id"] == "job-2"
    assert state2["entries"][1]["job_id"] == "job-1"

    # Add 1 more in second session
    store2.record({
        "job_id": "job-3",
        "action": "Interpolate",
        "status": "completed",
        "video_names": ["c.mp4"],
        "params": {"model": "gmfss"},
        "output_dir": str(tmp_path),
        "output_files": [],
        "elapsed_seconds": 3.0,
    })
    state3 = store2.get()
    assert state3["unread"] == 1
    assert state3["entries"][0]["job_id"] == "job-3"


def test_job_thread_records_states_and_handles_store_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task 2.1: JobManager records completed, failed, cancelled and tolerates store failure."""
    from video_upscaler.web import activity

    test_store = ActivityStore(data_dir=tmp_path)
    monkeypatch.setattr(activity, "_store", test_store)

    mgr = JobManager()
    broadcast_events: list[dict] = []
    mgr.broadcast_sync = lambda data: broadcast_events.append(data)

    # 1. Completed job
    job1 = JobInfo(
        job_id="job-c",
        action="Upscale",
        video_names=["v1.mp4"],
        params={"profile": "2x_Balanced"},
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        "video_upscaler.web.jobs.process_videos",
        lambda *args, **kwargs: {"success": [tmp_path / "v1_2x.mp4"], "failed": [], "times": [1.0]},
    )
    mgr.jobs["job-c"] = job1
    mgr.active_job_id = "job-c"
    mgr._run_job_thread("job-c", [tmp_path / "v1.mp4"], job1.params, Path(job1.output_dir))

    entries = test_store.get()["entries"]
    assert len(entries) == 1
    assert entries[0]["job_id"] == "job-c"
    assert entries[0]["status"] == "completed"
    assert entries[0]["output_files"] == [str(tmp_path / "v1_2x.mp4")]
    assert any(ev.get("type") == "job_completed" for ev in broadcast_events)

    # 2. Failed job
    broadcast_events.clear()
    job2 = JobInfo(
        job_id="job-f",
        action="Upscale",
        video_names=["v2.mp4"],
        params={"profile": "2x_Balanced"},
        output_dir=str(tmp_path),
    )
    def fail_proc(*args, **kwargs):
        raise RuntimeError("Simulated processing crash")
    monkeypatch.setattr("video_upscaler.web.jobs.process_videos", fail_proc)

    mgr.jobs["job-f"] = job2
    mgr.active_job_id = "job-f"
    mgr._run_job_thread("job-f", [tmp_path / "v2.mp4"], job2.params, Path(job2.output_dir))

    entries = test_store.get()["entries"]
    assert len(entries) == 2
    assert entries[0]["job_id"] == "job-f"
    assert entries[0]["status"] == "failed"
    assert "Simulated processing crash" in (entries[0]["error_message"] or "")
    assert any(ev.get("type") == "job_completed" for ev in broadcast_events)

    # 3. Cancelled job
    broadcast_events.clear()
    job3 = JobInfo(
        job_id="job-x",
        action="Upscale",
        video_names=["v3.mp4"],
        params={"profile": "2x_Balanced"},
        output_dir=str(tmp_path),
        cancel_requested=True,
    )
    mgr.jobs["job-x"] = job3
    mgr.active_job_id = "job-x"
    mgr._run_job_thread("job-x", [tmp_path / "v3.mp4"], job3.params, Path(job3.output_dir))

    entries = test_store.get()["entries"]
    assert len(entries) == 3
    assert entries[0]["job_id"] == "job-x"
    assert entries[0]["status"] == "cancelled"
    assert any(ev.get("type") == "job_completed" for ev in broadcast_events)

    # 4. Storage error must not prevent job_completed broadcast
    broadcast_events.clear()
    job4 = JobInfo(
        job_id="job-err",
        action="Upscale",
        video_names=["v4.mp4"],
        params={"profile": "2x_Balanced"},
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(activity, "record", MagicMock(side_effect=IOError("Disk blown")))
    mgr.jobs["job-err"] = job4
    mgr.active_job_id = "job-err"
    mgr._run_job_thread("job-err", [tmp_path / "v4.mp4"], job4.params, Path(job4.output_dir))

    assert any(ev.get("type") == "job_completed" and ev["job"]["job_id"] == "job-err" for ev in broadcast_events)


def test_api_activity_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Task 2.2: GET /api/activity, POST /api/activity/seen, DELETE /api/activity."""
    from video_upscaler.web import activity

    test_store = ActivityStore(data_dir=tmp_path)
    monkeypatch.setattr(activity, "_store", test_store)

    client = TestClient(create_app())

    # Initially empty
    res = client.get("/api/activity")
    assert res.status_code == 200
    data = res.json()
    assert data["entries"] == []
    assert data["unread"] == 0

    # Record two jobs
    test_store.record({
        "job_id": "j1",
        "action": "Upscale",
        "status": "completed",
        "video_names": ["a.mp4"],
        "params": {"profile": "2x_Balanced"},
        "output_dir": str(tmp_path),
        "output_files": [str(tmp_path / "a_2x.mp4")],
        "elapsed_seconds": 1.0,
    })
    test_store.record({
        "job_id": "j2",
        "action": "Slow-motion",
        "status": "failed",
        "video_names": ["b.mp4"],
        "params": {"model": "AMT-S", "factor": 2},
        "output_dir": str(tmp_path),
        "output_files": [],
        "error_message": "Decode error",
        "elapsed_seconds": 2.0,
    })

    # GET /api/activity returns 2 entries, unread=2
    res = client.get("/api/activity")
    assert res.status_code == 200
    data = res.json()
    assert len(data["entries"]) == 2
    assert data["unread"] == 2
    assert data["entries"][0]["job_id"] == "j2"
    assert data["entries"][1]["job_id"] == "j1"

    # POST /api/activity/seen marks seen
    res_seen = client.post("/api/activity/seen")
    assert res_seen.status_code == 200
    assert res_seen.json()["ok"] is True

    # Now unread is 0
    res_after = client.get("/api/activity")
    assert res_after.json()["unread"] == 0
    assert len(res_after.json()["entries"]) == 2

    # DELETE /api/activity clears history
    res_del = client.delete("/api/activity")
    assert res_del.status_code == 200
    assert res_del.json()["ok"] is True

    # Now empty
    res_empty = client.get("/api/activity")
    assert res_empty.json()["entries"] == []
    assert res_empty.json()["unread"] == 0
