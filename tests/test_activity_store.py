"""Unit tests for the activity history store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_upscaler.web.activity import (
    MAX_ENTRIES,
    MAX_ERROR_CHARS,
    STATE_VERSION,
    ActivityStore,
    filter_summary,
    truncate_error,
)


def test_params_whitelisting() -> None:
    raw_params = {
        "profile": "1x_Balanced",
        "model": "gmfss",
        "factor": 2,
        "npass": 1,
        "outputs": ["green_screen", "alpha_matte"],
        "blend_edges": True,
        "mask_png": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
        "huge_binary_blob": b"\x00\x01\x02",
        "random_param": 12345,
    }
    summary = filter_summary("MatAnyone2", raw_params)
    assert "mask_png" not in summary
    assert "huge_binary_blob" not in summary
    assert "random_param" not in summary
    assert summary["profile"] == "1x_Balanced"
    assert summary["model"] == "gmfss"
    assert summary["factor"] == 2
    assert summary["npass"] == 1
    assert summary["outputs"] == ["green_screen", "alpha_matte"]
    assert summary["blend_edges"] is True


def test_error_truncation() -> None:
    assert truncate_error(None) is None
    assert truncate_error("") is None
    short_err = "File not found"
    assert truncate_error(short_err) == short_err

    long_err = "X" * (MAX_ERROR_CHARS + 500)
    truncated = truncate_error(long_err)
    assert truncated is not None
    assert len(truncated) == MAX_ERROR_CHARS + 1  # 2000 chars + '…'
    assert truncated.endswith("…")


def test_append_order_and_seq(tmp_path: Path) -> None:
    store = ActivityStore(data_dir=tmp_path)
    res = store.get()
    assert res["entries"] == []
    assert res["unread"] == 0

    e1 = store.record({
        "job_id": "job-1",
        "action": "Upscale",
        "status": "completed",
        "video_names": ["a.mp4"],
        "params": {"profile": "2x_Balanced"},
        "output_dir": str(tmp_path),
        "output_files": [str(tmp_path / "a_2x.mp4")],
        "elapsed_seconds": 1.5,
    })
    assert e1["seq"] == 1

    e2 = store.record({
        "job_id": "job-2",
        "action": "Slow-motion",
        "status": "failed",
        "video_names": ["b.mp4"],
        "params": {"model": "AMT-S", "factor": 2},
        "output_dir": str(tmp_path),
        "output_files": [],
        "error_message": "CUDA out of memory",
        "elapsed_seconds": 0.5,
    })
    assert e2["seq"] == 2

    state = store.get()
    assert len(state["entries"]) == 2
    # Newest first
    assert state["entries"][0]["job_id"] == "job-2"
    assert state["entries"][1]["job_id"] == "job-1"
    assert state["unread"] == 2

    # Mark seen
    store.mark_seen()
    assert store.get()["unread"] == 0


def test_trim_at_20_and_unread_after_trim(tmp_path: Path) -> None:
    store = ActivityStore(data_dir=tmp_path)

    # Record 25 entries
    for i in range(1, 26):
        store.record({
            "job_id": f"job-{i}",
            "action": "Upscale",
            "status": "completed",
            "video_names": [f"vid_{i}.mp4"],
            "params": {"profile": "2x_Balanced"},
            "output_dir": str(tmp_path),
            "output_files": [str(tmp_path / f"vid_{i}_2x.mp4")],
            "elapsed_seconds": float(i),
        })

    state = store.get()
    # Must be capped at MAX_ENTRIES (20)
    assert len(state["entries"]) == MAX_ENTRIES
    # Newest is job-25 (seq 25)
    assert state["entries"][0]["job_id"] == "job-25"
    assert state["entries"][0]["seq"] == 25
    # Oldest retained is job-6 (seq 6)
    assert state["entries"][-1]["job_id"] == "job-6"
    assert state["entries"][-1]["seq"] == 6

    # All 20 retained entries are unread since last_seen_seq is 0
    assert state["unread"] == 20

    # Now mark seen
    store.mark_seen()
    assert store.get()["unread"] == 0

    # Add 1 more entry (seq 26)
    store.record({
        "job_id": "job-26",
        "action": "Upscale",
        "status": "completed",
        "video_names": ["vid_26.mp4"],
        "params": {},
        "output_dir": str(tmp_path),
        "output_files": [],
        "elapsed_seconds": 1.0,
    })
    # Now unread is exactly 1 (seq 26 > last_seen 25)
    state2 = store.get()
    assert state2["unread"] == 1
    assert len(state2["entries"]) == MAX_ENTRIES
    assert state2["entries"][-1]["seq"] == 7  # job-6 was trimmed


def test_clear_history(tmp_path: Path) -> None:
    store = ActivityStore(data_dir=tmp_path)
    store.record({
        "job_id": "job-1",
        "action": "Upscale",
        "status": "completed",
        "video_names": ["a.mp4"],
        "params": {},
        "output_dir": str(tmp_path),
        "output_files": [],
        "elapsed_seconds": 1.0,
    })
    assert len(store.get()["entries"]) == 1
    store.clear()
    state = store.get()
    assert state["entries"] == []
    assert state["unread"] == 0


def test_resilience_missing_corrupt_and_wrong_version(tmp_path: Path) -> None:
    store = ActivityStore(data_dir=tmp_path)
    # 1. Missing file: returns clean default
    assert store.get() == {"entries": [], "unread": 0}

    activity_file = tmp_path / "activity.json"

    # 2. Corrupt JSON
    activity_file.write_text("not json {{{", encoding="utf-8")
    assert store.get() == {"entries": [], "unread": 0}

    # 3. Wrong version
    activity_file.write_text(json.dumps({"version": 999, "entries": [{"job_id": "bad"}]}), encoding="utf-8")
    assert store.get() == {"entries": [], "unread": 0}


def test_resilience_unwritable_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ActivityStore(data_dir=tmp_path)
    # Monkeypatch NamedTemporaryFile to simulate permission denied
    def fake_temp(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr("tempfile.NamedTemporaryFile", fake_temp)
    # record must not raise
    store.record({
        "job_id": "job-err",
        "action": "Upscale",
        "status": "completed",
        "video_names": ["a.mp4"],
        "params": {},
        "output_dir": str(tmp_path),
        "output_files": [],
        "elapsed_seconds": 1.0,
    })
    # Should handle error gracefully without throwing
