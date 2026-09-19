"""A cancelled job keeps the render slot until its worker thread exits.

A cancelled render can still be inside an uninterruptible TensorRT engine
build; allowing a second job to start would race both of them on the same
engine/ONNX cache files and on ``config.OUTPUT_DIR``.
"""

from __future__ import annotations

import threading
import time

import pytest

from video_upscaler.web import jobs as jobs_mod
from video_upscaler.web.jobs import JobManager


def test_cancelled_job_blocks_a_new_render_until_the_thread_exits(
    tmp_path, monkeypatch
):
    started = threading.Event()
    release = threading.Event()

    def fake_process(videos, model_key, factor, progress_cb, stage_cb=None):
        if stage_cb is not None:
            stage_cb("Building the engine…")
        started.set()
        # Simulate an uninterruptible engine build: cancellation is only
        # observed at the next progress checkpoint.
        release.wait(timeout=10)
        progress_cb(1, 1, 100)
        return {"success": [], "failed": [], "times": []}

    monkeypatch.setattr(jobs_mod, "process_interpolate", fake_process)
    monkeypatch.setattr("video_upscaler.interp.check_amt", lambda key: None)
    monkeypatch.setattr(
        "video_upscaler.memory.free_gpu_memory", lambda: {"success": True}
    )

    manager = JobManager()
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"input")
    out = tmp_path / "out"
    out.mkdir()
    params = {"model_key": "AMT-S", "factor": 2}

    job = manager.submit_job("Slow-motion", [src], params, out)
    assert started.wait(timeout=5)

    assert manager.cancel_job(job.job_id) is True
    assert manager.get_active_job() is not None, "slot released before the thread finished"

    with pytest.raises(RuntimeError, match="still finishing"):
        manager.submit_job("Slow-motion", [src], params, out)

    release.set()
    deadline = time.time() + 10
    while time.time() < deadline and manager.get_active_job() is not None:
        time.sleep(0.05)
    assert manager.get_active_job() is None

    # Once the worker has exited, a new render is accepted again.
    job2 = manager.submit_job("Slow-motion", [src], params, out)
    assert job2.job_id
