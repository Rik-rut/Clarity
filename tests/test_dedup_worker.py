"""Persistent MultiPassDedup worker: protocol handling and progress mapping.

The worker keeps the interpolation model resident so repeated renders skip
the per-video load. These tests exercise the client protocol and the
``process_dedup`` progress mapping without loading real models.
"""

from __future__ import annotations

import io
import json

import pytest

from video_upscaler import config
from video_upscaler import dedup as dedup_mod


class _FakeProcess:
    """Duck-typed stand-in for the worker subprocess pipes."""

    def __init__(self, events):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("".join(json.dumps(e) + "\n" for e in events))
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True


def _worker_with(monkeypatch, events):
    worker = dedup_mod._DedupWorker()
    fake = _FakeProcess(events)

    def fake_start():
        worker._process = fake
        return fake

    monkeypatch.setattr(worker, "_start", fake_start)
    return worker, fake


def test_exchange_forwards_progress_and_returns_done(monkeypatch):
    worker, fake = _worker_with(
        monkeypatch,
        [
            {"event": "progress", "done": 10, "total": 20},
            {"event": "progress", "done": 20, "total": 20},
            {"event": "done", "output": "x.mp4"},
        ],
    )
    seen: list[tuple[int, int]] = []

    event = worker._exchange(
        {"cmd": "run"}, on_progress=lambda done, total: seen.append((done, total))
    )

    assert seen == [(10, 20), (20, 20)]
    assert event["event"] == "done"
    assert json.loads(fake.stdin.getvalue())["cmd"] == "run"


def test_exchange_raises_worker_error_and_keeps_worker_alive(monkeypatch):
    """A per-video error must not kill the healthy (model-loaded) worker."""
    worker, fake = _worker_with(monkeypatch, [{"event": "error", "message": "boom"}])

    with pytest.raises(RuntimeError, match="boom"):
        worker._exchange({"cmd": "run"})

    assert worker._process is fake
    assert fake.killed is False


def test_exchange_tears_down_when_progress_callback_raises(monkeypatch):
    """A cancelled render must not leave a busy worker behind."""
    worker, fake = _worker_with(
        monkeypatch, [{"event": "progress", "done": 1, "total": 10}]
    )

    def canceling(done, total):
        raise RuntimeError("Job cancelled by user")

    with pytest.raises(RuntimeError, match="cancelled"):
        worker._exchange({"cmd": "run"}, on_progress=canceling)

    assert worker._process is None
    assert fake.killed is True


def test_exchange_reports_worker_death(monkeypatch):
    worker = dedup_mod._DedupWorker()
    fake = _FakeProcess([])

    def fake_start():
        worker._process = fake
        return fake

    monkeypatch.setattr(worker, "_start", fake_start)

    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        worker._exchange({"cmd": "run"})


def test_preload_marks_model_loaded(monkeypatch):
    worker, fake = _worker_with(monkeypatch, [{"event": "ready", "model": "gmfss"}])

    assert worker.preload("gmfss", 1.0) is True
    assert worker.is_loaded("gmfss", 1.0) is True
    assert json.loads(fake.stdin.getvalue())["cmd"] == "load"

    # Second preload is a no-op (already resident).
    assert worker.preload("gmfss", 1.0) is True


def test_process_dedup_maps_worker_progress_to_file_percent(tmp_path, monkeypatch):
    from video_upscaler import processor

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(config, "OUTPUT_DIR", out)

    def fake_infer(**kwargs):
        kwargs["progress_cb"](0.5)
        kwargs["progress_cb"](1.0)

    monkeypatch.setattr("video_upscaler.dedup.run_dedup_infer", fake_infer)
    updates: list[tuple[int, int, int]] = []

    results = processor.process_dedup(
        [src],
        "gmfss",
        0,
        2,
        progress_cb=lambda index, count, percent: updates.append(
            (index, count, percent)
        ),
    )

    assert results["success"]
    assert (1, 1, 50) in updates
    assert updates[-1] == (1, 1, 100)
