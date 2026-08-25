"""End-to-end MatAnyone2 job on a synthetic clip (needs checkpoint + ffmpeg).

Skipped cleanly when the matanyone2.pth checkpoint or ffmpeg is unavailable
so CI machines without weights stay green.
"""

import base64
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest

from video_upscaler import config
from video_upscaler.ffmpeg import probe
from video_upscaler.matanyone2.mask import decode_mask_b64  # sanity import
from video_upscaler.matanyone2.model import checkpoint_path

# Generous bound: loading the real 141MB checkpoint can take a while on CPU.
# Once the job reaches a cancellable stage, cancel fires on the next progress
# callback, which is fast; these deadlines only cover the slow model setup.
_DEADLINE = 900.0

# test_interp.py mutates config.MODELS_DIR / AMT_SEGMENT_FRAMES via direct
# assignment without restoring them, so by the time this module's tests run
# in the full suite the dir has been pointed at a temp folder. Capture the
# pristine value at import (collection runs before any test) and re-pin it.
_REAL_MODELS_DIR = config.MODELS_DIR


@pytest.fixture(autouse=True)
def _restore_models_dir(monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", _REAL_MODELS_DIR)


_JOB_PARAMS = {
    "mask_png": None,  # filled per call
    "backend": "auto",
    "precision": "fp32",
    "warmup": 2,
}


def _checkpoint_available():
    try:
        checkpoint_path()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _checkpoint_available(), reason="matanyone2.pth not installed"),
    pytest.mark.skipif(config.ffmpeg_path() is None, reason="ffmpeg required"),
]


def _synthetic_clip(tmp_path, seconds=2.0, fps=12.0, with_audio=True):
    src = tmp_path / "integ.mp4"
    cmd = [config.ffmpeg_path(), "-y",
           "-f", "lavfi", "-i", f"testsrc=size=128x96:rate={fps}:duration={seconds}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(str(src))
    subprocess.run(cmd, check=True, capture_output=True)
    return src


def _rect_mask_b64(w=128, h=96):
    mask = np.zeros((h, w), np.uint8)
    mask[30:70, 45:85] = 255
    ok, buf = cv2.imencode(".png", mask)
    return base64.b64encode(buf.tobytes()).decode()


def _job_params():
    params = dict(_JOB_PARAMS)
    params["mask_png"] = _rect_mask_b64()
    return params


def _manager(tmp_path, monkeypatch):
    from video_upscaler.web.jobs import JobManager

    out_dir = tmp_path / "out"
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    manager = JobManager()
    original_broadcast = manager.broadcast_sync

    def broadcast_capture(payload):
        if payload.get("type") == "job_completed":
            completed.append(payload)
        original_broadcast(payload)

    completed = []
    manager.broadcast_sync = broadcast_capture
    return manager, out_dir, completed


def _run_job(src, tmp_path, monkeypatch):
    manager, _, completed = _manager(tmp_path, monkeypatch)
    started = manager.submit_job("MatAnyone2", [src], _job_params(), tmp_path / "out")

    deadline = time.time() + _DEADLINE
    while time.time() < deadline and not completed:
        time.sleep(0.25)
    assert completed, "job_completed never broadcast"
    return manager.get_job(started.job_id)


def test_matanyone2_e2e(tmp_path, monkeypatch):
    """Full pipeline: synthetic clip -> mask -> matte+green-screen MP4s.

    Verifies the merge-gating contract: streamed frames land at the source
    fps/duration, audio is routed to the green-screen output only.
    """
    src = _synthetic_clip(tmp_path)
    job = _run_job(src, tmp_path, monkeypatch)
    assert job is not None
    assert job.status == "completed", (job.status, job.error_message)

    out_dir = tmp_path / "out"
    fg = out_dir / "integ_greenscreen.mp4"
    matte = out_dir / "integ_matte.mp4"
    assert fg.exists(), "green-screen MP4 missing"
    assert matte.exists(), "matte MP4 missing"

    info_fg = probe(fg)
    assert abs(info_fg["fps"] - 12.0) < 0.05
    assert abs(info_fg["duration"] - 2.0) < 0.5
    assert info_fg["has_audio"] is True

    info_matte = probe(matte)
    assert info_matte["has_audio"] is False


def test_matanyone2_cancel(tmp_path, monkeypatch):
    """Cancel mid-job lands on 'cancelled' and stops further processing."""
    src = _synthetic_clip(tmp_path, seconds=30.0)
    manager, _, _ = _manager(tmp_path, monkeypatch)
    started = manager.submit_job("MatAnyone2", [src], _job_params(), tmp_path / "out")

    stages_seen = []
    deadline = time.time() + _DEADLINE
    cancellable = False
    while time.time() < deadline:
        current = manager.get_job(started.job_id)
        if current is None:
            time.sleep(0.2)
            continue
        stages_seen.append(current.stage)
        if current.stage in ("Loading model", "Warming up") or "Processing" in current.stage:
            cancellable = True
            break
        if current.status in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.2)
    assert cancellable, (
        f"job never reached a cancellable stage (stages seen: {stages_seen[-20:]})"
    )

    assert manager.cancel_job(started.job_id) is True

    deadline = time.time() + _DEADLINE
    settled = None
    while time.time() < deadline:
        current = manager.get_job(started.job_id)
        if current is not None and current.status in ("cancelled", "failed", "completed"):
            settled = current
            break
        time.sleep(0.2)
    assert settled is not None, "job did not settle after cancel"
    assert settled.status == "cancelled"
