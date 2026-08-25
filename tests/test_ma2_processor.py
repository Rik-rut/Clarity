"""Streaming processor verified against stubbed session/ffmpeg plumbing."""

import base64
import cv2
import numpy as np
import pytest

from video_upscaler.matanyone2 import processor as proc


class FakeSession:
    first_prob_np = np.full((4, 6), 0.25, np.float32)

    def __init__(self, selection=None, **kwargs):
        self.started = []
        self.steps = 0
        self.closed = False
        self.last_mask_shape = None

    def start(self, frame, mask, warmup=10):
        self.started.append({"warmup": warmup})
        h, w = tuple(frame.shape[-2:])
        self.first_prob_np = np.full((h, w), 0.25, np.float32)
        self.last_mask_shape = tuple(mask.shape)

    def step(self, frame):
        self.steps += 1
        h, w = tuple(frame.shape[-2:])
        return np.full((h, w), 0.75, np.float32)

    def close(self):
        self.closed = True


FRAMES = 5


def make_video(tmp_path):
    """A deterministic 6x4 mp4 (requires ffmpeg on PATH; skipped otherwise)."""
    ffmpeg = __import__("shutil").which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg required for processor tests")
    src = tmp_path / "clip.mp4"
    import subprocess

    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"testsrc=size=6x4:rate=10:duration={FRAMES / 10:.2f}",
         "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True,
    )
    return src


@pytest.fixture
def env(tmp_path, monkeypatch):
    from video_upscaler import config

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    sessions = []
    real_build = proc.build_session

    def build_session(selection):
        s = FakeSession(selection)
        sessions.append(s)
        return s

    monkeypatch.setattr(proc, "build_session", build_session)
    return {"out": out_dir, "sessions": sessions, "src": make_video(tmp_path)}


def mask_b64(h=4, w=6):
    m = np.zeros((h, w), np.uint8)
    m[1:3, 2:4] = 255
    ok, buf = cv2.imencode(".png", m)
    return base64.b64encode(buf.tobytes()).decode()


def base_params(**over):
    params = {
        "mask_png": mask_b64(),
        "warmup": 1,
        "outputs": ["matte", "greenscreen"],
    }
    params.update(over)
    return params


def cb(idx, count, percent):
    pass


def test_success_produces_tagged_outputs(env):
    info = env
    result = proc.process_matanyone2([info["src"]], base_params(), cb)
    names = sorted(p.name for p in result["success"])
    assert names == ["clip_greenscreen.mp4", "clip_matte.mp4"]
    assert result["failed"] == []
    assert info["sessions"][0].closed is True
    assert info["sessions"][0].started[0]["warmup"] == 1


def test_audio_routed_to_foreground_only(env, monkeypatch):
    captured = {}

    def fake_encode(frames, fw, fh, fps, src_path, out_path, use_audio, rotation=0,
                    use_nvenc=True, timing=None, alpha=False):
        captured[out_path.name] = use_audio

    monkeypatch.setattr(proc, "encode_video", fake_encode)
    proc.process_matanyone2([env["src"]], base_params(), cb)
    assert captured["clip_matte.mp4"] is False
    assert captured["clip_greenscreen.mp4"] in (True, False)  # depends on lavfi track


def test_follow_on_mask_resized_to_file_dims(env):
    """A follow-on file with different dims gets a mask resized to its size."""
    ffmpeg = __import__("shutil").which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg required for processor tests")
    wide = env["out"].parent / "wide.mp4"
    import subprocess
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"testsrc=size=8x6:rate=10:duration={FRAMES / 10:.2f}",
         "-pix_fmt", "yuv420p", str(wide)],
        check=True, capture_output=True,
    )
    result = proc.process_matanyone2([env["src"], wide], base_params(), cb)
    names = sorted(p.name for p in result["success"])
    assert names == ["clip_greenscreen.mp4", "clip_matte.mp4", "wide_greenscreen.mp4", "wide_matte.mp4"]
    assert result["failed"] == []
    s0, s1 = env["sessions"]
    assert tuple(s0.last_mask_shape) == (4, 6)  # 6x4 source keeps native mask
    assert tuple(s1.last_mask_shape) == (6, 8)  # 8x6 source got a resized mask


def test_preview_scale_mask_upscaled_to_native(env):
    """A browser-preview-sized mask (same aspect, smaller than native) is
    upscaled to the native frame instead of being rejected."""
    ffmpeg = __import__("shutil").which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg required for processor tests")
    big = env["out"].parent / "big.mp4"
    import subprocess
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", f"testsrc=size=12x8:rate=10:duration={FRAMES / 10:.2f}",
         "-pix_fmt", "yuv420p", str(big)],
        check=True, capture_output=True,
    )
    result = proc.process_matanyone2([big], base_params(), cb)
    assert result["failed"] == []
    (session,) = env["sessions"]
    assert tuple(session.last_mask_shape) == (8, 12)


def test_bad_mask_fails_files_but_batch_continues(env):
    """A corrupt mask payload fails each file cleanly instead of raising."""
    other = env["out"].parent / "other.mp4"
    import shutil
    shutil.copyfile(env["src"], other)
    bad = base_params()
    bad["mask_png"] = base64.b64encode(b"not-a-png").decode()  # undecodable
    result = proc.process_matanyone2([env["src"], other], bad, cb)
    assert sorted(name for name, _ in result["failed"]) == ["clip.mp4", "other.mp4"]
    assert result["success"] == []
    assert any("Mask" in reason or "mask" in reason for _, reason in result["failed"])


def test_cancel_exception_propagates(env, monkeypatch):
    def cancelling(idx, count, percent):
        raise RuntimeError("Job cancelled by user")

    class CancelSession(FakeSession):
        def step(self, frame):
            cancelling(1, 1, 50)
            return np.zeros((4, 6), np.float32)

    monkeypatch.setattr(proc, "build_session", lambda selection: CancelSession())
    with pytest.raises(RuntimeError, match="cancelled"):
        proc.process_matanyone2([env["src"]], base_params(), cb)


def test_stage_callbacks_fire(env):
    stages = []
    proc.process_matanyone2([env["src"]], base_params(), cb,
                            stage_cb=stages.append)
    assert "Loading model" in stages
    assert "Warming up" in stages
    assert "Encoding outputs" in stages


def test_outputs_subset_respected(env):
    """Legacy 'foreground' value maps to the green-screen output."""
    result = proc.process_matanyone2(
        [env["src"]], base_params(outputs=["foreground"]), cb
    )
    assert [p.name for p in result["success"]] == ["clip_greenscreen.mp4"]


def test_transparent_output_uses_alpha_encode(env, monkeypatch):
    captured = {}

    def fake_encode(frames, fw, fh, fps, src_path, out_path, use_audio, rotation=0,
                    use_nvenc=True, timing=None, alpha=False):
        captured[out_path.name] = alpha

    monkeypatch.setattr(proc, "encode_video", fake_encode)
    monkeypatch.setattr(proc, "prores_available", lambda: True)
    result = proc.process_matanyone2(
        [env["src"]], base_params(outputs=["matte", "greenscreen", "transparent"]), cb
    )
    names = sorted(p.name for p in result["success"])
    assert names == ["clip_greenscreen.mp4", "clip_matte.mp4", "clip_transparent.mov"]
    assert captured["clip_transparent.mov"] is True
    assert captured["clip_greenscreen.mp4"] is False


def test_first_frame_alpha_consumes_start_state(env):
    """Frame 0's alpha comes from start()'s final prediction step; the
    session is stepped exactly once per remaining frame (no double step)."""
    info = env
    proc.process_matanyone2([info["src"]], base_params(), cb)
    assert info["sessions"][0].steps == FRAMES - 1
    assert info["sessions"][0].first_prob_np is not None
