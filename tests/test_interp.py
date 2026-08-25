"""Tests for AMT frame-interpolation (slow motion) support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import video_upscaler.config as config
from video_upscaler import models as m
from video_upscaler.processor import (
    slowed_output_tag,
    unique_interp_output_path,
)


def test_interp_factor_mapping():
    assert m.INTERP_FACTORS == {2: 1, 4: 2, 8: 3}
    assert m.niters_for_factor(2) == 1
    assert m.niters_for_factor(4) == 2
    assert m.niters_for_factor(8) == 3


def test_interp_model_manifest():
    assert set(m.INTERP_MODELS) == {"AMT-S", "AMT-L", "AMT-G"}
    assert m.default_interp_model() == "AMT-S"
    for key in m.INTERP_MODELS:
        assert m.ckpt_for_interp_model(key).endswith(".pth")
        assert m.description_for_interp_model(key)


def test_interp_output_tag_and_unique(tmp_path: Path):
    config.OUTPUT_DIR = tmp_path
    tag = slowed_output_tag(4, "AMT-L")
    assert tag == "slowed4x_amt-l"
    out = unique_interp_output_path(tmp_path, "clip.mp4", 4, "AMT-L")
    assert out.name == "clip_slowed4x_amt-l.mp4"
    out.touch()
    out2 = unique_interp_output_path(tmp_path, "clip.mp4", 4, "AMT-L")
    assert out2.name == "clip_slowed4x_amt-l_1.mp4"


def test_check_amt_missing_message(tmp_path: Path):
    from video_upscaler.interp import check_amt

    config.MODELS_DIR = tmp_path
    msg = check_amt("AMT-S")
    assert msg is not None
    assert "amt-s.pth" in msg
    (tmp_path / "amt-s.pth").touch()
    assert check_amt("AMT-S") is None


def test_transfer_timing_is_opt_in():
    import video_upscaler.interp as interp

    class FailingCuda:
        def synchronize(self, device):
            raise AssertionError("ordinary AMT work must not synchronize for timing")

    class FakeTorch:
        cuda = FailingCuda()

    timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
    assert interp._timed_transfer(
        lambda: "value",
        "cuda",
        FakeTorch(),
        timings,
        "h2d_time_s",
        enabled=False,
    ) == "value"
    assert timings == {"h2d_time_s": 0.0, "d2h_time_s": 0.0}


def test_engine_infer_batch_uses_inference_mode_and_loaded_model():
    pytest.importorskip("torch")
    import torch
    from video_upscaler.interp import AMTInterpEngine

    observed = {}

    class FakeModel:
        def __call__(self, frame_a, frame_b, embt, *, scale_factor, eval):
            observed["grad_enabled"] = torch.is_grad_enabled()
            observed["shape"] = frame_a.shape
            observed["scale_factor"] = scale_factor
            observed["eval"] = eval
            return {"imgt_pred": (frame_a + frame_b) / 2}

    engine = object.__new__(AMTInterpEngine)
    engine._torch = torch
    engine.device = "cpu"
    engine.model = FakeModel()
    engine._transfer_timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
    engine._timing_enabled = False
    engine._batch_scale = 0.75

    result = engine.infer_batch(
        torch.zeros((2, 3, 16, 16)), torch.ones((2, 3, 16, 16))
    )

    assert result.shape == (2, 3, 16, 16)
    assert observed == {
        "grad_enabled": False,
        "shape": torch.Size([2, 3, 16, 16]),
        "scale_factor": 0.75,
        "eval": True,
    }


def test_engine_batch_path_owns_adaptive_prepare_and_rgb8_finalize(monkeypatch):
    pytest.importorskip("torch")
    import torch
    from video_upscaler.interp import AMTInterpEngine

    class FakePadder:
        def pad(self, tensor):
            return torch.nn.functional.pad(tensor, [1, 1, 1, 1], mode="replicate")

        def unpad(self, *tensors):
            return [tensor[..., 1:-1, 1:-1] for tensor in tensors]

    engine = object.__new__(AMTInterpEngine)
    engine._torch = torch
    engine.device = "cpu"
    engine._transfer_timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
    engine._timing_enabled = False
    padder = FakePadder()
    monkeypatch.setattr(engine, "_prepare", lambda height, width: (0.5, padder))

    source = np.full((2, 3, 3), 64, dtype=np.uint8)
    prepared = engine.prepare_frames([source])
    assert prepared[0].shape == (3, 4, 5)
    assert engine._batch_scale == 0.5

    output = engine.finalize_frames([prepared[0].squeeze(0)])
    assert len(output) == 1
    assert output[0].shape == (2, 3, 3)
    assert output[0].dtype == np.uint8
    assert int(output[0][0, 0, 0]) == 64


def test_engine_batch_host_transfer_uses_timed_boundary(monkeypatch):
    pytest.importorskip("torch")
    import torch
    import video_upscaler.interp as interp
    from video_upscaler.interp import AMTInterpEngine

    engine = object.__new__(AMTInterpEngine)
    engine._torch = torch
    engine.device = "cpu"
    engine._transfer_timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
    engine._timing_enabled = True
    calls = []

    def fake_transfer(operation, device, torch_module, timings, key, enabled=False):
        calls.append((key, enabled))
        timings[key] += 0.5
        return operation()

    monkeypatch.setattr(interp, "_timed_transfer", fake_transfer)
    result = engine.transfer_batch_to_host(torch.ones((2, 3, 4, 4)))

    assert result.device.type == "cpu"
    assert calls == [("d2h_time_s", True)]
    assert engine._transfer_timings["d2h_time_s"] == 0.5


def test_final_tensor_conversion_is_included_in_d2h_timing(monkeypatch):
    pytest.importorskip("torch")
    import torch
    import video_upscaler.interp as interp

    calls = []

    def fake_transfer(operation, device, torch_module, timings, key, enabled=False):
        calls.append(key)
        timings[key] += 0.25
        return operation()

    monkeypatch.setattr(interp, "_timed_transfer", fake_transfer)
    timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
    result = interp._tensor2img_timed(
        torch.zeros((1, 3, 2, 2)), "cpu", torch, timings, enabled=True
    )

    assert result.shape == (2, 2, 3)
    assert calls == ["d2h_time_s"]
    assert timings["d2h_time_s"] == 0.25


def test_engine_missing_ckpt_raises():
    pytest.importorskip("torch")
    from video_upscaler.interp import AMTInterpEngine, check_amt

    tmp = Path.home() / ".cache" / "clarity_test_no_ckpt"
    tmp.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR = tmp
    assert check_amt("AMT-S") is not None
    with pytest.raises(FileNotFoundError):
        AMTInterpEngine("AMT-S")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_interp_frame_count_and_segmentation():
    pytest.importorskip("torch")
    import tempfile

    from video_upscaler.amt.networks.amt_s import Model as AMT_S
    from video_upscaler.interp import AMTInterpEngine, _MODEL_PARAMS

    tmp = Path(tempfile.mkdtemp())
    config.MODELS_DIR = tmp
    net = AMT_S(**_MODEL_PARAMS["AMT-S"])
    torch = __import__("torch")
    torch.save({"state_dict": net.state_dict()}, tmp / "amt-s.pth")

    engine = AMTInterpEngine("AMT-S")
    # Smooth gradient frames (realistic-ish input; avoids NaN on random noise).
    g = np.linspace(0, 255, 64, dtype=np.uint8)
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    base[..., 0] = g[None, :]
    base[..., 1] = g[:, None]
    frames = [base.copy() for _ in range(5)]

    full = list(engine.interpolate(frames, niters=1))
    # 2x of 5 frames -> (2^1)*(5-1)+1 = 9
    assert len(full) == 9
    assert all(f.shape == (64, 64, 3) and f.dtype == "uint8" for f in full)

    config.AMT_SEGMENT_FRAMES = 2
    segmented = list(engine.interpolate(frames, niters=1))
    assert len(segmented) == 9
    assert all(f.shape == (64, 64, 3) for f in segmented)


class _FakeInterpEngine:
    """Mimics AMT's per-window output count without needing torch/ffmpeg.

    For a window of M source frames, emits M sources interleaved with
    ``2**niters - 1`` synthetic frames per gap, i.e. ``2**niters * M -
    (2**niters - 1)`` frames — the same count the real AMT engine yields, so
    the windowing/dedup math can be exercised in isolation.
    """

    def __init__(self, niters: int) -> None:
        import itertools

        self.niters = niters
        self._mid = itertools.count()

    def interpolate(self, window, niters):
        t = 2 ** niters
        out = []
        for i, frame in enumerate(window):
            out.append(frame)
            if i < len(window) - 1:
                for _ in range(t - 1):
                    out.append(("mid", next(self._mid)))
        return out


def _run_stream(n_source: int, seg: int, niters: int):
    # Each source frame is a tiny unique-id array.
    def raw_iter():
        for i in range(n_source):
            arr = np.zeros((2, 2, 3), dtype=np.uint8)
            arr[0, 0, 0] = i
            yield arr.tobytes()

    from video_upscaler.processor import _interp_window_stream

    engine = _FakeInterpEngine(niters)
    return list(_interp_window_stream(raw_iter(), engine, niters, seg, (2, 2)))


def test_window_stream_total_count_no_dup():
    t = 1  # 2x
    for n_source in (3, 5, 17, 120, 121, 200):
        for seg in (2, 60, 120):
            frames = _run_stream(n_source, seg, t)
            expected = (2 ** t) * n_source - (2 ** t - 1)
            assert len(frames) == expected, (n_source, seg, len(frames), expected)
            # Every source frame id appears exactly once (no dup, no loss).
            ids = sorted(int(f[0, 0, 0]) for f in frames if not isinstance(f, tuple))
            assert ids == list(range(n_source)), (n_source, seg, ids)


def test_window_stream_recovery_on_exact_boundary():
    # 120 source frames, seg=120 -> ends exactly on a full-window boundary,
    # so the dropped boundary must be re-emitted via the recovery window.
    frames = _run_stream(120, 120, 1)
    assert len(frames) == 2 * 120 - 1
    ids = sorted(int(f[0, 0, 0]) for f in frames if not isinstance(f, tuple))
    assert ids == list(range(120))
