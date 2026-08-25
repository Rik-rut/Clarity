"""Task 6 quality tests: frame counting, ordering, FPS/duration, passthrough.

These tests exercise the AMT pipeline through fake backends and small clips.
They do not require a GPU, a real AMT checkpoint, or a TensorRT engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from video_upscaler.amt_benchmark import output_frame_count
from video_upscaler.amt_scheduler import AMTFrameScheduler
from video_upscaler.amt.utils.utils import InputPadder, img2tensor


class _FakeBackend:
    name = "fake"
    precision = "fp32"
    device = "cpu"

    def __init__(self) -> None:
        self.window_sizes: list[int] = []
        self._padder: InputPadder | None = None

    def prepare_frames(self, frames: list[np.ndarray]) -> list[torch.Tensor]:
        self.window_sizes.append(len(frames))
        self._padder = InputPadder(frames[0].shape[:2], 16)
        return [self._padder.pad(img2tensor(frame)).squeeze(0) for frame in frames]

    def infer_batch(
        self, frame_a: torch.Tensor, frame_b: torch.Tensor
    ) -> torch.Tensor:
        return (frame_a + frame_b) / 2

    def transfer_batch_to_host(self, batch_output: torch.Tensor) -> torch.Tensor:
        return batch_output.detach().cpu()

    def finalize_frames(self, frames: list[torch.Tensor]) -> list[np.ndarray]:
        assert self._padder is not None
        return [
            self._padder.unpad(frame.unsqueeze(0))
            .squeeze(0)
            .permute(1, 2, 0)
            .mul(255.0)
            .clamp(0, 255)
            .numpy()
            .astype(np.uint8)
            for frame in frames
        ]

    def warmup(self, shape: tuple[int, int], batch_size: int) -> None:
        pass

    def close(self) -> None:
        pass


def _source_frames(count: int, step: int = 40) -> list[np.ndarray]:
    return [np.full((3, 5, 3), index * step, dtype=np.uint8) for index in range(count)]


@pytest.mark.parametrize(
    ("source_frames", "niters", "expected"),
    [(3, 1, 5), (3, 2, 9), (3, 3, 17), (5, 1, 9), (5, 2, 17), (5, 3, 33)],
)
def test_exact_frame_count_for_fake_backend_across_factors(
    source_frames: int, niters: int, expected: int
) -> None:
    factor = 2 ** niters
    output = list(
        AMTFrameScheduler().interpolate_window(
            _source_frames(source_frames), niters, _FakeBackend(), batch_size=2
        )
    )
    assert len(output) == expected
    assert output_frame_count(source_frames, factor) == expected


@pytest.mark.parametrize("niters", [1, 2, 3])
def test_exact_frame_order_for_fake_backend_across_factors(niters: int) -> None:
    factor = 2 ** niters
    backend = _FakeBackend()
    output = list(
        AMTFrameScheduler().interpolate_window(
            _source_frames(4), niters, backend, batch_size=2
        )
    )

    expected_values = np.linspace(0, 3 * 40, factor * 3 + 1)
    assert [int(frame[0, 0, 0]) for frame in output] == list(expected_values)
    assert len(output) == factor * 3 + 1


def test_output_fps_preserved_and_duration_scales_with_factor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import processor

    source_frames = _source_frames(9, step=20)
    captured: dict = {}

    class FakeBackend(_FakeBackend):
        pass

    class FakeFactory:
        selection = type("S", (), {"batch_size": 2, "fallback_reason": None, "backend": "pytorch"})()

        def build(self, shape=None):
            return FakeBackend()

        def close(self):
            pass

    monkeypatch.setattr(processor, "_make_amt_backend_factory", lambda model, selection=None: FakeFactory())
    monkeypatch.setattr(processor, "AMTFrameScheduler", AMTFrameScheduler, raising=False)
    monkeypatch.setattr(processor, "probe", lambda path: {
        "width": 5, "height": 3, "fps": 30.0, "duration": 9 / 30.0,
        "codec_name": "h264", "has_audio": False, "rotation": 0,
    })
    monkeypatch.setattr(processor, "decode_frames", lambda path: iter(
        frame.tobytes() for frame in source_frames
    ))
    monkeypatch.setattr(processor.config, "OUTPUT_DIR", tmp_path)

    def fake_encode(frames, frame_w, frame_h, fps, src_path, out_path, use_audio, rotation=0, use_nvenc=True):
        captured["fps"] = fps
        captured["count"] = sum(1 for _ in frames)
        captured["size"] = (frame_w, frame_h)

    monkeypatch.setattr(processor, "encode_video", fake_encode)

    result = processor.process_interpolate(
        [tmp_path / "clip.mp4"], "AMT-S", 2, lambda *args: None
    )

    assert result["failed"] == []
    assert captured["fps"] == 30.0
    assert captured["count"] == output_frame_count(9, 2)
    assert captured["size"] == (5, 3)
    assert captured["count"] / captured["fps"] == 17 / 30.0


@pytest.mark.parametrize(
    ("has_audio", "rotation"),
    [(True, 90), (True, 0), (False, 180), (False, 0)],
)
def test_audio_and_rotation_passthrough_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    has_audio: bool,
    rotation: int,
) -> None:
    from video_upscaler import processor

    captured: dict = {}

    class FakeBackend(_FakeBackend):
        pass

    class FakeFactory:
        selection = type("S", (), {"batch_size": 2, "fallback_reason": None, "backend": "pytorch"})()

        def build(self, shape=None):
            return FakeBackend()

        def close(self):
            pass

    monkeypatch.setattr(processor, "_make_amt_backend_factory", lambda model, selection=None: FakeFactory())
    monkeypatch.setattr(processor, "probe", lambda path: {
        "width": 5, "height": 3, "fps": 30.0, "duration": 0.1,
        "codec_name": "h264", "has_audio": has_audio, "rotation": rotation,
    })
    monkeypatch.setattr(processor, "decode_frames", lambda path: iter(
        frame.tobytes() for frame in _source_frames(3)
    ))
    monkeypatch.setattr(processor.config, "OUTPUT_DIR", tmp_path)

    def fake_encode(frames, frame_w, frame_h, fps, src_path, out_path, use_audio, rotation=0, use_nvenc=True):
        captured["use_audio"] = use_audio
        captured["rotation"] = rotation

    monkeypatch.setattr(processor, "encode_video", fake_encode)

    result = processor.process_interpolate(
        [tmp_path / "clip.mp4"], "AMT-S", 2, lambda *args: None
    )

    assert result["failed"] == []
    assert captured["use_audio"] is has_audio
    assert captured["rotation"] == rotation


def test_final_output_frames_are_uint8_rgb_in_range_shaped_like_source() -> None:
    source = _source_frames(3)
    output = list(
        AMTFrameScheduler().interpolate_window(source, 2, _FakeBackend(), batch_size=2)
    )

    for frame in output:
        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.shape[:2] == source[0].shape[:2]
        assert frame.min() >= 0
        assert frame.max() <= 255


def test_windowed_stream_never_holds_more_than_one_window_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_upscaler.processor import _interp_window_stream

    source = _source_frames(37, step=5)
    seg = 10
    pulled: list[int] = []

    class CountingStream:
        def __init__(self, frames: list[np.ndarray]) -> None:
            self._frames = list(frames)

        def __iter__(self):
            for frame in self._frames:
                pulled.append(1)
                yield frame.tobytes()

    backend = _FakeBackend()
    stream = iter(CountingStream(source))
    output = list(
        _interp_window_stream(
            stream,
            backend,
            niters=1,
            seg=seg,
            shape=source[0].shape[:2],
            scheduler=AMTFrameScheduler(),
            batch_size=2,
        )
    )

    assert len(pulled) == 37
    assert max(backend.window_sizes) <= seg + 1
    assert len(output) == output_frame_count(37, 2)
    assert pulled[:seg] == [1] * seg
