"""Tests for the batched AMT pair scheduler."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from video_upscaler.amt_scheduler import AMTFrameScheduler
from video_upscaler.amt.utils.utils import InputPadder, img2tensor


class _FakeBackend:
    name = "fake"
    precision = "fp32"
    device = "cpu"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.host_transfer_calls = 0
        self.finalize_calls = 0
        self._padder: InputPadder | None = None

    def prepare_frames(self, frames: list[np.ndarray]) -> list[torch.Tensor]:
        self._padder = InputPadder(frames[0].shape[:2], 16)
        return [self._padder.pad(img2tensor(frame)).squeeze(0) for frame in frames]

    def infer_batch(
        self, frame_a: torch.Tensor, frame_b: torch.Tensor
    ) -> torch.Tensor:
        self.batch_sizes.append(frame_a.shape[0])
        self.inputs.append((frame_a.clone(), frame_b.clone()))
        # Encode the pair in the output so every generated frame has a stable,
        # inspectable identity while retaining the contract's tensor shape.
        return (frame_a + frame_b) / 2

    def transfer_batch_to_host(self, batch_output: torch.Tensor) -> torch.Tensor:
        self.host_transfer_calls += 1
        return batch_output.detach().cpu()

    def finalize_frames(self, frames: list[torch.Tensor]) -> list[np.ndarray]:
        self.finalize_calls += 1
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


def _source_frames(count: int) -> list[np.ndarray]:
    return [np.full((3, 5, 3), index * 40, dtype=np.uint8) for index in range(count)]


def test_scheduler_batches_adjacent_pairs_and_preserves_source_order() -> None:
    backend = _FakeBackend()

    output = list(
        AMTFrameScheduler().interpolate_window(
            _source_frames(3), niters=1, backend=backend, batch_size=2
        )
    )

    assert backend.batch_sizes == [2]
    assert backend.host_transfer_calls == 1
    assert backend.finalize_calls == 1
    assert len(output) == 5
    assert [int(frame[0, 0, 0]) for frame in output] == [0, 20, 40, 60, 80]
    assert all(frame.shape == (3, 5, 3) and frame.dtype == np.uint8 for frame in output)


def test_scheduler_recursively_orders_four_times_output_without_losing_sources() -> None:
    backend = _FakeBackend()

    output = list(
        AMTFrameScheduler().interpolate_window(
            _source_frames(3), niters=2, backend=backend, batch_size=1
        )
    )

    assert backend.batch_sizes == [1, 1, 1, 1, 1, 1]
    assert len(output) == 9
    assert [int(frame[0, 0, 0]) for frame in output] == [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
    ]


def test_scheduler_supports_eight_times_output_and_never_exceeds_batch_size() -> None:
    backend = _FakeBackend()

    output = list(
        AMTFrameScheduler().interpolate_window(
            _source_frames(2), niters=3, backend=backend, batch_size=3
        )
    )

    assert len(output) == 9
    assert backend.batch_sizes == [1, 2, 3, 1]
    assert backend.host_transfer_calls == 4
    assert max(backend.batch_sizes) <= 3
    assert [int(frame[0, 0, 0]) for frame in output] == [0, 5, 10, 15, 20, 25, 30, 35, 40]


def test_scheduler_eight_times_multiple_gaps_preserves_full_ordering() -> None:
    # 5 source frames at 8x covers multiple adjacent gaps end-to-end.
    backend = _FakeBackend()

    output = list(
        AMTFrameScheduler().interpolate_window(
            _source_frames(5), niters=3, backend=backend, batch_size=4
        )
    )

    assert len(output) == 33
    assert max(backend.batch_sizes) <= 4
    assert [int(frame[0, 0, 0]) for frame in output] == list(
        range(0, 160 + 1, 5)
    )


def test_scheduler_normalizes_and_pads_each_source_once() -> None:
    backend = _FakeBackend()

    list(
        AMTFrameScheduler().interpolate_window(
            _source_frames(2), niters=1, backend=backend, batch_size=4
        )
    )

    assert len(backend.inputs) == 1
    frame_a, frame_b = backend.inputs[0]
    assert frame_a.shape == (1, 3, 16, 16)
    assert frame_b.shape == (1, 3, 16, 16)
    assert frame_a.dtype == torch.float32
    assert frame_a[0, :, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert frame_b[0, :, 0, 0].tolist() == pytest.approx([40 / 255] * 3)
