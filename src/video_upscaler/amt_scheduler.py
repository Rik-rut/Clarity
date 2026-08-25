"""Batched recursive scheduling for AMT frame interpolation."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from video_upscaler.amt_backend import AMTBackend


class AMTFrameScheduler:
    """Prepare one frame window and recursively schedule adjacent AMT pairs."""

    def interpolate_window(
        self,
        frames: list[np.ndarray],
        niters: int,
        backend: AMTBackend,
        batch_size: int,
    ) -> Iterator[np.ndarray]:
        if niters < 0:
            raise ValueError("niters must be non-negative")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not frames:
            return

        current = backend.prepare_frames(frames)

        for _ in range(niters):
            predictions: list[torch.Tensor] = []
            for start in range(0, len(current) - 1, batch_size):
                pair_a_items = current[start : min(start + batch_size, len(current) - 1)]
                pair_b_items = current[start + 1 : start + 1 + len(pair_a_items)]
                pair_a = torch.stack(pair_a_items)
                pair_b = torch.stack(pair_b_items)
                batch_output = backend.transfer_batch_to_host(
                    backend.infer_batch(pair_a, pair_b)
                )
                predictions.extend(batch_output)

            interleaved: list[torch.Tensor] = []
            for index, prediction in enumerate(predictions):
                interleaved.extend((current[index], prediction))
            interleaved.append(current[-1])
            current = interleaved

        yield from backend.finalize_frames(current)
