"""Backend contract for batched AMT pair inference."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class AMTBackend(Protocol):
    name: str
    precision: str
    device: str

    def prepare_frames(self, frames: list[np.ndarray]) -> list[torch.Tensor]:
        """Normalize and pad source frames for this backend's inference scale."""
        ...

    def infer_batch(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        """Infer normalized padded [B, 3, H, W] pairs and return [B, 3, H, W]."""
        ...

    def transfer_batch_to_host(self, batch_output: torch.Tensor) -> torch.Tensor:
        """Move one output batch to host memory through the backend timing boundary."""
        ...

    def finalize_frames(self, frames: list[torch.Tensor]) -> list[np.ndarray]:
        """Unpad backend outputs and convert them to RGB8 host frames."""
        ...

    def warmup(self, shape: tuple[int, int], batch_size: int) -> None: ...

    def close(self) -> None: ...
