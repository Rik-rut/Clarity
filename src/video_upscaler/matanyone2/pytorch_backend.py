"""PyTorch/CUDA matting session over the vendored InferenceCore.

Mirrors upstream ``inference_matanyone2.py`` sequencing exactly:
first-frame mask encoding, then (1 + warmup) repeated first-frame prediction
steps whose outputs are consumed internally, then plain per-frame steps.
FP16 autocast applies only on CUDA sessions; CPU always runs FP32.

An internal resolution cap (shortest side) mirrors the upstream GUI's
``max_internal_size``: frames are downscaled before the core sees them and
probabilities are upscaled back to native size, keeping VRAM bounded so
1080p+ inputs do not spill into shared memory (which slows inference
50x on Windows/WDDM). Outputs stay at native resolution.
"""

from __future__ import annotations

import contextlib
import os
from typing import Callable, Iterator

import numpy as np

from video_upscaler.matanyone2.backend import BackendSelection

# Shortest-side cap for the internal inference resolution. 720 fits comfortably
# in ~4-6 GB GPUs; override with CLARITY_MA2_INTERNAL_SIZE (0 = uncapped).
DEFAULT_INTERNAL_SIZE = 720


def resolve_internal_size(device: str) -> int:
    """Internal shortest-side cap for this session (0 = no cap)."""
    env = os.environ.get("CLARITY_MA2_INTERNAL_SIZE")
    if env is not None and env.strip() != "":
        try:
            return max(0, int(env))
        except ValueError:
            return DEFAULT_INTERNAL_SIZE
    return DEFAULT_INTERNAL_SIZE if device == "cuda" else 0


def frame_to_tensor(img_rgb_u8: np.ndarray):
    """HWC uint8 RGB frame -> CHW float32 tensor in [0, 1]."""
    import torch

    contiguous = np.ascontiguousarray(img_rgb_u8).copy()
    return (
        torch.from_numpy(contiguous).permute(2, 0, 1).float().div_(255.0)
    )


class PyTorchSession:
    """Sequence-bound session; the shared model persists via model.get_model()."""

    def __init__(
        self,
        selection: BackendSelection,
        model_loader: Callable[[str], object] | None = None,
        core_factory: Callable[[object], object] | None = None,
        internal_size: int | None = None,
    ) -> None:
        self.selection = selection
        self.device = selection.device
        self._autocast_enabled = (
            selection.precision == "fp16" and selection.device == "cuda"
        )
        self._model_loader = model_loader or self._default_model_loader
        self._core_factory = core_factory or self._default_core_factory
        self._core = None
        self._internal_size = (
            resolve_internal_size(selection.device)
            if internal_size is None
            else max(0, int(internal_size))
        )
        # Alpha for output frame 0: probability produced by start()'s final
        # prediction step (upstream emits ti==0 from that same step).
        self.first_prob_np: np.ndarray | None = None

    @staticmethod
    def _default_model_loader(device: str):
        from video_upscaler.matanyone2.model import get_model

        return get_model(device)

    @staticmethod
    def _default_core_factory(model: object):
        from video_upscaler.matanyone2.vendor.matanyone2.inference.inference_core import (
            InferenceCore,
        )

        return InferenceCore(model, cfg=model.cfg)

    @contextlib.contextmanager
    def _autocast(self) -> Iterator[None]:
        if self._autocast_enabled:
            import torch

            with torch.autocast("cuda", dtype=torch.float16):
                yield
        else:
            yield

    def _work_tensor(self, tensor_chw):
        """Downscale a CHW tensor to the internal work resolution if capped."""
        if self._internal_size <= 0:
            return tensor_chw
        import torch.nn.functional as F

        _, h, w = tensor_chw.shape
        min_side = min(h, w)
        if min_side <= self._internal_size:
            return tensor_chw
        new_h = int(round(h / min_side * self._internal_size))
        new_w = int(round(w / min_side * self._internal_size))
        return F.interpolate(
            tensor_chw.unsqueeze(0), size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        )[0]

    def _restore_prob(self, prob, native_hw):
        """Upscale an HW probability tensor back to native (h, w)."""
        h, w = prob.shape[-2], prob.shape[-1]
        if (h, w) == tuple(native_hw):
            return prob
        import torch.nn.functional as F

        return F.interpolate(
            prob.unsqueeze(0).unsqueeze(0).float(), size=tuple(native_hw),
            mode="bilinear", align_corners=False,
        )[0, 0]

    def start(self, first_frame_chw, mask_hw, warmup: int = 10) -> None:
        import torch

        model = self._model_loader(self.device)
        self._core = self._core_factory(model)
        native_hw = tuple(first_frame_chw.shape[-2:])
        frame = self._work_tensor(first_frame_chw).to(self.device).float()
        mask = self._work_tensor(mask_hw.unsqueeze(0))[0].to(self.device).float()

        with self._autocast(), torch.inference_mode():
            self._core.step(frame, mask, objects=[1])
            prob = self._core.step(frame, first_frame_pred=True)
            for _ in range(max(0, int(warmup))):
                prob = self._core.step(frame, first_frame_pred=True)
            first_prob = self._core.output_prob_to_mask(prob)
            self.first_prob_np = (
                self._restore_prob(first_prob, native_hw)
                .detach().float().cpu().numpy()
            )

    def step(self, frame_chw) -> np.ndarray:
        import torch

        native_hw = tuple(frame_chw.shape[-2:])
        work = self._work_tensor(frame_chw).to(self.device).float()
        with self._autocast(), torch.inference_mode():
            prob = self._core.step(work)
            mask_t = self._core.output_prob_to_mask(prob)
            mask_t = self._restore_prob(mask_t, native_hw)
        return mask_t.detach().float().cpu().numpy()

    def close(self) -> None:
        self._core = None


def build_session(
    selection: BackendSelection,
    model_loader: Callable[[str], object] | None = None,
    core_factory: Callable[[object], object] | None = None,
):
    """Construct the session implementation for a resolved selection."""
    if selection.name != "pytorch":  # pragma: no cover - guarded by selector
        raise NotImplementedError(f"No session builder for backend '{selection.name}'")
    return PyTorchSession(selection, model_loader=model_loader, core_factory=core_factory)
