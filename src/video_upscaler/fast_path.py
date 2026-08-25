"""Triton fast-path injection for upscaler networks (NVIDIA CUDA fp16).

``tritonize(model)`` swaps eligible conv3x3 / pixel-shuffle modules for
thin wrappers that try the Triton kernel and fall back to the original
torch op on any miss (wrong device/dtype/shape, kernel disabled by the
speed gate, or runtime failure). Applied to both Real-CUGAN and RRDBNet;
the vendored upcunet_v3.py stays unmodified.
"""

from __future__ import annotations

import torch.nn as nn


def _eligible(x, module) -> bool:
    """True when a conv3x3 wrapper can try the kernel for this input."""
    if module.__class__.__name__ == "Conv2d":
        return (
            x.is_cuda
            and x.dtype.is_floating_point
            and module.kernel_size == (3, 3)
            and module.stride == (1, 1)
            and module.padding == (1, 1)
            and module.dilation == (1, 1)
            and module.groups == 1
        )
    return x.is_cuda and x.dtype.is_floating_point


class _TritonConv3x3(nn.Module):
    """Conv2d wrapper: Triton kernel with automatic torch fallback."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x):
        if not _eligible(x, self.conv):
            return self.conv(x)
        from video_upscaler.triton_kernels import conv3x3

        result = conv3x3(x, self.conv.weight, self.conv.bias)
        if result is None:
            return self.conv(x)
        return result


class _TritonPixelShuffle(nn.Module):
    """PixelShuffle wrapper: Triton kernel with automatic torch fallback."""

    def __init__(self, shuffle: nn.PixelShuffle) -> None:
        super().__init__()
        self.shuffle = shuffle

    def forward(self, x):
        if not _eligible(x, self.shuffle):
            return self.shuffle(x)
        from video_upscaler.triton_kernels import pixel_shuffle

        result = pixel_shuffle(x, self.shuffle.upscale_factor)
        if result is None:
            return self.shuffle(x)
        return result


def tritonize(model):
    """Wrap eligible conv3x3 / pixel-shuffle modules in Triton-aware wrappers."""
    from video_upscaler.triton_kernels import triton_available

    if not triton_available():
        return model
    params = next(model.parameters(), None)
    if params is None or not params.is_cuda:
        return model
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "Conv2d":
            parent = model
            target = module
            if "." in name:
                parent_name, attr = name.rsplit(".", 1)
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            else:
                attr = name
            if hasattr(parent, attr):
                setattr(parent, attr, _TritonConv3x3(module))
        elif module.__class__.__name__ == "PixelShuffle":
            parent = model
            if "." in name:
                parent_name, attr = name.rsplit(".", 1)
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            else:
                attr = name
            if hasattr(parent, attr):
                setattr(parent, attr, _TritonPixelShuffle(module))
    return model
