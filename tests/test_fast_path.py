"""GPU-free tests for triton wrapper injection + fallback."""

from __future__ import annotations

import torch
import torch.nn as nn

from video_upscaler import fast_path


def test_tritonize_leaves_non_eligible_modules_untouched() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),  # CPU -> not eligible at init
        nn.Conv2d(3, 8, kernel_size=5, padding=2),
        nn.PixelShuffle(2),
    )
    result = fast_path.tritonize(model)
    assert result is model
    assert all(not isinstance(m, fast_path._TritonConv3x3) for m in result.modules())


def test_wrapper_falls_back_when_kernel_unavailable(
    monkeypatch,
) -> None:
    import numpy as np

    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    wrapper = fast_path._TritonConv3x3(conv)
    monkeypatch.setattr(
        "video_upscaler.fast_path._eligible", lambda x, m: True
    )
    monkeypatch.setattr(
        "video_upscaler.triton_kernels.conv3x3", lambda x, w, b: None
    )
    x = torch.randn(1, 3, 16, 16)
    expected = conv(x)
    out = wrapper(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_pixel_shuffle_wrapper_falls_back(monkeypatch) -> None:
    shuffle = nn.PixelShuffle(2)
    wrapper = fast_path._TritonPixelShuffle(shuffle)
    monkeypatch.setattr(
        "video_upscaler.fast_path._eligible", lambda x, m: True
    )
    monkeypatch.setattr(
        "video_upscaler.triton_kernels.pixel_shuffle", lambda x, r: None
    )
    x = torch.randn(1, 8, 4, 4)
    expected = shuffle(x)
    out = wrapper(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_wrapper_uses_kernel_when_available(monkeypatch) -> None:
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    wrapper = fast_path._TritonConv3x3(conv)
    monkeypatch.setattr(
        "video_upscaler.fast_path._eligible", lambda x, m: True
    )
    fake_result = torch.zeros(1, 4, 16, 16)
    calls = {}

    def _fake_kernel(x, w, b):
        calls["called"] = True
        return fake_result

    monkeypatch.setattr(
        "video_upscaler.triton_kernels.conv3x3", _fake_kernel
    )
    x = torch.randn(1, 3, 16, 16)
    out = wrapper(x)
    assert calls["called"]
    assert torch.allclose(out, fake_result, atol=1e-6)
