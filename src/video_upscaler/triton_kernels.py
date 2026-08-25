"""Custom Triton kernels for the hot upscaler ops (NVIDIA CUDA only).

Conv3x3 (implicit GEMM, fp16) and pixel-shuffle replace cuDNN/torch ops
when they provably win. Auto-Broll pattern: per-shape benchmark results
are persisted in a device-fingerprint-keyed JSON cache so autotuning
never repeats across runs, and any kernel that fails validation or the
1.02x speed gate is permanently disabled for that shape.

Every public entry point returns None instead of raising: callers simply
fall back to torch. Set CLARITY_TRITON=0 to disable entirely.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

from video_upscaler.config import TRITON_ENABLED

_AUTOTUNE_CACHE_VERSION = 1
_MIN_SPEEDUP = 1.02
_SESSION: dict[str, dict] = {}
_DISK: dict[str, dict] = {}

_CANDIDATES = [
    {"block_m": 64, "block_n": 32, "block_c": 16},
    {"block_m": 64, "block_n": 64, "block_c": 32},
    {"block_m": 128, "block_n": 32, "block_c": 16},
    {"block_m": 128, "block_n": 64, "block_c": 32},
]


def _cache_path() -> Path:
    default = str(Path.home() / ".triton" / "autotune" / "clarity_autotune_cache.json")
    return Path(os.environ.get("CLARITY_TRITON_CACHE", default)).expanduser()


def _device_fingerprint() -> str:
    import torch

    props = torch.cuda.get_device_properties(0)
    try:
        import triton

        triton_ver = getattr(triton, "__version__", "0.0")
    except ImportError:
        triton_ver = "none"
    return (
        f"{props.name}|cc={props.major}.{props.minor}|sm={props.multi_processor_count}|"
        f"torch={torch.__version__}|triton={triton_ver}|clarity_cache_v={_AUTOTUNE_CACHE_VERSION}"
    )


def _load_cache() -> None:
    path = _cache_path()
    if not path.is_file():
        return
    try:
        _DISK.update(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        pass


def _save_cache() -> None:
    if not _DISK:
        return
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps({key: cfg for key, cfg in _DISK.items()}, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError:
        pass


atexit.register(_save_cache)


def _cache_key(kind: str, shape_key: str) -> str:
    return f"{_device_fingerprint()}|{kind}|{shape_key}"


def _set_cached(kind: str, shape_key: str, cfg: dict) -> None:
    key = f"{kind}|{shape_key}"  # tests use fingerprint-free keys
    _SESSION[key] = cfg
    _DISK[key] = cfg
    _save_cache()


def _get_cached_raw(kind: str, shape_key: str) -> dict | None:
    """Return the raw cached entry (including {"disabled": True})."""
    key = f"{kind}|{shape_key}"
    if key in _SESSION:
        return _SESSION[key]
    if not _DISK:
        _load_cache()
    return _DISK.get(key)


def _get_cached(kind: str, shape_key: str) -> dict | None:
    cfg = _get_cached_raw(kind, shape_key)
    if cfg is None:
        return None
    return None if cfg.get("disabled") else cfg


def _select_config(kind, shape_key, build_and_bench, candidates=None):
    """Cached config for a shape, or autotune + persist on first sight.

    ``build_and_bench(cfg)`` returns (triton_ms, torch_ms, valid) or None
    on kernel failure. Returns None when the shape is disabled.
    """
    storage_key = _cache_key(kind, shape_key)  # includes device fingerprint
    cached = _get_cached_raw(kind, storage_key)
    if cached is not None:
        return None if cached.get("disabled") else cached

    best: tuple[float, dict] | None = None
    for cfg in (candidates if candidates is not None else _CANDIDATES):
        measured = build_and_bench(cfg)
        if measured is None:
            continue
        triton_ms, torch_ms, valid = measured
        if not valid:
            continue
        if triton_ms < torch_ms / _MIN_SPEEDUP:
            if best is None or triton_ms < best[0]:
                best = (triton_ms, cfg)

    chosen = best[1] if best else {"disabled": True}
    _set_cached(kind, storage_key, chosen)
    return None if chosen.get("disabled") else chosen


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

def triton_available() -> bool:
    """True when CUDA + triton are usable and not disabled by env."""
    if not TRITON_ENABLED:
        return False
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _conv_shape_key(x, weight) -> str:
    n, c, h, w = x.shape
    return f"{n}x{h}x{w}x{c}x{weight.shape[0]}"


def conv3x3(x, weight, bias=None):
    """Implicit-GEMM conv3x3 (stride 1, pad 1) fp16. None = use torch."""
    if not triton_available():
        return None
    import torch
    import triton
    import triton.language as tl

    if x.dtype != torch.float16 or x.device.type != "cuda":
        return None

    @triton.jit
    def _kernel(
        x_ptr, w_ptr, b_ptr, y_ptr,
        N, H, W, C, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(N * H * W, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_C)

        n = offs_m // (H * W)
        rem = offs_m % (H * W)
        h = rem // W
        w = rem % W

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ci in range(0, C, BLOCK_C):
            offs_c = ci + offs_k
            mask_c = offs_c < C
            for dh in tl.static_range(3):
                ih = h + (dh - 1)  # pad=1 window
                mask_h = (ih >= 0) & (ih < H)
                for dw in tl.static_range(3):
                    iw = w + (dw - 1)
                    mask_w = (iw >= 0) & (iw < W)
                    x_ptrs = (
                        x_ptr
                        + n[:, None] * (C * H * W)
                        + offs_c[None, :] * (H * W)
                        + ih[:, None] * W
                        + iw[:, None]
                    )
                    x_mask = (
                        (offs_m[:, None] < N * H * W)
                        & mask_c[None, :]
                        & mask_h[:, None]
                        & mask_w[:, None]
                    )
                    a = tl.load(x_ptrs, mask=x_mask, other=0.0)
                    w_ptrs = (
                        w_ptr
                        + offs_n[:, None] * (C * 9)
                        + offs_c[None, :] * 9
                        + (dh * 3 + dw)
                    )
                    w_mask = (offs_n[:, None] < K) & mask_c[None, :]
                    b = tl.load(w_ptrs, mask=w_mask, other=0.0)
                    acc += tl.dot(a, tl.trans(b))

        if b_ptr is not None:
            bias = tl.load(b_ptr + offs_n, mask=offs_n < K, other=0.0)
            acc += bias[None, :]
        acc = acc.to(tl.float16)

        y_ptrs = y_ptr + offs_m[:, None] * K + offs_n[None, :]
        y_mask = (offs_m[:, None] < N * H * W) & (offs_n[None, :] < K)
        tl.store(y_ptrs, acc, mask=y_mask)

    def _run(cfg):
        n, c, h, w = x.shape
        k = weight.shape[0]
        y = torch.empty((n, k, h, w), dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(n * h * w, cfg["block_m"]) * triton.cdiv(k, cfg["block_n"]),)
        _kernel[grid](
            x, weight, bias, y, n, h, w, c, k,
            BLOCK_M=cfg["block_m"], BLOCK_N=cfg["block_n"], BLOCK_C=cfg["block_c"],
        )
        return y

    def _build_and_bench(cfg):
        try:
            import triton.testing

            ref = torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1)
            out = _run(cfg)
            valid = bool(torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2))
            if not valid:
                return None
            triton_ms = triton.testing.do_bench(lambda: _run(cfg), warmup=10, rep=50)
            torch_ms = triton.testing.do_bench(
                lambda: torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1),
                warmup=10, rep=50,
            )
            return triton_ms, torch_ms, True
        except Exception:
            return None

    cfg = _select_config("conv3x3", _conv_shape_key(x, weight), _build_and_bench)
    if cfg is None:
        return None
    try:
        return _run(cfg)
    except Exception:
        return None


def pixel_shuffle(x, r):
    """Pixel-shuffle (NCHW) kernel. None = use torch."""
    if not triton_available():
        return None
    import torch
    import triton
    import triton.language as tl

    if x.dtype != torch.float16 or x.device.type != "cuda":
        return None

    # pixel-shuffle is an output-indexed gather (input index -> different
    # output index), so iterate over output elements:
    @triton.jit
    def _gather_kernel(
        x_ptr, y_ptr, numel, C, H, W, R: tl.constexpr, BLOCK: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        n = offs // (C * H * W)
        rem = offs % (C * H * W)
        out_c = rem // (H * W * R * R)
        rem2 = rem % (H * W * R * R)
        oh = rem2 // (W * R)
        ow = rem2 % (W * R)
        in_c = out_c * (R * R) + (oh % R) * R + (ow % R)
        ih = oh // R
        iw = ow // R
        x_ptrs = x_ptr + n * (C * H * W) + in_c * (H * W) + ih * W + iw
        val = tl.load(x_ptrs, mask=mask)
        tl.store(y_ptr + offs, val, mask=mask)

    n, c, h, w = x.shape
    out_c = c // (r * r)
    numel = n * out_c * h * r * w * r

    def _run():
        y = torch.empty((n, out_c, h * r, w * r), dtype=x.dtype, device=x.device)
        block = 1024
        grid = (triton.cdiv(numel, block),)
        _gather_kernel[grid](x, y, numel, c, h, w, R=r, BLOCK=block)
        return y

    def _build_and_bench(cfg):
        try:
            import triton.testing

            ref = torch.nn.functional.pixel_shuffle(x, r)
            out = _run()
            valid = bool(torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2))
            if not valid:
                return None
            triton_ms = triton.testing.do_bench(_run, warmup=10, rep=50)
            torch_ms = triton.testing.do_bench(
                lambda: torch.nn.functional.pixel_shuffle(x, r), warmup=10, rep=50
            )
            return triton_ms, torch_ms, True
        except Exception:
            return None

    shape_key = f"{n}x{h}x{w}x{c}|r={r}"
    cfg = _select_config("pixel_shuffle", shape_key, _build_and_bench, candidates=[{}])
    if cfg is None:
        return None
    try:
        return _run()
    except Exception:
        return None
