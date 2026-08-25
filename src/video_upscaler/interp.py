"""AMT frame-interpolation engine (slow motion) and checkpoint management.

Vendored AMT networks live in ``video_upscaler.amt``. This module builds the
torch model, runs the recursive 2x interpolation (mirroring AMT's
``demos/demo_2x.py``), and installs the official pretrained checkpoints from
the Clarity model hub (see ``modelhub.py``).

Backend selection: ``select_amt_backend`` picks the AMT backend. AMT-S, AMT-L,
and AMT-G use a cached TensorRT FP16 engine when CUDA + the TensorRT runtime
are available, else the PyTorch backend. ``--backend`` and
``CLARITY_AMT_BACKEND`` influence the choice (see
``CLARITY_AMT_BACKEND``/``CLARITY_AMT_PRECISION`` in config.py). ncnn Vulkan
does not accelerate AMT.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video_upscaler import config
from video_upscaler.amt.networks.amt_s import Model as AMT_S
from video_upscaler.amt.networks.amt_l import Model as AMT_L
from video_upscaler.amt.networks.amt_g import Model as AMT_G
from video_upscaler.amt.utils.utils import (
    InputPadder,
    img2tensor,
)
from video_upscaler.models import ckpt_for_interp_model

_MODEL_CLASSES = {"AMT-S": AMT_S, "AMT-L": AMT_L, "AMT-G": AMT_G}
_MODEL_PARAMS = {
    "AMT-S": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 3},
    "AMT-L": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 5},
    "AMT-G": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 5},
}


class AMTBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested AMT backend cannot be used."""


@dataclass(frozen=True)
class AMTBackendSelection:
    backend: str
    precision: str
    batch_size: int
    fallback_reason: str | None = None
    explicit_tensorrt: bool = False


@dataclass(frozen=True)
class AMTResolvedProfile:
    source_height: int
    source_width: int
    padded_height: int
    padded_width: int
    scale: float


def _cuda_available() -> bool:
    return _torch_cuda_available()


def _tensorrt_available() -> bool:
    from video_upscaler.tensorrt_backend import tensorrt_available

    try:
        return bool(tensorrt_available())
    except (OSError, FileNotFoundError):
        # Windows raises OSError/FileNotFoundError when the TensorRT runtime
        # DLLs are missing, not ImportError; treat that as unavailable so auto
        # selection falls back to PyTorch instead of aborting. Other runtime
        # failures are allowed to surface so they are not misreported as a
        # missing install.
        return False


def _adaptive_amt_scale(height: int, width: int) -> float:
    """Mirror AMT's adaptive-resolution calculation without loading a model."""
    import torch

    if not torch.cuda.is_available():
        return 1.0
    anchor_resolution = 1024 * 512
    anchor_memory = 1500 * 1024 ** 2
    anchor_memory_bias = 2500 * 1024 ** 2
    vram_avail = torch.cuda.get_device_properties("cuda").total_memory
    available = max(1, vram_avail - anchor_memory_bias)
    scale = anchor_resolution / (height * width) * math.sqrt(available / anchor_memory)
    scale = min(1.0, scale)
    return 1 / math.floor(1 / math.sqrt(scale) * 16) * 16


def resolve_amt_profile(height: int, width: int) -> AMTResolvedProfile:
    """Resolve AMT scale and static padded dimensions before backend creation."""
    if height <= 0 or width <= 0:
        raise ValueError("AMT frame dimensions must be positive")
    scale = _adaptive_amt_scale(height, width)
    divisor = int(16 / scale)
    padding = InputPadder((height, width), divisor)
    return AMTResolvedProfile(
        source_height=height,
        source_width=width,
        padded_height=height + padding._pad[2] + padding._pad[3],
        padded_width=width + padding._pad[0] + padding._pad[1],
        scale=scale,
    )


# Auto batch defaults on CUDA. AMT-L and AMT-G are heavier than AMT-S (5 flows
# vs 3, larger update blocks), so their measured safe defaults on the 6 GB RTX
# 3050 at 1080p are 1 (AMT-S stays 2). Explicit CLARITY_AMT_BATCH always wins.
_AUTO_BATCH_BY_MODEL = {"AMT-S": 2, "AMT-L": 1, "AMT-G": 1}


def _amt_batch_size(model_key: str) -> int:
    value = config.AMT_BATCH
    if value == "auto":
        if not _cuda_available():
            return 1
        return _AUTO_BATCH_BY_MODEL.get(model_key, 2)
    try:
        batch_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CLARITY_AMT_BATCH must be auto or a positive integer") from exc
    if batch_size < 1:
        raise ValueError("CLARITY_AMT_BATCH must be auto or a positive integer")
    return batch_size


def _requested_amt_backend() -> tuple[str, bool]:
    """Return the AMT preference and whether it was explicitly configured."""
    if config.AMT_BACKEND_EXPLICIT:
        return config.AMT_BACKEND_PREF, True
    preferred = config.BACKEND_PREF
    if preferred == "tensorrt":
        return "tensorrt", True
    if preferred in ("torch", "pytorch"):
        return "pytorch", True
    if preferred == "ncnn":
        return "pytorch", False
    return "auto", False


def _pytorch_selection(
    precision: str, batch_size: int, reason: str | None = None
) -> AMTBackendSelection:
    if precision == "fp16" and not _cuda_available():
        cpu_reason = "FP16 is unavailable without CUDA; using PyTorch FP32 on CPU."
        reason = f"{reason} {cpu_reason}" if reason else cpu_reason
        precision = "fp32"
    return AMTBackendSelection("pytorch", precision, batch_size, reason)


def select_amt_backend(model_key: str) -> AMTBackendSelection:
    """Select AMT's backend, applying explicit errors and auto fallbacks."""
    if model_key not in _MODEL_CLASSES:
        raise ValueError(f"unknown AMT model: {model_key}")
    requested, explicit = _requested_amt_backend()
    if requested not in ("auto", "pytorch", "tensorrt"):
        raise ValueError("CLARITY_AMT_BACKEND must be auto, pytorch, or tensorrt")

    batch_size = _amt_batch_size(model_key)
    precision = config.AMT_PRECISION
    if precision not in ("fp32", "fp16"):
        raise ValueError("CLARITY_AMT_PRECISION must be fp32 or fp16")

    if requested == "pytorch":
        reason = None
        if not config.AMT_BACKEND_EXPLICIT and config.BACKEND_PREF == "ncnn":
            reason = "ncnn does not provide the AMT backend; using PyTorch."
        return _pytorch_selection(precision, batch_size, reason)

    if model_key not in ("AMT-S", "AMT-L", "AMT-G"):
        reason = (
            "TensorRT AMT support is limited to AMT-S, AMT-L, and AMT-G; "
            "using PyTorch for this model."
        )
        if requested == "tensorrt" and explicit:
            raise AMTBackendUnavailable(reason)
        return _pytorch_selection(precision, batch_size, reason)

    if requested == "tensorrt":
        if not _cuda_available():
            raise AMTBackendUnavailable(
                "AMT TensorRT requires a CUDA device; select --backend torch or use AMT on CUDA."
            )
        if not _tensorrt_available():
            raise AMTBackendUnavailable(
                "AMT TensorRT runtime is unavailable; install the TensorRT extra/runtime and retry."
            )
        return AMTBackendSelection("tensorrt", "fp16", batch_size, explicit_tensorrt=True)

    if _cuda_available() and _tensorrt_available():
        return AMTBackendSelection("tensorrt", "fp16", batch_size)
    if not _cuda_available():
        reason = "CUDA is unavailable; using PyTorch."
    else:
        reason = "TensorRT runtime is unavailable or invalid; using PyTorch."
    return _pytorch_selection(precision, batch_size, reason)


def _tensorrt_onnx_path(spec) -> Path:
    path_builder = globals().get("amt_engine_path")
    if path_builder is None:
        from video_upscaler.amt_tensorrt import amt_engine_path

        path_builder = amt_engine_path
    return Path(path_builder(spec)).with_suffix(".onnx")


class AMTBackendFactory:
    """Build one AMT backend and cache shape-specific resources per job."""

    def __init__(
        self,
        model_key: str,
        selection: AMTBackendSelection,
        cache_root: Path | None = None,
    ) -> None:
        self.model_key = model_key
        self.selection = selection
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self._backends: dict[tuple[int, int] | None, object] = {}

    def build(self, frame_shape: tuple[int, int] | None = None):
        if self.selection.backend == "pytorch":
            if None not in self._backends:
                self._backends[None] = AMTInterpEngine(
                    self.model_key, precision=self.selection.precision
                )
            return self._backends[None]

        if frame_shape is None:
            raise ValueError("TensorRT AMT backend requires a frame shape")
        height, width = frame_shape
        profile = resolve_amt_profile(height, width)
        key = (profile.padded_height, profile.padded_width)
        if key in self._backends:
            return self._backends[key]

        from video_upscaler.amt_export import EXPORT_VERSION
        from video_upscaler.amt_tensorrt import AMTEngineSpec

        export_amt_pair = globals().get("export_amt_pair")
        validate_amt_onnx = globals().get("validate_amt_onnx")
        engine_class = globals().get("AMTTensorRTEngine")
        engine_path_builder = globals().get("amt_engine_path")
        if export_amt_pair is None or validate_amt_onnx is None:
            from video_upscaler.amt_export import export_amt_pair, validate_amt_onnx
        if engine_class is None:
            from video_upscaler.amt_tensorrt import AMTTensorRTEngine

            engine_class = AMTTensorRTEngine
        if engine_path_builder is None:
            from video_upscaler.amt_tensorrt import amt_engine_path

            engine_path_builder = amt_engine_path

        spec = AMTEngineSpec(
            model_key=self.model_key,
            padded_height=profile.padded_height,
            padded_width=profile.padded_width,
            scale=profile.scale,
            precision="fp16",
            max_batch=self.selection.batch_size,
            opset=config.AMT_ONNX_OPSET,
            export_version=EXPORT_VERSION,
        )
        engine_path = Path(engine_path_builder(spec))
        onnx_path = _tensorrt_onnx_path(spec)
        if not config.AMT_ENGINE_BUILD and (
            not engine_path.is_file() or not onnx_path.is_file()
        ):
            raise AMTBackendUnavailable(
                f"AMT TensorRT engine is missing and engine building is disabled: {engine_path}"
            )
        if config.AMT_ENGINE_CACHE and onnx_path.is_file() and engine_path.is_file():
            validate_amt_onnx(onnx_path)
        else:
            print(
                f"Preparing AMT TensorRT engine for {self.model_key} "
                "(one-time setup — this can take a few minutes)..."
            )
            import warnings

            from torch.jit import TracerWarning

            with warnings.catch_warnings():
                # The vendored AMT export emits benign PyTorch tracer and
                # ONNX-mode warnings; hide them so first-run output stays clear.
                warnings.simplefilter("ignore", TracerWarning)
                warnings.simplefilter("ignore", UserWarning)
                onnx_path = export_amt_pair(
                    self.model_key,
                    profile.padded_height,
                    profile.padded_width,
                    profile.scale,
                    onnx_path,
                    config.AMT_ONNX_OPSET,
                )
            validate_amt_onnx(onnx_path)
        engine_options = {}
        if not config.AMT_ENGINE_BUILD:
            engine_options["allow_build"] = False
        if not config.AMT_ENGINE_CACHE:
            engine_options["use_cache"] = False
        backend = engine_class(spec, onnx_path, **engine_options)
        self._backends[key] = backend
        return backend

    def close(self) -> None:
        for backend in self._backends.values():
            close = getattr(backend, "close", None)
            if close is not None:
                close()


def build_amt_backend(
    model_key: str,
    frame_shape: tuple[int, int] | None = None,
    selection: AMTBackendSelection | None = None,
):
    """Build an AMT backend through the single selection/factory path."""
    selection = selection or select_amt_backend(model_key)
    return AMTBackendFactory(model_key, selection).build(frame_shape)


def _timed_transfer(
    operation,
    device: str,
    torch,
    timings: dict[str, float],
    key: str,
    enabled: bool = False,
):
    """Run a host/device transfer and record its elapsed time."""
    if not enabled:
        return operation()
    if device == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = operation()
    if device == "cuda":
        torch.cuda.synchronize(device)
    timings[key] += time.perf_counter() - started
    return result


def _tensor2img_host(img_t):
    """Convert an already-host AMT tensor to an RGB8 frame without a transfer."""
    return (
        (img_t * 255.0)
        .detach()
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )


def _tensor2img_timed(
    img_t, device: str, torch, timings: dict[str, float], enabled: bool = False
):
    """Convert an AMT tensor to RGB8 while timing its final device transfer."""
    image = (img_t * 255.0).detach().squeeze(0).permute(1, 2, 0)
    image = _timed_transfer(
        lambda: image.cpu(),
        device,
        torch,
        timings,
        "d2h_time_s",
        enabled=enabled,
    )
    return image.numpy().clip(0, 255).astype(np.uint8)


def check_amt(model_key: str) -> str | None:
    """Return None if the AMT checkpoint exists, else a clear error message."""
    model_path = config.MODELS_DIR / ckpt_for_interp_model(model_key)
    if model_path.is_file():
        return None
    return (
        "Required AMT model not found:\n\n"
        f"{model_path}\n\n"
        "Run the app again and answer [Y] to download it automatically, or\n"
        "run once with --download-models essential|all.\n"
    )


def download_amt_model(model_key: str) -> Path:
    """Download one AMT checkpoint from the Clarity model hub.

    Raises HubError (RuntimeError) with an actionable message on failure.
    """
    from pathlib import PurePosixPath

    from video_upscaler import modelhub

    name = ckpt_for_interp_model(model_key)
    matches = [
        entry
        for entry in modelhub.entries(group="amt")
        if PurePosixPath(entry["dest"]).name == name
    ]
    if not matches:
        raise RuntimeError(
            f"AMT checkpoint {name} is not in the model manifest; "
            "regenerate it with tools/package_models.py."
        )
    return modelhub.install_entry(matches[0])


def _torch_cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


class AMTInterpEngine:
    """AMT frame-interpolation engine for one model (in-process, torch)."""

    def __init__(self, model_key: str, precision: str = "fp32") -> None:
        import torch

        self._torch = torch
        ckpt_name = ckpt_for_interp_model(model_key)
        model_path = config.MODELS_DIR / ckpt_name
        missing = check_amt(model_key)
        if missing:
            raise FileNotFoundError(missing)

        from video_upscaler.cugan import detect_device

        self.device = detect_device()
        if self.device == "cuda":
            print("Device: NVIDIA CUDA")
        elif self.device == "mps":
            print("Device: Apple Silicon (MPS)")
        else:
            print("Device: CPU")
            print("No GPU acceleration is available.")
            print("Falling back to CPU.")
            print()
            print("Interpolation may be significantly slower.")

        net = _MODEL_CLASSES[model_key](**_MODEL_PARAMS[model_key])
        # The official AMT checkpoints are pickled with `typing.OrderedDict`,
        # which torch 2.6+'s default weights_only=True loader rejects. We
        # trust the official source, so load with weights_only=False.
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        net = net.to(self.device).eval()
        self.model = net
        self.name = "pytorch"
        if precision not in ("fp32", "fp16"):
            raise ValueError("AMT precision must be fp32 or fp16")
        self.precision = precision
        self._transfer_timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
        self._timing_enabled = False
        self._batch_scale = None
        self._batch_padder = None

    def reset_timing(self) -> None:
        """Reset transfer timing counters for the next inference window."""
        for key in self._transfer_timings:
            self._transfer_timings[key] = 0.0

    def timing_snapshot(self) -> dict[str, float]:
        """Return cumulative host/device transfer timings for this window."""
        return dict(self._transfer_timings)

    def set_timing_enabled(self, enabled: bool) -> None:
        """Enable expensive synchronized transfer timing for a benchmark run."""
        self._timing_enabled = enabled

    def prepare_frames(self, frames: list[np.ndarray]):
        """Prepare one scheduler window using AMT's adaptive scale and padding."""
        if not frames:
            return []
        height, width = frames[0].shape[:2]
        if any(frame.shape[:2] != (height, width) for frame in frames):
            raise ValueError("all frames in an AMT window must have the same dimensions")
        scale, padder = self._prepare(height, width)
        self._batch_scale = scale
        self._batch_padder = padder
        return [padder.pad(img2tensor(frame)).squeeze(0) for frame in frames]

    def infer_batch(self, frame_a: "torch.Tensor", frame_b: "torch.Tensor"):
        """Infer a batch of normalized, padded frame pairs with the loaded model."""
        torch = self._torch
        if frame_a.shape != frame_b.shape:
            raise ValueError("AMT pair tensors must have matching shapes")
        if frame_a.ndim != 4 or frame_a.shape[1] != 3:
            raise ValueError("AMT pair tensors must have shape [B, 3, H, W]")
        if self._batch_scale is None:
            raise RuntimeError("prepare_frames must be called before infer_batch")

        frame_a = _timed_transfer(
            lambda: frame_a.to(self.device),
            self.device,
            torch,
            self._transfer_timings,
            "h2d_time_s",
            enabled=self._timing_enabled,
        )
        frame_b = _timed_transfer(
            lambda: frame_b.to(self.device),
            self.device,
            torch,
            self._transfer_timings,
            "h2d_time_s",
            enabled=self._timing_enabled,
        )
        embt = torch.full(
            (frame_a.shape[0], 1, 1, 1),
            0.5,
            dtype=frame_a.dtype,
            device=self.device,
        )
        with torch.inference_mode():
            if getattr(self, "precision", "fp32") == "fp16" and self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.model(
                        frame_a, frame_b, embt, scale_factor=self._batch_scale, eval=True
                    )["imgt_pred"]
            return self.model(
                frame_a, frame_b, embt, scale_factor=self._batch_scale, eval=True
            )["imgt_pred"]

    def transfer_batch_to_host(self, batch_output: "torch.Tensor"):
        """Move one scheduler output batch to host memory with optional timing."""
        return _timed_transfer(
            lambda: batch_output.detach().cpu(),
            self.device,
            self._torch,
            self._transfer_timings,
            "d2h_time_s",
            enabled=self._timing_enabled,
        )

    def finalize_frames(self, frames: list["torch.Tensor"]) -> list[np.ndarray]:
        """Unpad one prepared scheduler window and convert it to RGB8."""
        if self._batch_padder is None:
            raise RuntimeError("prepare_frames must be called before finalize_frames")
        unpadded = self._batch_padder.unpad(
            *(frame.unsqueeze(0) for frame in frames)
        )
        return [_tensor2img_host(frame) for frame in unpadded]

    def warmup(self, shape: tuple[int, int], batch_size: int) -> None:
        """Run one small batch so cuDNN/torch kernels are primed for the job."""
        torch = self._torch
        height, width = shape
        frames = [np.zeros((height, width, 3), dtype=np.uint8)]
        prepared = self.prepare_frames(frames)
        pair = torch.stack([prepared[0]] * batch_size)
        self.transfer_batch_to_host(self.infer_batch(pair, pair))

    def close(self) -> None:
        """Release backend resources owned outside the model lifecycle."""
        return None

    def _prepare(self, h: int, w: int):
        """Compute the adaptive downscale factor and padder (AMT demo logic)."""
        torch = self._torch
        if self.device == "cuda":
            anchor_resolution = 1024 * 512
            anchor_memory = 1500 * 1024 ** 2
            anchor_memory_bias = 2500 * 1024 ** 2
            vram_avail = torch.cuda.get_device_properties(self.device).total_memory
            scale = anchor_resolution / (h * w) * math.sqrt(
                (vram_avail - anchor_memory_bias) / anchor_memory
            )
        else:
            scale = 1.0
        scale = 1.0 if scale > 1 else scale
        scale = 1 / math.floor(1 / math.sqrt(scale) * 16) * 16
        if scale < 1:
            print(
                f"Due to limited VRAM, frames will be processed at {scale:.2f}x "
                "internal resolution."
            )
        padding = int(16 / scale)
        padder = InputPadder((h, w), padding)
        return scale, padder

    def interpolate(self, frames: list[np.ndarray], niters: int):
        """Yield interpolated RGB uint8 frames for one window of source frames.

        Runs ``niters`` recursive 2x passes (2x/4x/8x). Callers feed windows
        (see ``processor.process_interpolate``), which handles overlap and
        bounded memory; this method yields every output frame of the window,
        including its boundary frame, so callers can decide stitching.
        """
        torch = self._torch
        self.reset_timing()
        h, w = frames[0].shape[:2]
        scale, padder = self._prepare(h, w)
        embt = _timed_transfer(
            lambda: torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(self.device),
            self.device,
            torch,
            self._transfer_timings,
            "h2d_time_s",
            enabled=self._timing_enabled,
        )

        inputs = [
            padder.pad(
                _timed_transfer(
                    lambda f=f: img2tensor(f).to(self.device),
                    self.device,
                    torch,
                    self._transfer_timings,
                    "h2d_time_s",
                    enabled=self._timing_enabled,
                )
            )
            for f in frames
        ]
        for _ in range(niters):
            outputs = [inputs[0]]
            for a, b in zip(inputs[:-1], inputs[1:]):
                with torch.no_grad():
                    pred = self.model(
                        a, b, embt, scale_factor=scale, eval=True
                    )["imgt_pred"]
                outputs += [
                    _timed_transfer(
                        lambda pred=pred: pred.cpu(),
                        self.device,
                        torch,
                        self._transfer_timings,
                        "d2h_time_s",
                        enabled=self._timing_enabled,
                    ),
                    _timed_transfer(
                        lambda b=b: b.cpu(),
                        self.device,
                        torch,
                        self._transfer_timings,
                        "d2h_time_s",
                        enabled=self._timing_enabled,
                    ),
                ]
            inputs = outputs

        out_tensors = padder.unpad(*inputs)
        for t in out_tensors:
            yield _tensor2img_timed(
                t,
                self.device,
                torch,
                self._transfer_timings,
                enabled=self._timing_enabled,
            )
