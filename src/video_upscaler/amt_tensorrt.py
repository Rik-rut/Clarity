"""TensorRT FP16 backend and cache for the fixed AMT-S/AMT-L/AMT-G pair graph."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from video_upscaler.amt_backend import AMTBackend
from video_upscaler import config

MODELS_DIR = config.MODELS_DIR
from video_upscaler.tensorrt_backend import (
    _cuda_version,
    gpu_name as _gpu_name,
    prepare_env,
)

ENGINE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class AMTEngineSpec:
    model_key: str
    padded_height: int
    padded_width: int
    scale: float
    precision: str
    max_batch: int
    opset: int
    export_version: int

    def __post_init__(self) -> None:
        if self.model_key not in ("AMT-S", "AMT-L", "AMT-G"):
            raise ValueError(
                "AMT TensorRT supports model_key 'AMT-S', 'AMT-L', or 'AMT-G', "
                f"got {self.model_key!r}"
            )
        if self.precision != "fp16":
            raise ValueError(f"AMT TensorRT supports precision 'fp16', got {self.precision!r}")
        if self.padded_height <= 0 or self.padded_width <= 0:
            raise ValueError("padded height and width must be positive")
        if not math.isfinite(self.scale) or not 0 < self.scale <= 1:
            raise ValueError("scale must be greater than 0 and no greater than 1")
        internal_height = math.floor(self.padded_height * self.scale)
        internal_width = math.floor(self.padded_width * self.scale)
        if min(internal_height, internal_width) < 128:
            raise ValueError("padded shape scaled dimensions must be at least 128x128")
        if internal_height % 16 or internal_width % 16:
            raise ValueError("padded shape scaled dimensions must be divisible by 16")
        if self.max_batch < 1:
            raise ValueError("max_batch must be positive")
        if self.opset < 11:
            raise ValueError("opset must be at least 11")
        if self.export_version < 1:
            raise ValueError("export_version must be positive")


def _scale_label(scale: float) -> str:
    text = f"{scale:g}"
    return text.replace(".", "p")


def _cache_dir() -> Path:
    return MODELS_DIR / "amt" / "tensorrt"


def amt_engine_path(spec: AMTEngineSpec) -> Path:
    """Return the deterministic AMT engine cache path for ``spec``."""
    return _cache_dir() / (
        f"{spec.model_key}_{spec.padded_height}x{spec.padded_width}_"
        f"s{_scale_label(spec.scale)}_{spec.precision}_b{spec.max_batch}_"
        f"o{spec.opset}_e{spec.export_version}_{_gpu_name()}.engine"
    )


def _onnx_path(spec: AMTEngineSpec) -> Path:
    return amt_engine_path(spec).with_suffix(".onnx")


def _checkpoint_path(model_key: str) -> Path:
    from video_upscaler.models import ckpt_for_interp_model

    return MODELS_DIR / ckpt_for_interp_model(model_key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_metadata() -> dict[str, str]:
    """Return runtime identity used to prevent stale engine reuse."""
    try:
        import tensorrt as trt
    except (ImportError, FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"TensorRT runtime unavailable: {exc}") from exc

    try:
        if torch.cuda.is_available():
            capability = ".".join(str(part) for part in torch.cuda.get_device_capability())
        else:
            capability = "unknown"
    except (AttributeError, RuntimeError):
        capability = "unknown"
    return {
        "gpu": _gpu_name(),
        "compute_capability": capability,
        "cuda": _cuda_version(),
        "tensorrt": str(trt.__version__),
    }


def _cache_metadata(spec: AMTEngineSpec, onnx_path: Path) -> dict[str, Any]:
    checkpoint_path = _checkpoint_path(spec.model_key)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"AMT checkpoint not found: {checkpoint_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"AMT ONNX graph not found: {onnx_path}")
    metadata = _runtime_metadata()
    return {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "onnx_sha256": _sha256(onnx_path),
        "model": spec.model_key,
        "model_key": spec.model_key,
        "gpu": metadata["gpu"],
        "compute_capability": metadata["compute_capability"],
        "cuda": metadata["cuda"],
        "tensorrt": metadata["tensorrt"],
        "padded_height": spec.padded_height,
        "padded_width": spec.padded_width,
        "padded_shape": [spec.padded_height, spec.padded_width],
        "scale": spec.scale,
        "precision": spec.precision,
        "max_batch": spec.max_batch,
        "opset": spec.opset,
        "export_version": spec.export_version,
        "engine_format_version": ENGINE_FORMAT_VERSION,
    }


def _metadata_path(engine_path: Path) -> Path:
    return engine_path.with_suffix(".json")


def _read_metadata(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def ensure_amt_engine(
    spec: AMTEngineSpec,
    onnx_path: Path,
    *,
    allow_build: bool = True,
    use_cache: bool = True,
) -> Path:
    """Reuse a compatible engine or build a new one from ``onnx_path``."""
    environment_error = prepare_env()
    if environment_error:
        raise RuntimeError(environment_error)

    onnx_path = Path(onnx_path)
    current = _cache_metadata(spec, onnx_path)
    engine_path = amt_engine_path(spec)
    cached = _read_metadata(_metadata_path(engine_path))
    if use_cache and engine_path.is_file() and cached == current:
        return engine_path
    if not allow_build:
        raise RuntimeError(
            f"AMT TensorRT engine is unavailable or incompatible and engine building "
            f"is disabled: {engine_path}"
        )
    return _build_amt_engine(spec, onnx_path)


def _configure_fp16(trt: Any, config: Any) -> bool:
    """Enable the legacy flag and always request an FP16 ONNX graph."""
    flag = getattr(trt.BuilderFlag, "FP16", None)
    if flag is not None:
        config.set_flag(flag)
    return True


def _convert_onnx_to_fp16(path: Path) -> Path:
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT 11 requires ONNX Runtime to convert the AMT graph to FP16"
        ) from exc
    with tempfile.NamedTemporaryFile(
        prefix="clarity_amt_fp16_", suffix=".onnx", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        model = onnx.load(str(path))
        onnx.save(convert_float_to_float16(model, keep_io_types=False), str(temporary))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _cleanup_temp_graph(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        # TensorRT may release its Windows file handle after the builder returns.
        pass


def _validate_graph_contract(network: Any, trt: Any, spec: AMTEngineSpec) -> None:
    if network.num_inputs != 2:
        raise RuntimeError(f"TensorRT AMT graph must have exactly two inputs, got {network.num_inputs}")
    inputs = [network.get_input(index) for index in range(network.num_inputs)]
    if [tensor.name for tensor in inputs] != ["frame_a", "frame_b"]:
        raise RuntimeError("TensorRT AMT graph input names must be frame_a and frame_b")

    expected_dtype = trt.DataType.HALF
    for tensor in inputs:
        shape = tuple(tensor.shape)
        if len(shape) != 4 or shape[0] != -1:
            raise RuntimeError(f"TensorRT AMT graph input must have a dynamic batch shape, got {shape}")
        if shape[1] != 3:
            raise RuntimeError(f"TensorRT AMT graph input channels must be 3, got {shape[1]}")
        if tuple(shape[2:]) != (spec.padded_height, spec.padded_width):
            raise RuntimeError(
                f"TensorRT AMT graph input shape is incompatible with padded shape: {shape}"
            )
        if tensor.dtype != expected_dtype:
            raise RuntimeError(f"TensorRT AMT graph input dtype must be FP16, got {tensor.dtype}")

    if network.num_outputs != 1:
        raise RuntimeError(
            f"TensorRT AMT graph must have exactly one output, got {network.num_outputs}"
        )
    output = network.get_output(0)
    if output.name != "output":
        raise RuntimeError(f"TensorRT AMT graph output name must be output, got {output.name}")
    output_shape = tuple(output.shape)
    if (
        len(output_shape) != 4
        or output_shape[0] != -1
        or output_shape[1] != 3
        or tuple(output_shape[2:]) != (spec.padded_height, spec.padded_width)
    ):
        raise RuntimeError(f"TensorRT AMT graph output shape is unsupported: {output_shape}")
    if output.dtype != expected_dtype:
        raise RuntimeError(f"TensorRT AMT graph output dtype must be FP16, got {output.dtype}")


def _build_amt_engine(spec: AMTEngineSpec, onnx_path: Path) -> Path:
    try:
        import tensorrt as trt
    except (ImportError, FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"TensorRT runtime unavailable: {exc}") from exc

    print(
        f"Building TensorRT FP16 engine for {spec.model_key} "
        f"({spec.padded_height}x{spec.padded_width}) — one-time, please wait..."
    )

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    builder_config = builder.create_builder_config()
    workspace_bytes = int(config.AMT_TRT_WORKSPACE_GIB * (1 << 30))
    if workspace_bytes <= 0:
        raise ValueError("CLARITY_AMT_TRT_WORKSPACE_GIB must be positive")
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    needs_fp16_graph = _configure_fp16(trt, builder_config)
    graph_path = _convert_onnx_to_fp16(onnx_path) if needs_fp16_graph else onnx_path
    try:
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(graph_path)):
            errors = "\n".join(
                f"  {parser.get_error(index)}" for index in range(parser.num_errors)
            )
            raise RuntimeError(f"TensorRT AMT ONNX parser failure:\n{errors}")

        _validate_graph_contract(network, trt, spec)
        inputs = [network.get_input(index) for index in range(network.num_inputs)]

        profile = builder.create_optimization_profile()
        opt_batch = min(2, spec.max_batch)
        for tensor in inputs:
            profile.set_shape(
                tensor.name,
                (1, 3, spec.padded_height, spec.padded_width),
                (opt_batch, 3, spec.padded_height, spec.padded_width),
                (spec.max_batch, 3, spec.padded_height, spec.padded_width),
            )
        builder_config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, builder_config)
        if serialized is None:
            raise RuntimeError("TensorRT AMT engine build failure")

        output_path = amt_engine_path(spec)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_bytes(bytes(serialized))
        temporary.replace(output_path)
        _write_metadata(_metadata_path(output_path), _cache_metadata(spec, onnx_path))
        return output_path
    finally:
        if graph_path != onnx_path:
            _cleanup_temp_graph(graph_path)


def _import_runtime():
    environment_error = prepare_env()
    if environment_error:
        raise RuntimeError(environment_error)
    try:
        import tensorrt as trt
    except (ImportError, FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"TensorRT runtime unavailable: {exc}") from exc
    return trt


class AMTTensorRTEngine(AMTBackend):
    """AMT-S/AMT-L/AMT-G TensorRT engine with persistent buffers and pair batching."""

    def __init__(
        self,
        spec: AMTEngineSpec,
        onnx_path: Path | None = None,
        *,
        allow_build: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.spec = spec
        self.name = "tensorrt"
        self.precision = spec.precision
        self.device = "cuda"
        self._padder = None
        self._source_shape: tuple[int, int] | None = None
        onnx_path = Path(onnx_path) if onnx_path is not None else _onnx_path(spec)
        self._onnx_path = onnx_path
        if not onnx_path.is_file():
            raise FileNotFoundError(f"AMT ONNX graph not found: {onnx_path}")
        engine_options = {}
        if not allow_build:
            engine_options["allow_build"] = False
        if not use_cache:
            engine_options["use_cache"] = False
        engine_path = ensure_amt_engine(spec, onnx_path, **engine_options)
        trt = _import_runtime()
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to load TensorRT AMT engine {engine_path}: {exc}") from exc
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT AMT engine: {engine_path}")
        self._runtime = runtime
        self._engine = engine
        self._context = engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=self.device)
        self._input_names = ["frame_a", "frame_b"]
        self._output_name = "output"
        self._input_buffers: list[torch.Tensor] | None = None
        self._output_buffer: torch.Tensor | None = None
        self._buffer_capacity: tuple[int, ...] | None = None
        self._transfer_timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
        self._timing_enabled = False

    def reset_timing(self) -> None:
        for key in self._transfer_timings:
            self._transfer_timings[key] = 0.0

    def timing_snapshot(self) -> dict[str, float]:
        return dict(self._transfer_timings)

    def set_timing_enabled(self, enabled: bool) -> None:
        self._timing_enabled = enabled

    def _timed_operation(self, key: str, operation):
        if not self._timing_enabled:
            return operation()
        started = time.perf_counter()
        result = operation()
        self._transfer_timings[key] += time.perf_counter() - started
        return result

    def _validate_batch(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> None:
        if frame_a.shape != frame_b.shape:
            raise ValueError("AMT pair tensors must have matching shapes")
        if frame_a.ndim != 4 or frame_a.shape[1] != 3:
            raise ValueError("AMT pair tensors must have shape [B, 3, H, W]")
        if frame_a.device.type != "cpu" or frame_b.device.type != "cpu":
            raise ValueError("AMT TensorRT inputs must be CPU tensors")
        if tuple(frame_a.shape[2:]) != (self.spec.padded_height, self.spec.padded_width):
            raise ValueError(
                "AMT TensorRT input has unsupported padded shape: "
                f"{tuple(frame_a.shape[2:])}"
            )
        if not 1 <= frame_a.shape[0] <= self.spec.max_batch:
            raise ValueError(f"AMT TensorRT batch must be between 1 and {self.spec.max_batch}")
        if not (frame_a.is_floating_point() and frame_b.is_floating_point()):
            raise ValueError("AMT TensorRT inputs must be floating point tensors")

    def _ensure_buffers(self, batch_size: int) -> None:
        shape = (batch_size, 3, self.spec.padded_height, self.spec.padded_width)
        capacity = (self.spec.max_batch, 3, self.spec.padded_height, self.spec.padded_width)
        self._context.set_input_shape(self._input_names[0], shape)
        self._context.set_input_shape(self._input_names[1], shape)
        output_shape = tuple(self._context.get_tensor_shape(self._output_name))
        if len(output_shape) != 4 or output_shape[0] != batch_size:
            raise RuntimeError(f"TensorRT AMT output batch shape is unsupported: {output_shape}")
        if tuple(output_shape[2:]) != (self.spec.padded_height, self.spec.padded_width) or output_shape[1] != 3:
            raise RuntimeError(f"TensorRT AMT output shape is unsupported: {output_shape}")
        if self._buffer_capacity == capacity:
            return
        self._input_buffers = [
            torch.empty(capacity, dtype=torch.float16, device=self.device),
            torch.empty(capacity, dtype=torch.float16, device=self.device),
        ]
        self._output_buffer = torch.empty(
            (self.spec.max_batch, 3, self.spec.padded_height, self.spec.padded_width),
            dtype=torch.float16,
            device=self.device,
        )
        for name, buffer in zip(self._input_names, self._input_buffers):
            self._context.set_tensor_address(name, buffer.data_ptr())
        self._context.set_tensor_address(self._output_name, self._output_buffer.data_ptr())
        self._buffer_capacity = capacity

    def infer_batch(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        self._validate_batch(frame_a, frame_b)
        self._ensure_buffers(frame_a.shape[0])
        assert self._input_buffers is not None
        assert self._output_buffer is not None
        batch_size = frame_a.shape[0]
        stream_ctx = (
            torch.cuda.stream(self._stream)
            if isinstance(self._stream, torch.cuda.Stream)
            else contextlib.nullcontext()
        )
        stream_handle = getattr(self._stream, "cuda_stream", 0)
        with stream_ctx:
            self._timed_operation("h2d_time_s", lambda: self._input_buffers[0][:batch_size].copy_(frame_a))
            self._timed_operation("h2d_time_s", lambda: self._input_buffers[1][:batch_size].copy_(frame_b))
            executed = self._context.execute_async_v3(stream_handle=stream_handle)
        if executed is False:
            raise RuntimeError("TensorRT AMT execution failed")
        return self._output_buffer[:batch_size]

    def prepare_frames(self, frames: list[np.ndarray]) -> list[torch.Tensor]:
        if not frames:
            return []
        if any(frame.ndim != 3 or frame.shape[2] < 3 for frame in frames):
            raise ValueError("AMT frames must be RGB or RGBA arrays")
        height, width = frames[0].shape[:2]
        if any(frame.shape[:2] != (height, width) for frame in frames):
            raise ValueError("all frames in an AMT window must have the same dimensions")
        from video_upscaler.amt.utils.utils import InputPadder, img2tensor

        divisor = int(round(16 / self.spec.scale))
        padder = InputPadder((height, width), divisor)
        prepared = [padder.pad(img2tensor(frame)).squeeze(0) for frame in frames]
        if any(tuple(frame.shape[-2:]) != (self.spec.padded_height, self.spec.padded_width) for frame in prepared):
            raise ValueError(
                "AMT input produces an unsupported padded shape; expected "
                f"({self.spec.padded_height}, {self.spec.padded_width})"
            )
        self._padder = padder
        self._source_shape = (height, width)
        return prepared

    def transfer_batch_to_host(self, batch_output: torch.Tensor) -> torch.Tensor:
        if batch_output.device.type != "cuda":
            raise ValueError("AMT TensorRT output must be a CUDA tensor")
        started = time.perf_counter() if self._timing_enabled else None
        torch.cuda.synchronize(self.device)
        result = batch_output.detach().float().cpu()
        torch.cuda.synchronize(self.device)
        if started is not None:
            self._transfer_timings["d2h_time_s"] += time.perf_counter() - started
        return result

    def finalize_frames(self, frames: list[torch.Tensor]) -> list[np.ndarray]:
        if self._padder is None:
            raise RuntimeError("prepare_frames must be called before finalize_frames")
        unpadded = self._padder.unpad(*(frame.unsqueeze(0) for frame in frames))
        return [
            (frame.detach().squeeze(0).permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().numpy())
            for frame in unpadded
        ]

    def warmup(self, shape: tuple[int, int], batch_size: int) -> None:
        if not 1 <= batch_size <= self.spec.max_batch:
            raise ValueError(f"warmup batch must be between 1 and {self.spec.max_batch}")
        height, width = shape
        prepared = self.prepare_frames(
            [np.zeros((height, width, 3), dtype=np.uint8)]
        )
        pair = torch.stack([prepared[0]] * batch_size)
        self.transfer_batch_to_host(self.infer_batch(pair, pair))

    def close(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self._input_buffers = None
        self._output_buffer = None
        self._buffer_capacity = None
        self._context = None
        self._engine = None
        self._runtime = None
