"""TensorRT backend for Real-CUGAN (NVIDIA only).

The vendored torch model is exported to ONNX (the whole-frame path,
tile_mode=0) and converted to a cached FP16 TensorRT engine. Pre/post
processing mirrors the torch engine exactly: reflect-pad the frame to even
dims + 36, run the engine, crop to h0*scale x w0*scale, *255 round clamp.

Engine cache lives in ``models/tensorrt/``:

    <model-stem>_x<scale>_fp16_<gpu>.engine
    metadata.json

Engines are rebuilt only when the model hash, scale, precision, TRT/CUDA
versions, GPU, or build parameters change. Never silently overwritten.

The whole module is import-lazy: nothing loads unless the tensorrt extra is
installed and an engine is actually requested. Set CLARITY_TENSORRT_DIR to
override the TensorRT install location. Module top level stays torch-free so
unit tests never load torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from video_upscaler.config import MODELS_DIR

# The official zip ships the Python wheel WITHOUT the nvinfer DLLs; the
# runtime DLLs live in <install>/bin and must precede the process DLL search
# path before ``import tensorrt`` runs (tensorrt's find_lib scans PATH).
_TRT_ENV_READY = False


def _trt_site_packages_dirs() -> list[Path]:
    """Candidate site-packages directories holding TensorRT runtime DLLs."""
    dirs: list[Path] = []
    try:
        import sys

        site_roots: list[Path] = []
        for p in sys.path:
            if p and Path(p).is_dir() and "site-packages" in Path(p).name:
                site_roots.append(Path(p))
        prefix_site = Path(sys.prefix) / "Lib" / "site-packages"
        if prefix_site.is_dir() and prefix_site not in site_roots:
            site_roots.append(prefix_site)

        for root in site_roots:
            for pattern in ("tensorrt_libs", "tensorrt_cu*_libs", "tensorrt"):
                for d in root.glob(pattern):
                    if d.is_dir() and d not in dirs:
                        dirs.append(d)
    except Exception:
        pass
    return dirs


def _trt_install_dirs() -> list[Path]:
    """Candidate TensorRT install roots (most recently modified first)."""
    roots: list[Path] = []
    env = os.environ.get("CLARITY_TENSORRT_DIR")
    if env:
        roots.append(Path(env))
    local = Path(__file__).resolve().parents[2] / "tools" / "TensorRT"
    if local.is_dir():
        roots.extend(
            sorted(
                local.glob("TensorRT-*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        )
    return roots


def _trt_bin_dir() -> Path | None:
    """Return the bin dir holding nvinfer DLLs, or None."""
    for root in _trt_install_dirs():
        for sub in (root, root / "bin"):
            for dll_name in ("nvinfer_11.dll", "nvinfer_10.dll", "nvinfer.dll"):
                if (sub / dll_name).is_file():
                    return sub
    for sub in _trt_site_packages_dirs():
        for dll_name in ("nvinfer_11.dll", "nvinfer_10.dll", "nvinfer.dll"):
            if (sub / dll_name).is_file():
                return sub
    return None


def _torch_lib_dir() -> Path | None:
    """torch/lib dir (cudart/cublas/cudnn DLLs TensorRT links against)."""
    try:
        import torch

        return Path(torch.__file__).resolve().parent / "lib"
    except ImportError:
        return None


def prepare_env() -> str | None:
    """Prepend TensorRT bin + torch/lib to PATH; returns an error or None."""
    global _TRT_ENV_READY
    if _TRT_ENV_READY:
        return None
    bin_dir = _trt_bin_dir()
    if bin_dir is None:
        return (
            "TensorRT install not found. Install via 'uv sync --extra tensorrt' (or pip install tensorrt), "
            "or download the TensorRT Windows zip from https://developer.nvidia.com/tensorrt/download "
            "and extract it into tools/TensorRT/, or set CLARITY_TENSORRT_DIR."
        )

    torch_lib = _torch_lib_dir()

    parts: list[str] = [str(bin_dir)]
    if torch_lib is not None:
        parts.append(str(torch_lib))

    os.environ["PATH"] = os.pathsep.join(parts) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        for p in parts:
            try:
                if Path(p).is_dir():
                    os.add_dll_directory(p)
            except Exception:
                pass

    try:
        import tensorrt as trt  # noqa: F401

        _TRT_ENV_READY = True
        return None
    except (ImportError, OSError, FileNotFoundError) as exc:
        return f"Failed to initialize TensorRT runtime: {exc}"


def tensorrt_available() -> bool:
    """Return True when the tensorrt package can be imported."""
    if prepare_env() is not None:
        return False
    try:
        import tensorrt  # noqa: F401

        return True
    except (ImportError, OSError, FileNotFoundError):
        # Windows surfaces a missing nvinfer DLL as OSError/FileNotFoundError
        # rather than ImportError; any of them means the runtime is unusable.
        return False


def gpu_name() -> str:
    """GPU name used in engine cache filenames (nvidia-smi, else torch)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip().splitlines()[0].strip()
            return "".join(c if c.isalnum() else "_" for c in name)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return "".join(c if c.isalnum() else "_" for c in name)
    except ImportError:
        pass
    return "unknown_gpu"


def _model_hash(model_path: Path) -> str:
    """SHA-256 of the weight file (streamed)."""
    digest = hashlib.sha256()
    with open(model_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _trt_version() -> str:
    import tensorrt as trt

    return str(trt.__version__)


def _cuda_version() -> str:
    try:
        import torch

        return torch.version.cuda or "unknown"
    except ImportError:
        return "unknown"


def _cache_dir() -> Path:
    return MODELS_DIR / "tensorrt"


def engine_path(model_name: str, scale: int) -> Path:
    stem = Path(model_name).stem
    return _cache_dir() / f"{stem}_x{scale}_fp16_{gpu_name()}.engine"


def metadata_path() -> Path:
    return _cache_dir() / "metadata.json"


def load_metadata() -> dict:
    path = metadata_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_metadata(entry: dict) -> None:
    data = load_metadata()
    data[entry["model"]] = entry
    path = metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(path)


def _make_export_net(model, scale: int):
    """Build the exportable whole-frame graph for a loaded upscaler model.

    Matches UpCunet{2,3,4}x forward with tile_mode=0, cache_mode=0,
    alpha=1: unet1 -> unet2(unet1_out) -> add cropped unet1_out. The 4x
    variant also runs conv_final + pixel shuffle; its residual add of the
    input happens in Python post-processing.
    """
    import torch
    from torch import nn as nn
    from torch.nn import functional as F

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unet1 = model.unet1
            self.unet2 = model.unet2
            self.conv_final = getattr(model, "conv_final", None)
            self.ps = getattr(model, "ps", None)

        def forward(self, x):
            x1 = self.unet1(x)
            x0 = self.unet2(x1, 1.0)
            x1c = F.pad(x1, (-20, -20, -20, -20))
            out = torch.add(x0, x1c)
            if self.conv_final is not None:
                out = self.conv_final(out)
                out = F.pad(out, (-1, -1, -1, -1))
                out = self.ps(out)
            return out

    return _Net().to("cuda").eval()


def export_onnx(model_name: str, scale: int, onnx_path: Path) -> None:
    """Export the whole-frame network for a model to ONNX (fp16 graph)."""
    import torch

    from video_upscaler.quantize import ensure_fp16
    from video_upscaler.upcunet_v3 import RealWaifuUpScaler

    weight = ensure_fp16(model_name)
    upscaler = RealWaifuUpScaler(scale, str(weight), half=True, device="cuda")
    net = _make_export_net(upscaler.model, scale)

    dummy = torch.zeros((1, 3, 720 + 36, 1280 + 36), dtype=torch.float16, device="cuda")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height2", 3: "width2"},
        },
    )


def build_engine(
    model_name: str,
    scale: int,
    *,
    precision: str = "fp16",
    workspace_gib: float = 1.0,
) -> Path:
    """Build (or reuse) the cached TRT engine for a model; return its path."""
    prepare_env()
    import tensorrt as trt

    onnx_path = _cache_dir() / f"{Path(model_name).stem}_x{scale}.onnx"
    export_onnx(model_name, scale, onnx_path)
    import torch

    torch.cuda.empty_cache()  # release the export model's memory for the build

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT 10+ is always explicit-batch; create_network() takes no flags.
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "".join(
            f"\n  {parser.get_error(i)}" for i in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT ONNX parse failed:{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30))
    )

    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    # Tactic workspaces scale with the profile MAX shape: the 2x-res unet2
    # convs need multi-GB buffers, so the max is capped at the padded
    # 960x1080 tile the runtime feeds the engine (mirroring torch's
    # tile_mode=1), and batch is capped at 2 tiles per engine call (a 6 GB
    # card cannot fit batch-4 tactics). Larger inputs are tiled by
    # RealCUGANTensorRTEngine.
    profile.set_shape(
        input_tensor.name,
        (1, 3, 128, 128),
        (1, 3, 1116, 996),
        (2, 3, 1116, 996),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed.")

    out_path = engine_path(model_name, scale)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(serialized)

    save_metadata(
        {
            "model": model_name,
            "scale": scale,
            "precision": precision,
            "model_hash": _model_hash(MODELS_DIR / model_name),
            "gpu": gpu_name(),
            "tensorrt": _trt_version(),
            "cuda": _cuda_version(),
            "python": platform.python_version(),
            "engine_version": 2,
        }
    )
    return out_path


def ensure_engine(model_name: str, scale: int) -> Path:
    """Return a valid cached engine path, rebuilding when stale."""
    prepare_env()
    path = engine_path(model_name, scale)
    meta = load_metadata().get(model_name)
    current = {
        "model": model_name,
        "scale": scale,
        "precision": "fp16",
        "model_hash": _model_hash(MODELS_DIR / model_name),
        "gpu": gpu_name(),
        "tensorrt": _trt_version(),
        "cuda": _cuda_version(),
        "engine_version": 2,
    }
    if (
        path.is_file()
        and meta is not None
        and meta.get("model_hash") == current["model_hash"]
        and meta.get("scale") == current["scale"]
        and meta.get("gpu") == current["gpu"]
        and meta.get("tensorrt") == current["tensorrt"]
        and meta.get("engine_version") == current["engine_version"]
    ):
        return path
    print(f"Building TensorRT engine for {model_name} (fp16)...")
    return build_engine(model_name, scale)


def _tile_grid(h0: int, w0: int) -> tuple[int, int, int, int]:
    """Tile geometry (crop_h, crop_w, ph, pw) mirroring torch tile_mode=1."""
    if w0 >= h0:
        crop_w = ((w0 - 1) // 4 * 4 + 4) // 2
        crop_h = (h0 - 1) // 2 * 2 + 2
    else:
        crop_h = ((h0 - 1) // 4 * 4 + 4) // 2
        crop_w = (w0 - 1) // 2 * 2 + 2
    ph = ((h0 - 1) // crop_h + 1) * crop_h
    pw = ((w0 - 1) // crop_w + 1) * crop_w
    return crop_h, crop_w, ph, pw


class RealCUGANTensorRTEngine:
    """Real-CUGAN via a cached FP16 TensorRT engine (whole-frame path)."""

    def __init__(self, model_name: str) -> None:
        import torch

        from video_upscaler.models import scale_for_model

        self._scale = scale_for_model(model_name)
        if self._scale != 2:
            raise RuntimeError(
                "TensorRT Real-CUGAN currently supports 2x models only "
                "(use the torch backend for 3x/4x)."
            )
        path = ensure_engine(model_name, self._scale)

        import tensorrt as trt

        prepare_env()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self._engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")
        self._context = self._engine.create_execution_context()
        self._device = "cuda"
        self._stream = torch.cuda.Stream(device=self._device)
        self._input_name = self._engine.get_tensor_name(0)
        self._output_name = self._engine.get_tensor_name(1)
        self._in_shape = None
        self._in_buf: torch.Tensor | None = None
        self._out_buf: torch.Tensor | None = None

    def _ensure_buffers(self, height: int, width: int, batch: int) -> None:
        """Allocate (or reuse) device buffers sized exactly for the input.

        TRT 11 writes the output assuming the buffer matches the tensor's
        exact shape â€” an oversized buffer yields corrupted results.
        """
        import torch

        shape = (batch, 3, height, width)
        if self._in_shape == shape:
            return
        self._in_buf = torch.empty(shape, dtype=torch.float16, device=self._device)
        self._context.set_input_shape(self._input_name, shape)
        out_shape = tuple(self._context.get_tensor_shape(self._output_name))
        self._out_buf = torch.empty(out_shape, dtype=torch.float16, device=self._device)
        self._context.set_tensor_address(self._input_name, self._in_buf.data_ptr())
        self._context.set_tensor_address(self._output_name, self._out_buf.data_ptr())
        self._in_shape = shape

    def _pre(self, frames: list[np.ndarray]) -> "torch.Tensor":
        """Convert RGB frames to fp16 tensors in [0,1] (unpadded)."""
        import torch

        tensors = [
            torch.from_numpy(np.ascontiguousarray(f.transpose(2, 0, 1)))
            .unsqueeze(0)
            .to(self._device)
            .half()
            / 255
            for f in frames
        ]
        return torch.cat(tensors, dim=0)

    def _tiles(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run the engine over tiles (torch tile_mode=1 geometry).

        The frame is reflect-padded to even tile multiples + 18 and cut into
        tiles of the engine's build size; the engine's output tiles (exactly
        2x the tile's unpadded area) are assembled. SE means are per-tile,
        a small deviation from the torch engine's global SE means.
        """
        import torch
        from torch.nn import functional as F

        n, c, h0, w0 = x.shape
        crop_h, crop_w, ph, pw = _tile_grid(h0, w0)
        x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), "reflect")
        n, c, h, w = x.shape
        out = torch.empty(
            (n, c, h * 2 - 72, w * 2 - 72), dtype=torch.float16, device=x.device
        )
        for i in range(0, h - 36, crop_h):
            for j in range(0, w - 36, crop_w):
                tile = x[:, :, i : i + crop_h + 36, j : j + crop_w + 36]
                tile_out = self._run(tile)
                out[:, :, i * 2 : i * 2 + crop_h * 2, j * 2 : j * 2 + crop_w * 2] = (
                    tile_out
                )
        return out

    def _post(self, out: "torch.Tensor", h0: int, w0: int) -> np.ndarray:
        """Crop the tiled output and quantize to uint8 RGB."""
        import torch

        x = out[:, :, : h0 * self._scale, : w0 * self._scale]
        x = (x * 255).round().clamp_(0, 255).byte()
        return np.transpose(x.squeeze(0).cpu().numpy(), (1, 2, 0))

    def _run(self, x: "torch.Tensor") -> "torch.Tensor":
        import contextlib
        import torch

        batch = x.shape[0]
        self._ensure_buffers(x.shape[2], x.shape[3], batch)
        stream_ctx = (
            torch.cuda.stream(self._stream)
            if isinstance(self._stream, torch.cuda.Stream)
            else contextlib.nullcontext()
        )
        stream_handle = getattr(self._stream, "cuda_stream", 0)
        if isinstance(self._stream, torch.cuda.Stream):
            # x is produced on the current (default) stream; the engine side
            # stream must wait for those kernels before copy_ reads it, or
            # tiles are intermittently corrupted mid-write.
            self._stream.wait_stream(torch.cuda.current_stream(self._device))
        with stream_ctx:
            self._in_buf.copy_(x)
            self._context.execute_async_v3(stream_handle=stream_handle)
        if hasattr(self._stream, "synchronize"):
            self._stream.synchronize()
        crop = {2: 72, 3: 84, 4: 152}[self._scale]
        out_h = x.shape[2] * self._scale - crop
        out_w = x.shape[3] * self._scale - crop
        return self._out_buf[:batch, :, :out_h, :out_w]

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        h0, w0 = frame.shape[:2]
        out = self._tiles(self._pre([frame]))
        return self._post(out, h0, w0)

    # Maximum number of tiles per engine call (profile batch max).
    BATCH_CAP = 2

    def enhance_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Enhance frames together, batching tiles into engine calls.

        All frames must share dimensions (true for video frames). Tiles are
        gathered from the frames and run in groups of ``BATCH_CAP`` per
        engine call, then assembled back per frame.
        """
        import torch
        from torch.nn import functional as F

        if not frames:
            return []
        h0, w0 = frames[0].shape[:2]
        n = len(frames)
        x = self._pre(frames)
        crop_h, crop_w, ph, pw = _tile_grid(h0, w0)
        x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), "reflect")
        n, c, h, w = x.shape
        out = torch.empty(
            (n, c, h * 2 - 72, w * 2 - 72), dtype=torch.float16, device=x.device
        )
        positions = [
            (i, j)
            for i in range(0, h - 36, crop_h)
            for j in range(0, w - 36, crop_w)
        ]
        tile_list = [
            (k, i, j)
            for k in range(n)
            for (i, j) in positions
        ]
        for start in range(0, len(tile_list), self.BATCH_CAP):
            group = tile_list[start : start + self.BATCH_CAP]
            batch = torch.cat(
                [
                    x[k, :, i : i + crop_h + 36, j : j + crop_w + 36].unsqueeze(0)
                    for (k, i, j) in group
                ],
                dim=0,
            )
            outs = self._run(batch)
            for (k, i, j), tile_out in zip(group, outs):
                out[k, :, i * 2 : i * 2 + crop_h * 2, j * 2 : j * 2 + crop_w * 2] = (
                    tile_out
                )
        return [self._post(out[k : k + 1], h0, w0) for k in range(n)]
