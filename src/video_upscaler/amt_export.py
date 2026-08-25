"""Export and validate the fixed AMT-S/AMT-L/AMT-G frame-pair graph."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from video_upscaler import config
from video_upscaler.interp import _MODEL_CLASSES, _MODEL_PARAMS
from video_upscaler.models import ckpt_for_interp_model

EXPORT_VERSION = 1
# Models with a validated TensorRT FP16 export path.
SUPPORTED_MODEL_KEYS = frozenset({"AMT-S", "AMT-L", "AMT-G"})
_MIN_INTERNAL_DIMENSION = 128
_INTERNAL_FEATURE_DIVISOR = 16


class AMTPairWrapper(nn.Module):
    """Expose AMT's fixed midpoint pair inference as one tensor operation."""

    def __init__(self, model: nn.Module, scale: float) -> None:
        super().__init__()
        self.model = model.eval()
        self.scale = float(scale)

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        if frame_a.shape != frame_b.shape:
            raise ValueError("AMT pair tensors must have matching shapes")
        if frame_a.ndim != 4 or frame_a.shape[1] != 3:
            raise ValueError("AMT pair tensors must have shape [B, 3, H, W]")

        # Construct embt from the input so only the batch dimension remains dynamic.
        embt = frame_a[:, :1, :1, :1].new_ones((frame_a.shape[0], 1, 1, 1)) * 0.5
        result = self.model(
            frame_a,
            frame_b,
            embt,
            scale_factor=self.scale,
            eval=True,
        )
        return result["imgt_pred"]


def _checkpoint_path(model_key: str) -> Path:
    if model_key not in SUPPORTED_MODEL_KEYS:
        raise ValueError(
            f"model_key must be one of {sorted(SUPPORTED_MODEL_KEYS)}, got {model_key!r}"
        )
    return config.MODELS_DIR / ckpt_for_interp_model(model_key)


def _validate_export_args(model_key: str, height: int, width: int, scale: float, opset: int) -> None:
    _checkpoint_path(model_key)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not 0 < scale <= 1:
        raise ValueError("scale must be greater than 0 and no greater than 1")

    internal_height = math.floor(height * scale)
    internal_width = math.floor(width * scale)
    if min(internal_height, internal_width) < _MIN_INTERNAL_DIMENSION:
        raise ValueError(
            "scaled internal dimensions must be at least 128x128; "
            f"got {internal_height}x{internal_width}"
        )
    if (
        internal_height % _INTERNAL_FEATURE_DIVISOR
        or internal_width % _INTERNAL_FEATURE_DIVISOR
    ):
        raise ValueError(
            "scaled internal dimensions must be divisible by 16 for the "
            f"AMT feature hierarchy; got {internal_height}x{internal_width}"
        )
    if opset < 11:
        raise ValueError("opset must be at least 11")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_amt_model(model_key: str, checkpoint_path: Path) -> nn.Module:
    model = _MODEL_CLASSES[model_key](**_MODEL_PARAMS[model_key])
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def _require_onnx():
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires the optional 'tensorrt' dependencies; "
            "install them with `uv sync --extra tensorrt`."
        ) from exc
    return onnx


def _fixed_pair(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.linspace(0.0, 1.0, steps=3 * height * width).reshape(
        1, 3, height, width
    )
    return values, 1.0 - values


def _set_static_output_shape(path: Path, output_shape: tuple[int, ...]) -> None:
    onnx = _require_onnx()
    model = onnx.load(str(path))
    dimensions = model.graph.output[0].type.tensor_type.shape.dim
    dimensions[0].ClearField("dim_value")
    dimensions[0].dim_param = "batch"
    for dimension, size in zip(dimensions[1:], output_shape[1:]):
        dimension.ClearField("dim_param")
        dimension.dim_value = int(size)
    onnx.save(model, str(path))


def export_amt_pair(
    model_key: str,
    height: int,
    width: int,
    scale: float,
    output_path: Path,
    opset: int,
) -> Path:
    """Export one fixed-spatial AMT-S/AMT-L/AMT-G pair graph and adjacent metadata."""
    _validate_export_args(model_key, height, width, scale, opset)
    checkpoint_path = _checkpoint_path(model_key)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"AMT checkpoint not found: {checkpoint_path}")

    onnx = _require_onnx()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = _load_amt_model(model_key, checkpoint_path)
    wrapper = AMTPairWrapper(model, scale)
    frame_a, frame_b = _fixed_pair(height, width)

    with torch.inference_mode():
        reference = wrapper(frame_a, frame_b)
    reference_finite = bool(torch.isfinite(reference).all().item())
    if not reference_finite:
        metadata = {
            "model_key": model_key,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "input_shapes": [
                ["batch", 3, height, width],
                ["batch", 3, height, width],
            ],
            "output_shape": ["batch", 3, *reference.shape[2:]],
            "internal_scale": float(scale),
            "opset": int(opset),
            "pytorch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "export_version": EXPORT_VERSION,
            "onnx_checker_passed": False,
            "shape_inference_passed": False,
            "reference_output_shape": list(reference.shape),
            "reference_finite": False,
            "numerical_comparison": {
                "status": "failed",
                "provider": None,
                "reason": "PyTorch reference output contains non-finite values",
                "reference_finite": False,
            },
        }
        Path(f"{output_path}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise RuntimeError(
            "PyTorch reference output contains non-finite values; see "
            f"{output_path}.json for diagnostics"
        )

    torch.onnx.export(
        wrapper,
        (frame_a, frame_b),
        str(output_path),
        input_names=["frame_a", "frame_b"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={
            "frame_a": {0: "batch"},
            "frame_b": {0: "batch"},
            "output": {0: "batch"},
        },
        dynamo=False,
    )

    _set_static_output_shape(output_path, tuple(reference.shape))
    summary = validate_amt_onnx(output_path)
    comparison = compare_amt_pair(output_path, wrapper, frame_a, frame_b)
    metadata = {
        "model_key": model_key,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "input_shapes": [list(item["shape"]) for item in summary["inputs"]],
        "output_shape": list(summary["outputs"][0]["shape"]),
        "internal_scale": float(scale),
        "opset": int(opset),
        "pytorch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "export_version": EXPORT_VERSION,
        "onnx_checker_passed": summary["checker_passed"],
        "shape_inference_passed": summary["shape_inference_passed"],
        "reference_output_shape": list(reference.shape),
        "reference_finite": reference_finite,
        "numerical_comparison": comparison,
    }
    Path(f"{output_path}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if comparison.get("status") == "failed":
        raise RuntimeError(
            "ONNX Runtime comparison failed; see "
            f"{output_path}.json for diagnostics: {comparison.get('reason', 'unknown reason')}"
        )
    return output_path


def _shape_from_value_info(value_info: Any) -> list[int | str | None]:
    tensor_type = value_info.type.tensor_type
    shape: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.dim_param:
            shape.append(dimension.dim_param)
        elif dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        else:
            shape.append(None)
    return shape


def validate_amt_onnx(path: Path) -> dict:
    """Run ONNX checker and shape inference, returning JSON-compatible details."""
    onnx = _require_onnx()
    path = Path(path)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)

    def describe(value_info: Any) -> dict[str, Any]:
        return {
            "name": value_info.name,
            "shape": _shape_from_value_info(value_info),
            "element_type": int(value_info.type.tensor_type.elem_type),
        }

    opsets = [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")]
    return {
        "path": str(path),
        "checker_passed": True,
        "shape_inference_passed": True,
        "ir_version": int(model.ir_version),
        "opset": int(opsets[0]) if opsets else None,
        "inputs": [describe(value) for value in inferred.graph.input],
        "outputs": [describe(value) for value in inferred.graph.output],
    }


def compare_amt_pair(
    path: Path,
    wrapper: AMTPairWrapper,
    frame_a: torch.Tensor,
    frame_b: torch.Tensor,
) -> dict[str, Any]:
    """Compare one fixed pair when an executable ONNX Runtime provider exists."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"status": "skipped", "reason": "onnxruntime is not installed"}

    available = ort.get_available_providers()
    provider = next(
        (name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available),
        None,
    )
    if provider is None:
        return {"status": "skipped", "reason": "no supported ONNX Runtime provider"}

    try:
        session = ort.InferenceSession(str(path), providers=[provider])
        with torch.inference_mode():
            expected = wrapper(frame_a, frame_b).detach().cpu().numpy()
        actual = session.run(
            ["output"],
            {
                "frame_a": frame_a.detach().cpu().numpy(),
                "frame_b": frame_b.detach().cpu().numpy(),
            },
        )[0]

        if expected.shape != actual.shape:
            raise ValueError(
                "output shape mismatch: "
                f"PyTorch {list(expected.shape)} != ONNX Runtime {list(actual.shape)}"
            )

        reference_finite = bool(np.isfinite(expected).all())
        onnx_finite = bool(np.isfinite(actual).all())
        if not reference_finite or not onnx_finite:
            raise ValueError(
                "comparison output contains non-finite values "
                f"(reference_finite={reference_finite}, onnx_finite={onnx_finite})"
            )

        difference = np.abs(expected - actual)
        return {
            "status": "measured",
            "provider": provider,
            "reference_finite": reference_finite,
            "onnx_finite": onnx_finite,
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
        }
    except Exception as exc:
        return {"status": "failed", "reason": str(exc), "provider": provider}
