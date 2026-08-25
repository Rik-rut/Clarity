"""Tests for exporting a fixed AMT-S/AMT-L/AMT-G frame pair to ONNX."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from video_upscaler.amt.networks.amt_s import Model as AMT_S
from video_upscaler.amt_export import (
    AMTPairWrapper,
    _validate_export_args,
    compare_amt_pair,
    export_amt_pair,
    validate_amt_onnx,
)
from video_upscaler.interp import _MODEL_PARAMS


class _RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed: tuple[torch.Tensor, torch.Tensor, float, float, bool] | None = None

    def forward(self, frame_a, frame_b, embt, *, scale_factor, eval):
        self.observed = (
            frame_a,
            frame_b,
            float(embt[0, 0, 0, 0]),
            float(scale_factor),
            eval,
        )
        return {"imgt_pred": (frame_a + frame_b) / 2}


@pytest.fixture
def amt_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    from video_upscaler import config

    checkpoint = tmp_path / "amt-s.pth"
    model = AMT_S(**_MODEL_PARAMS["AMT-S"])
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    return checkpoint


def test_pair_wrapper_uses_fixed_time_and_scale() -> None:
    model = _RecordingModel()
    wrapper = AMTPairWrapper(model, scale=0.5)
    frame_a = torch.zeros(2, 3, 16, 16)
    frame_b = torch.ones(2, 3, 16, 16)

    output = wrapper(frame_a, frame_b)

    assert output.shape == frame_a.shape
    assert model.observed is not None
    assert model.observed[2:] == (0.5, 0.5, True)
    assert not model.training


def test_export_writes_valid_graph_and_metadata(amt_checkpoint: Path, tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    output_path = tmp_path / "amt-s-pair.onnx"

    result = export_amt_pair("AMT-S", 128, 128, 1.0, output_path, 17)

    assert result == output_path
    assert output_path.is_file()
    metadata_path = Path(f"{output_path}.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_key"] == "AMT-S"
    assert metadata["checkpoint_sha256"] == hashlib.sha256(
        amt_checkpoint.read_bytes()
    ).hexdigest()
    assert metadata["input_shapes"] == [
        ["batch", 3, 128, 128],
        ["batch", 3, 128, 128],
    ]
    assert metadata["output_shape"] == ["batch", 3, 128, 128]
    assert metadata["internal_scale"] == 1.0
    assert metadata["opset"] == 17
    assert metadata["pytorch_version"] == torch.__version__
    assert metadata["onnx_version"]
    assert metadata["export_version"]
    comparison = metadata["numerical_comparison"]
    try:
        import onnxruntime as ort
    except ImportError:
        has_provider = False
    else:
        has_provider = any(
            provider in {"CUDAExecutionProvider", "CPUExecutionProvider"}
            for provider in ort.get_available_providers()
        )
    if has_provider:
        assert comparison["status"] == "measured"
        assert comparison["reference_finite"] is True
        assert comparison["onnx_finite"] is True
        assert math.isfinite(comparison["max_absolute_error"])
        assert math.isfinite(comparison["mean_absolute_error"])
    else:
        assert comparison["status"] == "skipped"
    if comparison["status"] == "measured":
        assert comparison["max_absolute_error"] >= 0.0
        assert comparison["mean_absolute_error"] >= 0.0

    summary = validate_amt_onnx(output_path)
    assert summary["checker_passed"] is True
    assert summary["shape_inference_passed"] is True
    assert summary["inputs"][0]["shape"] == ["batch", 3, 128, 128]
    assert summary["outputs"][0]["shape"] == ["batch", 3, 128, 128]


@pytest.mark.parametrize(
    ("model_key", "height", "width", "scale", "opset", "message"),
    [
        ("invalid", 128, 128, 1.0, 17, "model_key"),
        ("AMT-X", 128, 128, 1.0, 17, "model_key"),
        ("AMT-S", 64, 64, 1.0, 17, "at least 128"),
        ("AMT-S", 128, 0, 1.0, 17, "positive"),
        ("AMT-S", 128, 128, 1.0, 10, "opset"),
        ("AMT-S", 129, 128, 1.0, 17, "internal dimensions"),
    ],
)
def test_export_rejects_invalid_profile(
    model_key: str,
    height: int,
    width: int,
    scale: float,
    opset: int,
    message: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=message):
        export_amt_pair(model_key, height, width, scale, tmp_path / "out.onnx", opset)


@pytest.mark.parametrize("model_key", ["AMT-L", "AMT-G"])
def test_export_accepts_amt_profile(model_key: str) -> None:
    _validate_export_args(model_key, 1100, 1925, 0.64, 17)


def test_export_accepts_padded_raw_shape_when_internal_shape_is_safe() -> None:
    _validate_export_args("AMT-S", 1100, 1925, 0.64, 17)


def test_failed_ort_comparison_writes_diagnostics_and_raises(
    amt_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("onnx")
    import video_upscaler.amt_export as export_module

    output_path = tmp_path / "amt-s-pair.onnx"
    failure = {
        "status": "failed",
        "provider": "CPUExecutionProvider",
        "reason": "comparison output contains non-finite values",
    }
    monkeypatch.setattr(export_module, "compare_amt_pair", lambda *args: failure)

    with pytest.raises(RuntimeError, match="ONNX Runtime comparison failed"):
        export_amt_pair("AMT-S", 128, 128, 1.0, output_path, 17)

    metadata = json.loads(Path(f"{output_path}.json").read_text(encoding="utf-8"))
    assert metadata["numerical_comparison"] == failure


def test_tensorrt_parser_accepts_128_graph(
    amt_checkpoint: Path, tmp_path: Path
) -> None:
    pytest.importorskip("onnx")
    from video_upscaler.tensorrt_backend import prepare_env

    error = prepare_env()
    if error:
        pytest.skip(error)
    try:
        import tensorrt as trt
    except (ImportError, FileNotFoundError, OSError) as exc:
        pytest.skip(f"TensorRT runtime unavailable: {exc}")

    output_path = tmp_path / "amt-s-pair.onnx"
    export_amt_pair("AMT-S", 128, 128, 1.0, output_path, 17)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    assert parser.parse_from_file(str(output_path)), " | ".join(
        str(parser.get_error(index)) for index in range(parser.num_errors)
    )


def test_compare_helper_reports_shape_failure_as_json_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ShapeMismatchSession:
        def run(self, output_names, inputs):
            return [torch.zeros(1, 3, 1, 128).numpy()]

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda path, providers: _ShapeMismatchSession(),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    result = compare_amt_pair(
        Path("missing.onnx"),
        AMTPairWrapper(_RecordingModel(), scale=1.0),
        torch.zeros(1, 3, 128, 128),
        torch.ones(1, 3, 128, 128),
    )

    assert result["status"] == "failed"
    assert result["provider"] == "CPUExecutionProvider"
    assert "shape mismatch" in result["reason"]


def test_nonfinite_reference_writes_diagnostics_without_ort(
    amt_checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("onnx")
    import video_upscaler.amt_export as export_module

    class _NonFiniteModel(torch.nn.Module):
        def forward(self, frame_a, frame_b, embt, *, scale_factor, eval):
            return {"imgt_pred": torch.full_like(frame_a, float("nan"))}

    output_path = tmp_path / "amt-s-pair.onnx"
    monkeypatch.setattr(
        export_module,
        "_load_amt_model",
        lambda model_key, checkpoint_path: _NonFiniteModel(),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(RuntimeError, match="PyTorch reference output"):
        export_amt_pair("AMT-S", 128, 128, 1.0, output_path, 17)

    metadata = json.loads(Path(f"{output_path}.json").read_text(encoding="utf-8"))
    assert metadata["numerical_comparison"]["status"] == "failed"
    assert metadata["numerical_comparison"]["reference_finite"] is False
    assert not output_path.exists()


def test_compare_helper_reports_optional_runtime_status(
    amt_checkpoint: Path, tmp_path: Path
) -> None:
    pytest.importorskip("onnx")
    output_path = tmp_path / "amt-s-pair.onnx"
    export_amt_pair("AMT-S", 128, 128, 1.0, output_path, 17)
    model = AMT_S(**_MODEL_PARAMS["AMT-S"])
    checkpoint = torch.load(amt_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    wrapper = AMTPairWrapper(model.eval(), scale=1.0)
    result = compare_amt_pair(
        output_path,
        wrapper,
        torch.zeros(1, 3, 128, 128),
        torch.ones(1, 3, 128, 128),
    )
    assert result["status"] in {"measured", "skipped", "failed"}
    if result["status"] == "measured":
        assert result["max_absolute_error"] >= 0.0
        assert result["mean_absolute_error"] >= 0.0
