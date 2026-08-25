"""Task 5 tests for AMT backend selection and scheduled processing."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from video_upscaler import config
from video_upscaler.amt_scheduler import AMTFrameScheduler
import video_upscaler.interp as interp


def _select_amt_backend(*args, **kwargs):
    selector = getattr(interp, "select_amt_backend", None)
    if selector is None:
        pytest.fail("Task 5 AMT backend selection API is not implemented")
    return selector(*args, **kwargs)


def _selection(*args, **kwargs):
    selection_type = getattr(interp, "AMTBackendSelection", None)
    if selection_type is None:
        pytest.fail("Task 5 AMT backend selection API is not implemented")
    return selection_type(*args, **kwargs)


def _reset_amt_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "AMT_BACKEND_PREF", "auto", raising=False)
    monkeypatch.setattr(config, "AMT_BACKEND_EXPLICIT", False, raising=False)
    monkeypatch.setattr(config, "AMT_PRECISION", "fp32", raising=False)
    monkeypatch.setattr(config, "AMT_BATCH", "auto", raising=False)
    monkeypatch.setattr(config, "AMT_ENGINE_BUILD", True, raising=False)
    monkeypatch.setattr(config, "AMT_ENGINE_CACHE", True, raising=False)
    monkeypatch.setattr(config, "AMT_TRT_WORKSPACE_GIB", 1.0, raising=False)
    monkeypatch.setattr(config, "AMT_ONNX_OPSET", 17, raising=False)


def test_amt_config_exposes_task5_defaults() -> None:
    assert config.AMT_BACKEND_PREF in {"auto", "pytorch", "tensorrt"}
    assert config.AMT_PRECISION in {"fp32", "fp16"}
    assert config.AMT_BATCH == "auto" or int(config.AMT_BATCH) >= 1
    assert isinstance(config.AMT_ENGINE_BUILD, bool)
    assert isinstance(config.AMT_ENGINE_CACHE, bool)
    assert config.AMT_ONNX_OPSET == 17


def test_cli_backend_influences_amt_without_explicit_amt_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr(config, "BACKEND_PREF", "tensorrt")
    monkeypatch.setattr("video_upscaler.interp._cuda_available", lambda: True, raising=False)
    monkeypatch.setattr("video_upscaler.interp._tensorrt_available", lambda: True, raising=False)

    selection = _select_amt_backend("AMT-S")

    assert selection.backend == "tensorrt"
    assert selection.precision == "fp16"
    assert selection.batch_size == 2


def test_explicit_amt_backend_wins_over_upscaling_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr(config, "AMT_BACKEND_PREF", "pytorch")
    monkeypatch.setattr(config, "AMT_BACKEND_EXPLICIT", True)
    monkeypatch.setattr(config, "BACKEND_PREF", "tensorrt")

    selection = _select_amt_backend("AMT-S")

    assert selection.backend == "pytorch"
    assert selection.fallback_reason is None


def test_auto_falls_back_to_pytorch_with_visible_runtime_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr("video_upscaler.interp._cuda_available", lambda: True, raising=False)
    monkeypatch.setattr("video_upscaler.interp._tensorrt_available", lambda: False, raising=False)

    selection = _select_amt_backend("AMT-S")

    assert selection.backend == "pytorch"
    assert "TensorRT" in (selection.fallback_reason or "")


def test_auto_tensorrt_dll_load_failure_is_a_pytorch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr("video_upscaler.interp._cuda_available", lambda: True)
    monkeypatch.setattr(
        "video_upscaler.tensorrt_backend.tensorrt_available",
        lambda: (_ for _ in ()).throw(OSError("nvinfer_11.dll missing")),
    )

    selection = _select_amt_backend("AMT-S")

    assert selection.backend == "pytorch"
    assert "TensorRT" in (selection.fallback_reason or "")


def test_explicit_tensorrt_dll_load_failure_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr(config, "AMT_BACKEND_PREF", "tensorrt")
    monkeypatch.setattr(config, "AMT_BACKEND_EXPLICIT", True)
    monkeypatch.setattr("video_upscaler.interp._cuda_available", lambda: True)
    monkeypatch.setattr(
        "video_upscaler.tensorrt_backend.tensorrt_available",
        lambda: (_ for _ in ()).throw(OSError("nvinfer_11.dll missing")),
    )

    with pytest.raises(RuntimeError, match="TensorRT.*unavailable|DLL|nvinfer"):
        _select_amt_backend("AMT-S")


def test_auto_tensorrt_build_failure_retries_video_with_pytorch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from video_upscaler import processor

    trt_selection = _selection("tensorrt", "fp16", 2, None)
    torch_selection = _selection("pytorch", "fp32", 2, None)
    build_calls: list[str] = []

    class FakeBackend:
        name = "pytorch"
        precision = "fp32"
        device = "cpu"

        def interpolate(self, frames, niters):
            raise AssertionError("legacy pair loop was used")

    class FakeScheduler:
        def interpolate_window(self, frames, niters, backend, batch_size):
            return iter(frames)

    class FakeFactory:
        def __init__(self, selection):
            self.selection = selection

        def build(self, shape=None):
            build_calls.append(self.selection.backend)
            if self.selection.backend == "tensorrt":
                raise RuntimeError("profile build failed")
            return FakeBackend()

        def close(self):
            pass

    def make_factory(model_key, selection=None):
        return FakeFactory(selection or trt_selection)

    monkeypatch.setattr(processor, "_make_amt_backend_factory", make_factory)
    monkeypatch.setattr(processor, "AMTFrameScheduler", FakeScheduler, raising=False)
    monkeypatch.setattr(processor, "probe", lambda path: {
        "width": 2, "height": 2, "fps": 30.0, "duration": 0.1,
        "has_audio": False, "rotation": 0,
    })
    monkeypatch.setattr(processor, "decode_frames", lambda path: iter([bytes(12)] * 2))
    monkeypatch.setattr(processor, "encode_video", lambda frames, *args, **kwargs: list(frames))
    monkeypatch.setattr(processor.config, "OUTPUT_DIR", tmp_path)

    result = processor.process_interpolate(
        [tmp_path / "clip.mp4"], "AMT-S", 2, lambda *args: None
    )

    assert result["failed"] == []
    assert build_calls == ["tensorrt", "pytorch"]
    assert "falling back to PyTorch" in capsys.readouterr().out


def test_cpu_pytorch_fp16_downgrades_to_fp32_with_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr(config, "AMT_BACKEND_PREF", "pytorch")
    monkeypatch.setattr(config, "AMT_BACKEND_EXPLICIT", True)
    monkeypatch.setattr(config, "AMT_PRECISION", "fp16")
    monkeypatch.setattr("video_upscaler.interp._cuda_available", lambda: False)

    selection = _select_amt_backend("AMT-S")

    assert selection.backend == "pytorch"
    assert selection.precision == "fp32"
    assert "FP16" in (selection.fallback_reason or "")


def test_ncnn_cli_backend_keeps_amt_on_pytorch_without_amt_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr(config, "BACKEND_PREF", "ncnn")
    monkeypatch.setattr("video_upscaler.interp._cuda_available", lambda: True)
    monkeypatch.setattr("video_upscaler.interp._tensorrt_available", lambda: True)

    selection = _select_amt_backend("AMT-S")

    assert selection.backend == "pytorch"
    assert "ncnn" in (selection.fallback_reason or "")


def test_explicit_tensorrt_reports_actionable_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr(config, "AMT_BACKEND_PREF", "tensorrt")
    monkeypatch.setattr(config, "AMT_BACKEND_EXPLICIT", True)
    monkeypatch.setattr("video_upscaler.interp._cuda_available", lambda: False, raising=False)

    with pytest.raises(RuntimeError, match="CUDA|TensorRT"):
        _select_amt_backend("AMT-S")


def test_amt_l_and_g_select_tensorrt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr("video_upscaler.interp._cuda_available", lambda: True, raising=False)
    monkeypatch.setattr("video_upscaler.interp._tensorrt_available", lambda: True, raising=False)

    for model_key in ("AMT-L", "AMT-G"):
        selection = _select_amt_backend(model_key)
        assert selection.backend == "tensorrt"
        assert selection.precision == "fp16"


def test_explicit_tensorrt_accepts_all_amt_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr(config, "AMT_BACKEND_PREF", "tensorrt")
    monkeypatch.setattr(config, "AMT_BACKEND_EXPLICIT", True)
    monkeypatch.setattr("video_upscaler.interp._cuda_available", lambda: True)
    monkeypatch.setattr("video_upscaler.interp._tensorrt_available", lambda: True)

    for model_key in ("AMT-S", "AMT-L", "AMT-G"):
        assert _select_amt_backend(model_key).backend == "tensorrt"

    with pytest.raises(ValueError, match="unknown AMT model"):
        _select_amt_backend("AMT-X")


def test_batch_auto_defaults_per_model_and_honors_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr("video_upscaler.interp._cuda_available", lambda: True, raising=False)
    monkeypatch.setattr("video_upscaler.interp._tensorrt_available", lambda: True, raising=False)
    assert _select_amt_backend("AMT-S").batch_size == 2
    assert _select_amt_backend("AMT-L").batch_size == 1
    assert _select_amt_backend("AMT-G").batch_size == 1

    monkeypatch.setattr(config, "AMT_BATCH", "1")
    assert _select_amt_backend("AMT-S").batch_size == 1

    monkeypatch.setattr(config, "AMT_BATCH", "2")
    assert _select_amt_backend("AMT-L").batch_size == 2


def test_profile_resolves_adaptive_scale_and_padded_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("video_upscaler.interp._adaptive_amt_scale", lambda h, w: 0.5, raising=False)

    resolver = getattr(interp, "resolve_amt_profile", None)
    if resolver is None:
        pytest.fail("Task 5 AMT profile resolution API is not implemented")
    profile = resolver(127, 129)

    assert profile.scale == 0.5
    assert profile.padded_height == 128
    assert profile.padded_width == 160


def test_tensorrt_factory_exports_validates_and_constructs_once_per_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _reset_amt_config(monkeypatch)
    selection = _selection("tensorrt", "fp16", 2, None)
    exported: list[tuple] = []
    constructed: list[tuple] = []
    onnx_path = tmp_path / "amt-pair.onnx"
    onnx_path.write_bytes(b"onnx")

    monkeypatch.setattr(
        "video_upscaler.interp.export_amt_pair",
        lambda *args: exported.append(args) or onnx_path,
        raising=False,
    )
    monkeypatch.setattr(
        "video_upscaler.interp.validate_amt_onnx",
        lambda path: {"path": str(path), "checker_passed": True},
        raising=False,
    )
    monkeypatch.setattr(
        "video_upscaler.interp.AMTTensorRTEngine",
        lambda spec, path, **kwargs: constructed.append((spec, path, kwargs)) or object(),
        raising=False,
    )
    factory_type = getattr(interp, "AMTBackendFactory", None)
    if factory_type is None:
        pytest.fail("Task 5 AMT backend factory API is not implemented")
    factory = factory_type("AMT-S", selection, tmp_path)

    first = factory.build((127, 129))
    second = factory.build((127, 129))

    assert first is second
    assert len(exported) == 1
    assert len(constructed) == 1
    assert constructed[0][1] == onnx_path
    assert constructed[0][0].max_batch == 2


def test_engine_build_disabled_requires_existing_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr(config, "AMT_ENGINE_BUILD", False, raising=False)
    selection = _selection("tensorrt", "fp16", 2, None)
    factory_type = getattr(interp, "AMTBackendFactory", None)
    if factory_type is None:
        pytest.fail("Task 5 AMT backend factory API is not implemented")
    monkeypatch.setattr(
        "video_upscaler.interp.amt_engine_path",
        lambda spec: tmp_path / "missing.engine",
        raising=False,
    )
    monkeypatch.setattr(
        "video_upscaler.interp._tensorrt_onnx_path",
        lambda spec: tmp_path / "pair.onnx",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="build"):
        factory_type("AMT-S", selection, tmp_path).build((128, 128))


def test_engine_cache_disabled_is_forwarded_to_tensorrt_constructor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _reset_amt_config(monkeypatch)
    monkeypatch.setattr(config, "AMT_ENGINE_CACHE", False, raising=False)
    selection = _selection("tensorrt", "fp16", 2, None)
    engine_path = tmp_path / "profile.engine"
    onnx_path = tmp_path / "profile.onnx"
    onnx_path.write_bytes(b"onnx")
    observed: list[dict] = []

    monkeypatch.setattr(
        "video_upscaler.interp.amt_engine_path",
        lambda spec: engine_path,
        raising=False,
    )
    monkeypatch.setattr(
        "video_upscaler.interp._tensorrt_onnx_path",
        lambda spec: onnx_path,
        raising=False,
    )
    monkeypatch.setattr(
        "video_upscaler.interp.export_amt_pair",
        lambda *args: onnx_path,
        raising=False,
    )
    monkeypatch.setattr(
        "video_upscaler.interp.validate_amt_onnx",
        lambda path: {},
        raising=False,
    )
    monkeypatch.setattr(
        "video_upscaler.interp.AMTTensorRTEngine",
        lambda spec, path, **kwargs: observed.append(kwargs) or object(),
        raising=False,
    )
    factory_type = getattr(interp, "AMTBackendFactory", None)
    if factory_type is None:
        pytest.fail("Task 5 AMT backend factory API is not implemented")

    factory_type("AMT-S", selection, tmp_path).build((128, 128))

    assert observed == [{"use_cache": False}]


def test_processor_routes_windows_through_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from video_upscaler import processor

    calls: list[tuple] = []

    class FakeBackend:
        name = "pytorch"
        precision = "fp32"
        device = "cpu"

        def interpolate(self, frames, niters):
            raise AssertionError("legacy pair loop was used")

    class FakeScheduler:
        def interpolate_window(self, frames, niters, backend, batch_size):
            calls.append((len(frames), niters, backend, batch_size))
            return iter(frames)

    monkeypatch.setattr(processor, "AMTFrameScheduler", FakeScheduler, raising=False)
    monkeypatch.setattr(processor, "_make_amt_backend_factory", lambda model, selection=None: SimpleNamespace(
        selection=_selection("pytorch", "fp32", 2, None),
        build=lambda shape=None: FakeBackend(),
    ))
    monkeypatch.setattr(processor, "probe", lambda path: {
        "width": 2, "height": 2, "fps": 30.0, "duration": 0.1,
        "has_audio": False, "rotation": 0,
    })
    monkeypatch.setattr(processor, "decode_frames", lambda path: iter([bytes(12)] * 2))
    monkeypatch.setattr(processor, "encode_video", lambda frames, *args, **kwargs: list(frames))
    monkeypatch.setattr(processor.config, "OUTPUT_DIR", tmp_path)

    result = processor.process_interpolate([tmp_path / "clip.mp4"], "AMT-S", 2, lambda *args: None)

    assert result["failed"] == []
    assert calls and calls[0][1] == 1 and calls[0][3] == 2


def test_processor_warms_up_each_backend_once_per_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from video_upscaler import processor

    warmups: list[tuple[tuple[int, int], int]] = []

    class WarmableBackend:
        name = "pytorch"
        precision = "fp32"
        device = "cpu"

        def interpolate(self, frames, niters):
            return iter(frames)

        def warmup(self, shape: tuple[int, int], batch_size: int) -> None:
            warmups.append((shape, batch_size))

    class FakeScheduler:
        def interpolate_window(self, frames, niters, backend, batch_size):
            return iter(frames)

    monkeypatch.setattr(processor, "AMTFrameScheduler", FakeScheduler, raising=False)
    monkeypatch.setattr(
        processor,
        "_make_amt_backend_factory",
        lambda model, selection=None: SimpleNamespace(
            selection=_selection("pytorch", "fp32", 2, None),
            build=lambda shape=None: WarmableBackend(),
        ),
    )
    monkeypatch.setattr(processor, "probe", lambda path: {
        "width": 2, "height": 2, "fps": 30.0, "duration": 0.1,
        "has_audio": False, "rotation": 0,
    })
    monkeypatch.setattr(processor, "decode_frames", lambda path: iter([bytes(12)] * 2))
    monkeypatch.setattr(processor, "encode_video", lambda frames, *args, **kwargs: list(frames))
    monkeypatch.setattr(processor.config, "OUTPUT_DIR", tmp_path)

    result = processor.process_interpolate(
        [tmp_path / "clip.mp4", tmp_path / "clip2.mp4"], "AMT-S", 2, lambda *args: None
    )

    assert result["failed"] == []
    assert warmups == [((2, 2), 2)]


def test_exact_boundary_recovery_uses_scheduler_not_legacy_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_upscaler.processor import _interp_window_stream

    calls: list[int] = []

    class SchedulerOnly:
        def interpolate_window(self, frames, niters, backend, batch_size):
            calls.append(len(frames))
            return iter(frames)

    class NoLegacyEngine:
        def interpolate(self, frames, niters):
            raise AssertionError("boundary recovery used legacy interpolation")

    raw_frames = (np.zeros((2, 2, 3), dtype=np.uint8).tobytes() for _ in range(4))
    output = list(
        _interp_window_stream(
            raw_frames,
            NoLegacyEngine(),
            1,
            4,
            (2, 2),
            scheduler=SchedulerOnly(),
            batch_size=2,
        )
    )

    assert len(output) == 4
    assert calls == [4, 1]


def test_benchmark_uses_scheduler_backend_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from video_upscaler import amt_benchmark

    calls: list[int] = []

    class FakeBackend:
        device = "cpu"
        name = "pytorch"
        precision = "fp32"

        def prepare_frames(self, frames):
            return frames

        def infer_batch(self, frame_a, frame_b):
            return frame_a

        def transfer_batch_to_host(self, output):
            return output

        def finalize_frames(self, frames):
            return frames

    monkeypatch.setattr(amt_benchmark, "AMTFrameScheduler", lambda: SimpleNamespace(
        interpolate_window=lambda frames, niters, backend, batch_size: calls.append(batch_size) or iter(frames)
    ), raising=False)

    output = amt_benchmark._scheduled_inference(
        FakeBackend(), [np.zeros((2, 2, 3), dtype=np.uint8)], 1, 2
    )

    assert list(output)
    assert calls == [2]
