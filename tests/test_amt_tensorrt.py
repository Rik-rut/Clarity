"""Tests for the AMT-S TensorRT FP16 engine and cache."""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from video_upscaler.amt_tensorrt import (
    AMTEngineSpec,
    AMTTensorRTEngine,
    amt_engine_path,
    ensure_amt_engine,
)


def _spec() -> AMTEngineSpec:
    return AMTEngineSpec(
        model_key="AMT-S",
        padded_height=128,
        padded_width=128,
        scale=1.0,
        precision="fp16",
        max_batch=2,
        opset=17,
        export_version=1,
    )


def test_spec_accepts_only_supported_amt_fp16_profiles() -> None:
    assert _spec().max_batch == 2
    # AMT-S, AMT-L, and AMT-G all have validated TensorRT paths.
    AMTEngineSpec("AMT-L", 128, 128, 1.0, "fp16", 2, 17, 1)
    AMTEngineSpec("AMT-G", 128, 128, 1.0, "fp16", 2, 17, 1)
    with pytest.raises(ValueError, match="AMT-S|AMT-L|AMT-G|model_key"):
        AMTEngineSpec("AMT-X", 128, 128, 1.0, "fp16", 2, 17, 1)
    with pytest.raises(ValueError, match="fp16"):
        AMTEngineSpec("AMT-S", 128, 128, 1.0, "fp32", 2, 17, 1)
    with pytest.raises(ValueError, match="padded"):
        AMTEngineSpec("AMT-S", 127, 128, 1.0, "fp16", 2, 17, 1)


def test_engine_path_is_deterministic_and_separate_from_real_cugan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import amt_tensorrt as at

    monkeypatch.setattr(at, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(at, "_gpu_name", lambda: "NVIDIA_GeForce_RTX_3050")

    first = amt_engine_path(_spec())
    second = amt_engine_path(_spec())

    assert first == second
    assert first.parent == tmp_path / "amt" / "tensorrt"
    assert first.name == "AMT-S_128x128_s1_fp16_b2_o17_e1_NVIDIA_GeForce_RTX_3050.engine"
    assert first.parent != tmp_path / "tensorrt"


def test_matching_cache_metadata_reuses_engine_without_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import amt_tensorrt as at

    monkeypatch.setattr(at, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(at, "prepare_env", lambda: None)
    monkeypatch.setattr(at, "_runtime_metadata", lambda: {
        "gpu": "GPU_X",
        "compute_capability": "8.6",
        "cuda": "12.9",
        "tensorrt": "11.1",
    })
    checkpoint = tmp_path / "amt-s.pth"
    checkpoint.write_bytes(b"checkpoint")
    onnx_path = tmp_path / "pair.onnx"
    onnx_path.write_bytes(b"onnx")
    monkeypatch.setattr(at, "_checkpoint_path", lambda model_key: checkpoint)

    engine = amt_engine_path(_spec())
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"engine")
    metadata = at._cache_metadata(_spec(), onnx_path)
    engine.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    built: list[Path] = []
    monkeypatch.setattr(at, "_build_amt_engine", lambda spec, path: built.append(path) or engine)

    assert ensure_amt_engine(_spec(), onnx_path) == engine
    assert built == []


@pytest.mark.parametrize("changed", ["checkpoint", "onnx", "shape", "runtime"])
def test_cache_invalidation_rebuilds_when_any_compatibility_value_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str
) -> None:
    from video_upscaler import amt_tensorrt as at

    monkeypatch.setattr(at, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(at, "prepare_env", lambda: None)
    runtime = {
        "gpu": "GPU_X",
        "compute_capability": "8.6",
        "cuda": "12.9",
        "tensorrt": "11.1",
    }
    monkeypatch.setattr(at, "_runtime_metadata", lambda: runtime)
    checkpoint = tmp_path / "amt-s.pth"
    checkpoint.write_bytes(b"checkpoint")
    onnx_path = tmp_path / "pair.onnx"
    onnx_path.write_bytes(b"onnx")
    monkeypatch.setattr(at, "_checkpoint_path", lambda model_key: checkpoint)
    engine = amt_engine_path(_spec())
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"engine")
    metadata = at._cache_metadata(_spec(), onnx_path)
    if changed == "checkpoint":
        checkpoint.write_bytes(b"new checkpoint")
    elif changed == "onnx":
        onnx_path.write_bytes(b"new onnx")
    elif changed == "shape":
        metadata["padded_height"] = 256
    else:
        runtime["cuda"] = "13.0"
    engine.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    rebuilt: list[Path] = []
    monkeypatch.setattr(at, "_build_amt_engine", lambda spec, path: rebuilt.append(path) or engine)

    ensure_amt_engine(_spec(), onnx_path)
    assert rebuilt == [onnx_path]


def test_missing_runtime_raises_before_tensor_rt_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import amt_tensorrt as at

    monkeypatch.setattr(at, "prepare_env", lambda: "TensorRT install not found")

    with pytest.raises(RuntimeError, match="TensorRT install not found"):
        ensure_amt_engine(_spec(), tmp_path / "pair.onnx")


def test_backend_constructor_uses_external_onnx_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import amt_tensorrt as at

    external_onnx = tmp_path / "exports" / "amt-pair.onnx"
    external_onnx.parent.mkdir()
    external_onnx.write_bytes(b"onnx")
    engine_path = tmp_path / "amt.engine"
    engine_path.write_bytes(b"engine")
    observed: list[Path] = []

    class _Context:
        pass

    class _Engine:
        def create_execution_context(self):
            return _Context()

    class _Runtime:
        def deserialize_cuda_engine(self, data):
            assert data == b"engine"
            return _Engine()

    class _Logger:
        WARNING = 1

        def __call__(self, level):
            return object()

    fake_trt = SimpleNamespace(
        Logger=_Logger(),
        Runtime=lambda logger: _Runtime(),
    )
    monkeypatch.setattr(at, "ensure_amt_engine", lambda spec, path: observed.append(path) or engine_path)
    monkeypatch.setattr(at, "_import_runtime", lambda: fake_trt)
    monkeypatch.setattr(at.torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=1))

    backend = AMTTensorRTEngine(_spec(), external_onnx)

    assert observed == [external_onnx]
    assert backend._engine is not None


def test_close_synchronizes_stream_before_releasing_resources() -> None:
    backend = object.__new__(AMTTensorRTEngine)
    context = object()
    buffers = [object(), object()]
    events: list[str] = []

    class _Stream:
        def synchronize(self) -> None:
            assert backend._context is context
            assert backend._input_buffers is buffers
            events.append("synchronize")

    backend._stream = _Stream()
    backend._context = context
    backend._input_buffers = buffers
    backend._output_buffer = object()
    backend._buffer_shape = (1, 3, 128, 128)
    backend._engine = object()
    backend._runtime = object()

    backend.close()

    assert events == ["synchronize"]
    assert backend._context is None
    assert backend._input_buffers is None


def test_timing_contract_reports_h2d_and_d2h_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(AMTTensorRTEngine)
    backend.spec = _spec()
    backend.device = "cuda"
    backend._torch = torch
    backend._timing_enabled = False
    backend._transfer_timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
    backend._input_buffers = [torch.zeros(1, 3, 128, 128), torch.zeros(1, 3, 128, 128)]

    class _Output:
        device = SimpleNamespace(type="cuda")

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return torch.zeros(1, 3, 128, 128)

        def __getitem__(self, selection):
            return self

    backend._output_buffer = _Output()
    backend._buffer_shape = (1, 3, 128, 128)
    backend._buffer_capacity = None
    backend._ensure_buffers = lambda batch_size: None
    backend._context = SimpleNamespace(execute_async_v3=lambda stream_handle: True)
    backend._stream = SimpleNamespace(cuda_stream=1)
    monkeypatch.setattr(backend._torch.cuda, "synchronize", lambda device=None: None, raising=False)

    backend.reset_timing()
    backend.set_timing_enabled(True)
    backend.infer_batch(torch.zeros(1, 3, 128, 128), torch.zeros(1, 3, 128, 128))
    backend.transfer_batch_to_host(backend._output_buffer)
    snapshot = backend.timing_snapshot()

    assert set(snapshot) == {"h2d_time_s", "d2h_time_s"}
    assert snapshot["h2d_time_s"] >= 0.0
    assert snapshot["d2h_time_s"] >= 0.0


def test_buffer_capacity_is_reused_across_partial_and_full_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_upscaler import amt_tensorrt as at

    class _Buffer:
        _next_pointer = 100

        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self._pointer = _Buffer._next_pointer
            _Buffer._next_pointer += 1

        def data_ptr(self) -> int:
            return self._pointer

        def copy_(self, source: torch.Tensor) -> "_Buffer":
            assert tuple(source.shape[1:]) == self.shape[1:]
            return self

        def __getitem__(self, selection):
            if isinstance(selection, slice):
                return _Buffer((selection.stop, *self.shape[1:]))
            raise AssertionError("test only expects a batch slice")

    class _Context:
        def __init__(self) -> None:
            self.batch_size = 0
            self.addresses: dict[str, int] = {}

        def set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
            self.batch_size = shape[0]

        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            return (self.batch_size, 3, 128, 128)

        def set_tensor_address(self, name: str, pointer: int) -> None:
            self.addresses[name] = pointer

        def execute_async_v3(self, stream_handle: int) -> bool:
            return True

    backend = object.__new__(AMTTensorRTEngine)
    backend.spec = _spec()
    backend.device = "cuda"
    backend._input_names = ["frame_a", "frame_b"]
    backend._output_name = "output"
    backend._context = _Context()
    backend._stream = SimpleNamespace(cuda_stream=1)
    backend._input_buffers = None
    backend._output_buffer = None
    backend._buffer_shape = None
    backend._buffer_capacity = None
    backend._timing_enabled = False
    backend._transfer_timings = {"h2d_time_s": 0.0, "d2h_time_s": 0.0}
    monkeypatch.setattr(at.torch, "empty", lambda shape, dtype, device: _Buffer(tuple(shape)))

    batch_1 = torch.zeros(1, 3, 128, 128)
    batch_2 = torch.zeros(2, 3, 128, 128)
    output_1 = backend.infer_batch(batch_1, batch_1)
    input_pointers = tuple(buffer.data_ptr() for buffer in backend._input_buffers)
    output_pointer = backend._output_buffer.data_ptr()
    output_2 = backend.infer_batch(batch_2, batch_2)

    assert output_1.shape == (1, 3, 128, 128)
    assert output_2.shape == (2, 3, 128, 128)
    assert input_pointers == tuple(buffer.data_ptr() for buffer in backend._input_buffers)
    assert output_pointer == backend._output_buffer.data_ptr()


def test_ensure_buffers_rejects_unresolved_output_batch_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_upscaler import amt_tensorrt as at

    class _Buffer:
        def data_ptr(self) -> int:
            return 1

    class _Context:
        def set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
            pass

        def get_tensor_shape(self, name: str) -> tuple[int, ...]:
            return (-1, 3, 128, 128)

        def set_tensor_address(self, name: str, pointer: int) -> None:
            pass

    backend = object.__new__(AMTTensorRTEngine)
    backend.spec = _spec()
    backend.device = "cuda"
    backend._input_names = ["frame_a", "frame_b"]
    backend._output_name = "output"
    backend._context = _Context()
    backend._input_buffers = None
    backend._output_buffer = None
    backend._buffer_capacity = None
    monkeypatch.setattr(at.torch, "empty", lambda shape, dtype, device: _Buffer())

    with pytest.raises(RuntimeError, match="output batch shape"):
        backend._ensure_buffers(1)


def _fake_graph(
    input_names=("frame_a", "frame_b"),
    input_shape=(-1, 3, 128, 128),
    input_dtype="half",
    output_names=("output",),
    output_shape=(-1, 3, 128, 128),
    output_dtype="half",
):
    return SimpleNamespace(
        num_inputs=len(input_names),
        num_outputs=len(output_names),
        get_input=lambda index: SimpleNamespace(
            name=input_names[index], shape=input_shape, dtype=input_dtype
        ),
        get_output=lambda index: SimpleNamespace(
            name=output_names[index], shape=output_shape, dtype=output_dtype
        ),
    )


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (_fake_graph(input_names=("left", "frame_b")), "input names"),
        (_fake_graph(input_shape=(1, 3, 128, 128)), "dynamic batch"),
        (_fake_graph(input_shape=(-1, 1, 128, 128)), "input channels"),
        (_fake_graph(input_dtype="float"), "input dtype"),
        (_fake_graph(output_names=("wrong",)), "output name"),
        (_fake_graph(output_shape=(-1, 3, 64, 128)), "output shape"),
        (_fake_graph(output_dtype="float"), "output dtype"),
        (_fake_graph(output_names=("output", "extra")), "one output"),
    ],
)
def test_parser_rejects_incompatible_pair_graph_contract(graph, message: str) -> None:
    from video_upscaler import amt_tensorrt as at

    trt = SimpleNamespace(DataType=SimpleNamespace(HALF="half"))

    with pytest.raises(RuntimeError, match=message):
        at._validate_graph_contract(graph, trt, _spec())


def test_parser_accepts_exact_pair_graph_contract() -> None:
    from video_upscaler import amt_tensorrt as at

    trt = SimpleNamespace(DataType=SimpleNamespace(HALF="half"))
    at._validate_graph_contract(_fake_graph(), trt, _spec())


@pytest.mark.parametrize(
    "changed",
    ["gpu", "compute_capability", "tensorrt", "scale", "precision", "max_batch", "opset", "export_version", "engine_format_version"],
)
def test_cache_invalidation_covers_remaining_compatibility_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str
) -> None:
    from video_upscaler import amt_tensorrt as at

    monkeypatch.setattr(at, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(at, "prepare_env", lambda: None)
    runtime = {"gpu": "GPU_X", "compute_capability": "8.6", "cuda": "12.9", "tensorrt": "11.1"}
    monkeypatch.setattr(at, "_runtime_metadata", lambda: runtime)
    checkpoint = tmp_path / "amt-s.pth"
    checkpoint.write_bytes(b"checkpoint")
    onnx_path = tmp_path / "pair.onnx"
    onnx_path.write_bytes(b"onnx")
    monkeypatch.setattr(at, "_checkpoint_path", lambda model_key: checkpoint)
    engine = amt_engine_path(_spec())
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"engine")
    metadata = at._cache_metadata(_spec(), onnx_path)
    if changed in runtime:
        runtime[changed] = {"gpu": "GPU_Y", "compute_capability": "9.0", "tensorrt": "12.0"}[changed]
    elif changed == "scale":
        metadata[changed] = 0.5
    elif changed == "precision":
        metadata[changed] = "fp32"
    elif changed == "max_batch":
        metadata[changed] = 4
    elif changed == "opset":
        metadata[changed] = 18
    elif changed == "export_version":
        metadata[changed] = 2
    else:
        metadata[changed] = 2
    engine.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    rebuilt: list[Path] = []
    monkeypatch.setattr(at, "_build_amt_engine", lambda spec, path: rebuilt.append(path) or engine)

    ensure_amt_engine(_spec(), onnx_path)

    assert rebuilt == [onnx_path]


@pytest.mark.skipif(
    os.environ.get("CLARITY_RUN_AMT_TENSORRT_INTEGRATION") != "1",
    reason="set CLARITY_RUN_AMT_TENSORRT_INTEGRATION=1 to run TensorRT integration",
)
def test_optional_amt_tensorrt_integration_builds_batches_and_reuses_buffers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import amt_export, config
    from video_upscaler.amt.networks.amt_s import Model as AMT_S
    from video_upscaler.interp import _MODEL_PARAMS
    from video_upscaler.tensorrt_backend import prepare_env

    error = prepare_env()
    if error:
        pytest.skip(error)
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    checkpoint = tmp_path / "amt-s.pth"
    torch.save({"state_dict": AMT_S(**_MODEL_PARAMS["AMT-S"]).state_dict()}, checkpoint)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    from video_upscaler import amt_tensorrt as at

    monkeypatch.setattr(at, "MODELS_DIR", tmp_path)
    spec = _spec()
    onnx_path = tmp_path / "external" / "amt-pair.onnx"
    amt_export.export_amt_pair("AMT-S", 128, 128, 1.0, onnx_path, 17)
    backend = AMTTensorRTEngine(spec, onnx_path)
    input_1 = torch.zeros(1, 3, 128, 128)
    input_2 = torch.zeros(2, 3, 128, 128)
    output_1 = backend.transfer_batch_to_host(backend.infer_batch(input_1, input_1))
    backend.infer_batch(input_2, input_2)
    input_ptrs = tuple(buffer.data_ptr() for buffer in backend._input_buffers)
    output_2 = backend.transfer_batch_to_host(backend.infer_batch(input_2, input_2))

    assert output_1.shape == (1, 3, 128, 128)
    assert output_2.shape == (2, 3, 128, 128)
    assert torch.isfinite(output_1).all()
    assert torch.isfinite(output_2).all()
    assert input_ptrs == tuple(buffer.data_ptr() for buffer in backend._input_buffers)
    backend.close()


def test_trt_11_without_fp16_builder_flag_requires_fp16_graph() -> None:
    from video_upscaler import amt_tensorrt as at

    class _Config:
        def __init__(self) -> None:
            self.flags: list[object] = []

        def set_flag(self, flag: object) -> None:
            self.flags.append(flag)

    config = _Config()
    needs_converted_graph = at._configure_fp16(
        SimpleNamespace(BuilderFlag=SimpleNamespace()), config
    )

    assert needs_converted_graph is True
    assert config.flags == []


def test_locked_temporary_fp16_graph_cleanup_does_not_mask_build_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import amt_tensorrt as at

    temporary = tmp_path / "converted.onnx"
    temporary.write_bytes(b"graph")

    def locked_unlink(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("file is still held by TensorRT")

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    at._cleanup_temp_graph(temporary)


def test_backend_prepare_and_finalize_use_amt_padding_contract() -> None:
    backend = object.__new__(AMTTensorRTEngine)
    backend.spec = _spec()
    backend._padder = None

    frames = [np.full((127, 127, 3), 64, dtype=np.uint8)]
    prepared = backend.prepare_frames(frames)

    assert len(prepared) == 1
    assert prepared[0].shape == (3, 128, 128)
    assert prepared[0].dtype == torch.float32
    assert prepared[0].min().item() == pytest.approx(64 / 255)
    finalized = backend.finalize_frames(prepared)
    assert finalized[0].shape == (127, 127, 3)
    assert finalized[0].dtype == np.uint8


def test_backend_rejects_input_shape_and_device_contracts() -> None:
    backend = object.__new__(AMTTensorRTEngine)
    backend.spec = _spec()
    backend._padder = None

    with pytest.raises(ValueError, match="same dimensions"):
        backend.prepare_frames([
            np.zeros((127, 127, 3), dtype=np.uint8),
            np.zeros((128, 127, 3), dtype=np.uint8),
        ])
    with pytest.raises(ValueError, match="padded shape"):
        backend.prepare_frames([np.zeros((129, 129, 3), dtype=np.uint8)])

    with pytest.raises(ValueError, match="CPU"):
        backend._validate_batch(torch.zeros(1, 3, 128, 128, device="meta"), torch.zeros(1, 3, 128, 128))
    with pytest.raises(ValueError, match="shape"):
        backend._validate_batch(torch.zeros(1, 1, 128, 128), torch.zeros(1, 1, 128, 128))


def test_cache_metadata_records_full_hashes_and_engine_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_upscaler import amt_tensorrt as at

    monkeypatch.setattr(at, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(at, "_runtime_metadata", lambda: {
        "gpu": "GPU_X",
        "compute_capability": "8.6",
        "cuda": "12.9",
        "tensorrt": "11.1",
    })
    checkpoint = tmp_path / "amt-s.pth"
    checkpoint.write_bytes(b"checkpoint")
    onnx_path = tmp_path / "pair.onnx"
    onnx_path.write_bytes(b"onnx")
    monkeypatch.setattr(at, "_checkpoint_path", lambda model_key: checkpoint)

    metadata = at._cache_metadata(_spec(), onnx_path)

    assert metadata["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert metadata["onnx_sha256"] == hashlib.sha256(b"onnx").hexdigest()
    assert metadata["model"] == "AMT-S"
    assert metadata["engine_format_version"] == 1
    assert metadata["padded_shape"] == [128, 128]
    assert metadata["max_batch"] == 2
