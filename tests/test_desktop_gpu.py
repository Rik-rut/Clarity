"""Unit tests for GPU probing, compute capability parsing, and variant selection."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock
import pytest

from video_upscaler.desktop import gpu
from video_upscaler.desktop.gpu import (
    CUDA_126_VARIANT,
    CUDA_130_VARIANT,
    CPU_VARIANT,
    GpuInfo,
    detect_gpu,
    is_legacy_nvidia_name,
    parse_compute_cap,
    parse_nvidia_smi_output,
    parse_nvidia_smi_query_output,
)


def test_parse_compute_cap_valid():
    assert parse_compute_cap("12.0") == (12, 0)
    assert parse_compute_cap("8.6") == (8, 6)
    assert parse_compute_cap("7.5") == (7, 5)
    assert parse_compute_cap("6.1") == (6, 1)
    assert parse_compute_cap("5.2") == (5, 2)
    assert parse_compute_cap("  8.9  ") == (8, 9)


def test_parse_compute_cap_invalid():
    assert parse_compute_cap("") is None
    assert parse_compute_cap("N/A") is None
    assert parse_compute_cap("[Not Supported]") is None
    assert parse_compute_cap("invalid.cap") is None
    assert parse_compute_cap("12") is None


def test_is_legacy_nvidia_name():
    # Legacy architectures (< 7.5: Pascal, Maxwell, Kepler)
    assert is_legacy_nvidia_name("NVIDIA GeForce GTX 1080 Ti") is True
    assert is_legacy_nvidia_name("GeForce GTX 1060 6GB") is True
    assert is_legacy_nvidia_name("GeForce GTX 970") is True
    assert is_legacy_nvidia_name("GeForce GTX 750 Ti") is True
    assert is_legacy_nvidia_name("GeForce GT 730") is True
    assert is_legacy_nvidia_name("TITAN X (Pascal)") is True
    assert is_legacy_nvidia_name("Quadro P4000") is True
    assert is_legacy_nvidia_name("Quadro M2000") is True
    assert is_legacy_nvidia_name("Tesla P100-PCIE-16GB") is True

    # Modern architectures (>= 7.5: Turing, Ampere, Ada, Blackwell)
    assert is_legacy_nvidia_name("NVIDIA GeForce RTX 5060 Ti") is False
    assert is_legacy_nvidia_name("NVIDIA GeForce RTX 4090") is False
    assert is_legacy_nvidia_name("GeForce RTX 3050") is False
    assert is_legacy_nvidia_name("GeForce RTX 2060") is False
    assert is_legacy_nvidia_name("GeForce GTX 1660 Ti") is False
    assert is_legacy_nvidia_name("GeForce GTX 1650 Super") is False
    assert is_legacy_nvidia_name("AMD Radeon RX 7900 XTX") is False
    assert is_legacy_nvidia_name("") is False


def test_gpu_info_torch_variant_selection():
    # Non-NVIDIA
    assert GpuInfo(has_nvidia=False).torch_variant == CPU_VARIANT
    assert GpuInfo(has_nvidia=False, names=["AMD Radeon RX 6700"]).torch_variant == CPU_VARIANT

    # NVIDIA with compute capabilities
    assert GpuInfo(
        has_nvidia=True,
        names=["NVIDIA GeForce RTX 5060 Ti"],
        compute_capabilities=["12.0"],
    ).torch_variant == CUDA_130_VARIANT

    assert GpuInfo(
        has_nvidia=True,
        names=["GeForce RTX 3050"],
        compute_capabilities=["8.6"],
    ).torch_variant == CUDA_130_VARIANT

    assert GpuInfo(
        has_nvidia=True,
        names=["GeForce GTX 1660"],
        compute_capabilities=["7.5"],
    ).torch_variant == CUDA_130_VARIANT

    assert GpuInfo(
        has_nvidia=True,
        names=["GeForce GTX 1060"],
        compute_capabilities=["6.1"],
    ).torch_variant == CUDA_126_VARIANT

    assert GpuInfo(
        has_nvidia=True,
        names=["GeForce GTX 970"],
        compute_capabilities=["5.2"],
    ).torch_variant == CUDA_126_VARIANT

    # NVIDIA fallback without compute capability (e.g. legacy driver or -L fallback)
    assert GpuInfo(
        has_nvidia=True,
        names=["GeForce GTX 1070"],
        compute_capabilities=[],
    ).torch_variant == CUDA_126_VARIANT

    assert GpuInfo(
        has_nvidia=True,
        names=["NVIDIA GeForce RTX 4070"],
        compute_capabilities=[],
    ).torch_variant == CUDA_130_VARIANT

    # Serialization roundtrip check
    info = GpuInfo(
        has_nvidia=True,
        names=["RTX 5060 Ti"],
        compute_capabilities=["12.0"],
    )
    d = info.to_dict()
    assert d["has_nvidia"] is True
    assert d["names"] == ["RTX 5060 Ti"]
    assert d["compute_capabilities"] == ["12.0"]
    assert d["torch_variant"] == CUDA_130_VARIANT


def test_parse_nvidia_smi_query_output():
    raw = (
        "NVIDIA GeForce RTX 5060 Ti, 12.0\n"
        "NVIDIA GeForce GTX 1060 6GB, 6.1\n"
    )
    names, caps = parse_nvidia_smi_query_output(raw)
    assert names == ["NVIDIA GeForce RTX 5060 Ti", "NVIDIA GeForce GTX 1060 6GB"]
    assert caps == ["12.0", "6.1"]


def test_parse_nvidia_smi_output_L():
    raw = (
        "GPU 0: NVIDIA GeForce RTX 3050 (UUID: GPU-12345678-abcd)\n"
        "GPU 1: GeForce GTX 1080 (UUID: GPU-87654321-dcba)\n"
    )
    names = parse_nvidia_smi_output(raw)
    assert names == ["NVIDIA GeForce RTX 3050", "GeForce GTX 1080"]


def test_detect_gpu_success_query(monkeypatch):
    monkeypatch.setattr(
        gpu,
        "_run_smi_cmd",
        lambda cmd: (
            ("NVIDIA GeForce RTX 5060 Ti, 12.0\n", None)
            if "--query-gpu=name,compute_cap" in cmd
            else (None, "error")
        ),
    )
    info = detect_gpu()
    assert info.has_nvidia is True
    assert info.names == ["NVIDIA GeForce RTX 5060 Ti"]
    assert info.compute_capabilities == ["12.0"]
    assert info.torch_variant == CUDA_130_VARIANT


def test_detect_gpu_fallback_to_L(monkeypatch):
    def fake_run(cmd):
        if "--query-gpu=name,compute_cap" in cmd:
            return None, "query not supported"
        if "-L" in cmd:
            return "GPU 0: GeForce GTX 1060 6GB (UUID: GPU-xxx)\n", None
        return None, "unknown"

    monkeypatch.setattr(gpu, "_run_smi_cmd", fake_run)
    info = detect_gpu()
    assert info.has_nvidia is True
    assert info.names == ["GeForce GTX 1060 6GB"]
    assert info.compute_capabilities == []
    assert info.torch_variant == CUDA_126_VARIANT


def test_detect_gpu_no_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "_run_smi_cmd", lambda cmd: (None, "nvidia-smi not found"))
    info = detect_gpu()
    assert info.has_nvidia is False
    assert info.names == []
    assert info.compute_capabilities == []
    assert info.torch_variant == CPU_VARIANT


def _make_fake_venv(venv_dir: Path) -> Path:
    scripts = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    scripts.mkdir(parents=True, exist_ok=True)
    binary = scripts / ("python.exe" if os.name == "nt" else "python")
    binary.write_bytes(b"")
    return binary


def test_hardware_stamp_roundtrip(tmp_path: Path):
    from video_upscaler.desktop.gpu import read_hardware_stamp, write_hardware_stamp
    info = GpuInfo(
        has_nvidia=True,
        names=["NVIDIA GeForce RTX 5060 Ti"],
        compute_capabilities=["12.0"],
    )
    write_hardware_stamp(tmp_path, info)
    stamp = read_hardware_stamp(tmp_path)
    assert stamp is not None
    assert stamp["names"] == ["NVIDIA GeForce RTX 5060 Ti"]
    assert stamp["compute_capabilities"] == ["12.0"]
    assert stamp["torch_variant"] == CUDA_130_VARIANT


def test_check_venv_missing_stamp(tmp_path: Path):
    from video_upscaler.desktop.gpu import check_venv
    _make_fake_venv(tmp_path)
    assert check_venv(tmp_path) == 1


def test_check_venv_hardware_mismatch(tmp_path: Path, monkeypatch):
    from video_upscaler.desktop.gpu import check_venv, write_hardware_stamp
    _make_fake_venv(tmp_path)
    # Provisioned as GTX 1060 (cu126)
    old_info = GpuInfo(has_nvidia=True, names=["GeForce GTX 1060"], compute_capabilities=["6.1"])
    write_hardware_stamp(tmp_path, old_info)

    # Live hardware is RTX 5060 Ti (cu130)
    monkeypatch.setattr(
        gpu,
        "detect_gpu",
        lambda: GpuInfo(has_nvidia=True, names=["RTX 5060 Ti"], compute_capabilities=["12.0"]),
    )
    assert check_venv(tmp_path) == 1


def test_check_venv_cuda_usable_success(tmp_path: Path, monkeypatch):
    import subprocess
    from video_upscaler.desktop.gpu import check_venv, write_hardware_stamp
    _make_fake_venv(tmp_path)
    live_info = GpuInfo(has_nvidia=True, names=["RTX 5060 Ti"], compute_capabilities=["12.0"])
    write_hardware_stamp(tmp_path, live_info)
    monkeypatch.setattr(gpu, "detect_gpu", lambda: live_info)

    # Mock subprocess.run for CUDA check to return code 0
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: MagicMock(returncode=0, stdout="", stderr=""),
    )
    assert check_venv(tmp_path) == 0


def test_check_venv_cuda_usable_kernel_error(tmp_path: Path, monkeypatch):
    import subprocess
    from video_upscaler.desktop.gpu import check_venv, write_hardware_stamp
    _make_fake_venv(tmp_path)
    live_info = GpuInfo(has_nvidia=True, names=["RTX 5060 Ti"], compute_capabilities=["12.0"])
    write_hardware_stamp(tmp_path, live_info)
    monkeypatch.setattr(gpu, "detect_gpu", lambda: live_info)

    # Mock subprocess.run for CUDA check to return code 2 (broken kernel)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: MagicMock(returncode=2, stdout="", stderr="cudaErrorNoKernelImageForDevice"),
    )
    assert check_venv(tmp_path) == 2


def test_verify_state_missing_setup_json(tmp_path: Path):
    from video_upscaler.desktop.gpu import verify_state
    assert verify_state(tmp_path) == 1


def test_verify_state_hardware_changed(tmp_path: Path, monkeypatch):
    import json
    from video_upscaler.desktop.gpu import verify_state
    _make_fake_venv(tmp_path / "env")
    state = {
        "torch_variant": "cu126",
        "gpu_names": ["GeForce GTX 1060"],
        "compute_capabilities": ["6.1"],
    }
    (tmp_path / "setup.json").write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(
        gpu,
        "detect_gpu",
        lambda: GpuInfo(has_nvidia=True, names=["RTX 5060 Ti"], compute_capabilities=["12.0"]),
    )
    assert verify_state(tmp_path) == 1


def test_verify_state_matches_and_passes(tmp_path: Path, monkeypatch):
    import json
    import subprocess
    from video_upscaler.desktop.gpu import verify_state
    _make_fake_venv(tmp_path / "env")
    state = {
        "torch_variant": "cu130",
        "gpu_names": ["NVIDIA GeForce RTX 5060 Ti"],
        "compute_capabilities": ["12.0"],
    }
    (tmp_path / "setup.json").write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(
        gpu,
        "detect_gpu",
        lambda: GpuInfo(
            has_nvidia=True,
            names=["NVIDIA GeForce RTX 5060 Ti"],
            compute_capabilities=["12.0"],
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: MagicMock(returncode=0, stdout="", stderr=""),
    )
    assert verify_state(tmp_path) == 0


def test_cli_main_json_output(capsys, monkeypatch):
    monkeypatch.setattr(
        gpu,
        "detect_gpu",
        lambda: GpuInfo(has_nvidia=True, names=["RTX 5060 Ti"], compute_capabilities=["12.0"]),
    )
    rc = gpu.main(["--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RTX 5060 Ti" in out
    assert "cu130" in out
