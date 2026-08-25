"""Torch-free tests for the ncnn Vulkan engine (arg building, chunking, progress)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from video_upscaler import ncnn
from video_upscaler.config import TOOLS_DIR


def test_ncnn_exe_paths() -> None:
    assert ncnn.ncnn_exe("realcugan").name == "realcugan-ncnn-vulkan.exe"
    real = ncnn.ncnn_exe("realcugan")
    assert TOOLS_DIR / "ncnn" in real.parents
    assert real.parent == TOOLS_DIR / "ncnn" or real.parent.parent == TOOLS_DIR / "ncnn"


def test_ncnn_exe_resolves_nested_release_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nested = (
        tmp_path / "ncnn" / "realcugan-ncnn-vulkan-20220728-windows"
    )
    nested.mkdir(parents=True)
    exe = nested / "realcugan-ncnn-vulkan.exe"
    exe.touch()
    monkeypatch.setattr(ncnn, "TOOLS_DIR", tmp_path)
    assert ncnn.ncnn_exe("realcugan") == exe


def test_model_dir_prefers_param_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "ncnn"
    (root / "noise").mkdir(parents=True)
    models = root / "models-se"
    models.mkdir()
    (models / "up2x-latest-denoise2x.param").touch()
    monkeypatch.setattr(ncnn, "TOOLS_DIR", tmp_path)
    assert ncnn.ncnn_model_dir("realcugan") == models


def test_model_dir_prefers_models_se_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ncnn"
    models = root / "models"
    models_se = root / "models-se"
    models.mkdir(parents=True)
    models_se.mkdir(parents=True)
    (models / "up2x-latest-denoise2x.param").touch()
    (models_se / "up3x-latest-denoise3x.param").touch()
    monkeypatch.setattr(ncnn, "TOOLS_DIR", tmp_path)
    # models-se (full SE model set) wins by name over earlier matches.
    assert ncnn.ncnn_model_dir("realcugan") == models_se


def test_model_dir_falls_back_when_tool_has_no_matching_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ncnn"
    models = root / "models"
    models.mkdir(parents=True)
    (models / "up4x-latest-denoise3x.param").touch()
    monkeypatch.setattr(ncnn, "TOOLS_DIR", tmp_path)
    assert ncnn.ncnn_model_dir("realcugan") == models


def test_check_ncnn_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ncnn, "TOOLS_DIR", tmp_path)
    message = ncnn.check_ncnn("realcugan")
    assert message is not None
    assert "realcugan-ncnn-vulkan" in message


def test_build_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ncnn, "TOOLS_DIR", tmp_path)
    exe = tmp_path / "ncnn" / "realcugan-ncnn-vulkan.exe"
    exe.parent.mkdir()
    exe.touch()
    models = exe.parent / "models-se"
    models.mkdir()
    (models / "up2x-latest-denoise2x.param").touch()
    command = ncnn._build_command("realcugan", {"s": 2, "n": 2}, tmp_path / "in", tmp_path / "out")
    assert command[0] == str(exe)
    assert command[command.index("-s") + 1] == "2"
    assert command[command.index("-n") + 1] == "2"
    assert command[command.index("-c") + 1] == "0"
    assert command[command.index("-f") + 1] == "png"
    assert command[command.index("-m") + 1] == str(models)


def test_enhance_chunk_roundtrip_and_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cv2

    monkeypatch.setattr(ncnn, "TOOLS_DIR", tmp_path)
    exe = tmp_path / "ncnn" / "realcugan-ncnn-vulkan.exe"
    exe.parent.mkdir()
    exe.touch()
    models = exe.parent / "models-se"
    models.mkdir()
    (models / "up2x-latest-denoise2x.param").touch()

    captured = {}

    def _fake_run(command, **kwargs):
        captured["command"] = command
        out_dir = Path(command[command.index("-o") + 1])
        for path in sorted(Path(command[command.index("-i") + 1]).glob("*.png")):
            img = cv2.imread(str(path))
            cv2.imwrite(str(out_dir / path.name), img)
        return subprocess.CompletedProcess(
            command, 0, stdout="50.00%\n100.00%\n", stderr=""
        )

    monkeypatch.setattr(ncnn.subprocess, "run", _fake_run)
    engine = ncnn.NCNNEngine("2x_Balanced")
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(3)]
    progress = []
    result = engine.enhance_chunk(frames, on_progress=progress.append)

    assert [frame.shape for frame in result] == [(8, 8, 3)] * 3
    assert result[0][0, 0].tolist() == [0, 0, 0]
    assert captured["command"]
    assert progress, "stdout % lines must drive progress callbacks"


def test_enhance_chunk_raises_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ncnn, "TOOLS_DIR", tmp_path)
    exe = tmp_path / "ncnn" / "realcugan-ncnn-vulkan.exe"
    exe.parent.mkdir()
    exe.touch()
    models = exe.parent / "models-se"
    models.mkdir()
    (models / "up2x-latest-denoise2x.param").touch()

    def _fake_fail(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="vulkan broke")

    monkeypatch.setattr(ncnn.subprocess, "run", _fake_fail)
    engine = ncnn.NCNNEngine("2x_Balanced")
    with pytest.raises(RuntimeError, match="vulkan broke"):
        engine.enhance_chunk([np.zeros((8, 8, 3), dtype=np.uint8)])
