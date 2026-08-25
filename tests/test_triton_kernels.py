"""GPU-free tests for triton kernel gating + persistent autotune cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_upscaler import triton_kernels as tk


def test_cache_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLARITY_TRITON_CACHE", str(tmp_path / "cache.json"))
    assert tk._cache_path() == tmp_path / "cache.json"


def test_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tk, "_SESSION", {})
    monkeypatch.setattr(tk, "_DISK", {})
    disk = tmp_path / "cache.json"

    saved = {}

    def _fake_save():
        saved["payload"] = dict(tk._DISK)

    monkeypatch.setattr(tk, "_save_cache", _fake_save)
    entry = {"block_m": 64, "block_n": 64, "block_c": 32}
    tk._set_cached("conv3x3", "1x64x64x3x64", entry)
    assert saved["payload"] == {"conv3x3|1x64x64x3x64": entry}
    assert tk._get_cached("conv3x3", "1x64x64x3x64") == entry


def test_disabled_entry_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tk, "_SESSION", {})
    monkeypatch.setattr(tk, "_DISK", {})
    monkeypatch.setattr(tk, "_save_cache", lambda: None)
    tk._set_cached("conv3x3", "1x32x32x3x64", {"disabled": True})
    assert tk._get_cached("conv3x3", "1x32x32x3x64") is None


def test_triton_available_false_without_torch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr("video_upscaler.triton_kernels.TRITON_ENABLED", True)
    assert tk.triton_available() is False


def test_triton_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("video_upscaler.triton_kernels.TRITON_ENABLED", False)
    assert tk.triton_available() is False


def test_select_config_uses_fingerprinted_storage_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLARITY_TRITON_CACHE", str(tmp_path / "missing.json"))
    monkeypatch.setattr(tk, "_device_fingerprint", lambda: "fp-test")
    monkeypatch.setattr(tk, "_SESSION", {})
    monkeypatch.setattr(tk, "_DISK", {})
    monkeypatch.setattr(tk, "_save_cache", lambda: None)

    calls: list[dict] = []

    def fake_build_and_bench(cfg: dict) -> tuple[float, float, bool]:
        calls.append(cfg)
        return (0.5, 1.0, True)  # triton_ms < torch_ms / 1.02: candidate wins

    chosen = tk._select_config("conv3x3", "shape", fake_build_and_bench)
    assert chosen == {"block_m": 64, "block_n": 32, "block_c": 16}
    first_round_calls = len(calls)  # autotune benchmarks every candidate

    chosen_again = tk._select_config("conv3x3", "shape", fake_build_and_bench)
    assert chosen_again == chosen
    assert len(calls) == first_round_calls  # cached: no second benchmark round

    assert len(tk._DISK) == 1
    (stored_key,) = tk._DISK
    # _set_cached is the pinned raw-key primitive, so the stored key is
    # "<kind>|<fingerprinted storage key>"; the fingerprint must be present.
    assert stored_key == f"conv3x3|{tk._cache_key('conv3x3', 'shape')}"
    assert stored_key.startswith("conv3x3|fp-test|")


def test_public_kernels_return_none_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tk, "triton_available", lambda: False)
    assert tk.conv3x3(None, None) is None
    assert tk.pixel_shuffle(None, 2) is None
