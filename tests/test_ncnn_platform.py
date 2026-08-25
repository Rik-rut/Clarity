"""Platform-aware ncnn tool resolution tests (asset tokens, exe names)."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_upscaler import ncnn


def test_asset_token_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ncnn.os, "name", "nt")
    assert ncnn._os_asset_token() == "windows"


def test_asset_token_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ncnn.os, "name", "posix")
    monkeypatch.setattr(ncnn.sys, "platform", "darwin")
    assert ncnn._os_asset_token() == "macos"


def test_asset_token_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ncnn.os, "name", "posix")
    monkeypatch.setattr(ncnn.sys, "platform", "linux")
    assert ncnn._os_asset_token() == "ubuntu"


def test_exe_name_follows_os(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ncnn.os, "name", "nt")
    assert ncnn._exe_name("realcugan").endswith(".exe")
    monkeypatch.setattr(ncnn.os, "name", "posix")
    assert not ncnn._exe_name("realcugan").endswith(".exe")


def test_ncnn_exe_finds_posix_binary_in_nested_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A release layout extracted on Linux/macOS resolves without .exe."""
    nested = tmp_path / "ncnn" / "realcugan-ncnn-vulkan-20220728-ubuntu"
    nested.mkdir(parents=True)
    binary = nested / "realcugan-ncnn-vulkan"
    binary.touch()
    monkeypatch.setattr(ncnn, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(ncnn.os, "name", "posix")
    assert ncnn.ncnn_exe("realcugan") == binary


def test_latest_zip_url_prefers_os_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "assets": [
            {"browser_download_url": "https://example/x-ubuntu.zip"},
            {"browser_download_url": "https://example/x-macos.zip"},
            {"browser_download_url": "https://example/x-windows.zip"},
        ]
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json_bytes

    import json

    json_bytes = json.dumps(payload).encode()

    seen: dict[str, str] = {}

    def fake_urlopen(request, timeout=0):
        seen["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr(ncnn.urllib.request, "urlopen", fake_urlopen)

    monkeypatch.setattr(ncnn.os, "name", "posix")
    monkeypatch.setattr(ncnn.sys, "platform", "darwin")
    assert ncnn._latest_zip_url("nihui/realcugan-ncnn-vulkan").endswith("-macos.zip")

    monkeypatch.setattr(ncnn.os, "name", "nt")
    assert ncnn._latest_zip_url("nihui/realcugan-ncnn-vulkan").endswith("-windows.zip")
