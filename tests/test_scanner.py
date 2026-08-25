"""Tests for input directory scanning."""

from __future__ import annotations

from pathlib import Path

from video_upscaler.scanner import scan_videos


def _make_file(path: Path, name: str) -> None:
    (path / name).write_bytes(b"dummy")


def test_supported_extensions_included(tmp_path: Path) -> None:
    for name in ("a.mp4", "b.mkv", "c.mov", "d.webm", "e.avi", "f.m4v", "g.ts"):
        _make_file(tmp_path, name)
    names = {p.name for p in scan_videos(tmp_path)}
    assert names == {"a.mp4", "b.mkv", "c.mov", "d.webm", "e.avi", "f.m4v", "g.ts"}


def test_unsupported_extensions_excluded(tmp_path: Path) -> None:
    _make_file(tmp_path, "clip.mp4")
    _make_file(tmp_path, "notes.txt")
    _make_file(tmp_path, "photo.jpg")
    assert [p.name for p in scan_videos(tmp_path)] == ["clip.mp4"]


def test_uppercase_extension_accepted(tmp_path: Path) -> None:
    _make_file(tmp_path, "CLIP.MP4")
    assert [p.name for p in scan_videos(tmp_path)] == ["CLIP.MP4"]


def test_hidden_files_ignored(tmp_path: Path) -> None:
    _make_file(tmp_path, ".hidden.mp4")
    _make_file(tmp_path, "visible.mp4")
    assert [p.name for p in scan_videos(tmp_path)] == ["visible.mp4"]


def test_directories_ignored(tmp_path: Path) -> None:
    (tmp_path / "subdir.mp4").mkdir()
    _make_file(tmp_path, "real.mp4")
    assert [p.name for p in scan_videos(tmp_path)] == ["real.mp4"]


def test_alphabetical_sorting_case_insensitive(tmp_path: Path) -> None:
    for name in ("Zeta.mp4", "alpha.mkv", "Beta.mp4"):
        _make_file(tmp_path, name)
    assert [p.name for p in scan_videos(tmp_path)] == [
        "alpha.mkv",
        "Beta.mp4",
        "Zeta.mp4",
    ]


def test_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert scan_videos(tmp_path / "does-not-exist") == []
