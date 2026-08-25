"""Tests for the arrow-key CLI selection flow helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_upscaler.cli import select_videos


@pytest.fixture
def fake_videos(tmp_path: Path) -> list[Path]:
    return [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mkv"]


def test_single_video_auto_selects(monkeypatch, tmp_path: Path) -> None:
    """One video skips the menu entirely (no prompt, no typing)."""
    only = tmp_path / "only.mp4"
    only.touch()
    videos = [only]
    # _interactive() is True but no questionary call should happen.
    monkeypatch.setattr("video_upscaler.cli._interactive", lambda: True)
    monkeypatch.setattr(
        "video_upscaler.cli.questionary.checkbox",
        lambda *a, **k: pytest.fail("checkbox should not be shown for 1 video"),
    )
    assert select_videos(videos) == videos


def test_non_interactive_selects_all(monkeypatch, fake_videos: list[Path]) -> None:
    """Piped stdin selects all videos without prompting."""
    monkeypatch.setattr("video_upscaler.cli._interactive", lambda: False)
    assert select_videos(fake_videos) == fake_videos


def test_checkbox_nothing_preselected(monkeypatch, fake_videos: list[Path]) -> None:
    """No video is pre-checked; the user toggles with space."""
    monkeypatch.setattr("video_upscaler.cli._interactive", lambda: True)

    captured = {}

    class _FakeCheckbox:
        def __init__(self, *args, **kwargs):
            captured["choices"] = kwargs.get("choices")
            captured["default"] = kwargs.get("default")

        def ask(self):
            return []

    monkeypatch.setattr("video_upscaler.cli.questionary", type("Q", (), {"checkbox": _FakeCheckbox}))
    with pytest.raises(SystemExit) as excinfo:
        select_videos(fake_videos)
    assert captured["choices"] == ["a.mp4", "b.mp4", "c.mkv"]
    assert captured["default"] is None
    assert excinfo.value.code == 0


def test_checkbox_maps_choices_back_to_paths(monkeypatch, fake_videos: list[Path]) -> None:
    """Questionary checkbox results (names) map back to the Path objects."""
    monkeypatch.setattr("video_upscaler.cli._interactive", lambda: True)

    class _FakeCheckbox:
        def __init__(self, *args, **kwargs):
            pass

        def ask(self):
            return ["b.mp4"]

    monkeypatch.setattr("video_upscaler.cli.questionary", type("Q", (), {"checkbox": _FakeCheckbox}))
    assert select_videos(fake_videos) == [fake_videos[1]]


def test_empty_selection_reprompts(monkeypatch, fake_videos: list[Path]) -> None:
    """Deselecting everything re-prompts instead of running an empty batch."""
    monkeypatch.setattr("video_upscaler.cli._interactive", lambda: True)

    calls = {"count": 0}

    class _FakeCheckbox:
        def __init__(self, *args, **kwargs):
            pass

        def ask(self):
            calls["count"] += 1
            return [] if calls["count"] == 1 else ["a.mp4"]

    monkeypatch.setattr("video_upscaler.cli.questionary", type("Q", (), {"checkbox": _FakeCheckbox}))
    assert select_videos(fake_videos) == [fake_videos[0]]
    assert calls["count"] == 2


def test_cancelled_checkbox_quits(monkeypatch, fake_videos: list[Path]) -> None:
    """Esc/Ctrl-C (ask returns None) exits the app gracefully."""
    monkeypatch.setattr("video_upscaler.cli._interactive", lambda: True)

    class _FakeCheckbox:
        def __init__(self, *args, **kwargs):
            pass

        def ask(self):
            return None

    monkeypatch.setattr("video_upscaler.cli.questionary", type("Q", (), {"checkbox": _FakeCheckbox}))
    with pytest.raises(SystemExit) as excinfo:
        select_videos(fake_videos)
    assert excinfo.value.code == 0
