"""Unit tests for ffmpeg.py probe and audio probing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from video_upscaler import ffmpeg


def test_probe_with_audio_codec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sample = tmp_path / "video.mkv"
    sample.touch()

    monkeypatch.setattr(ffmpeg, "_ffprobe_path", lambda: "ffprobe")

    video_json = json.dumps({
        "streams": [{
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "24/1",
            "codec_name": "h264",
        }],
        "format": {"duration": "12.5"},
    })
    audio_json = json.dumps({
        "streams": [{
            "codec_name": "opus",
            "codec_type": "audio",
        }]
    })

    def fake_run(cmd, capture_output=True, text=True):
        cmd_str = " ".join(cmd)
        mock_res = MagicMock()
        mock_res.returncode = 0
        if "-select_streams v:0" in cmd_str:
            mock_res.stdout = video_json
        elif "-select_streams a" in cmd_str:
            mock_res.stdout = audio_json
        else:
            mock_res.stdout = "{}"
        return mock_res

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    info = ffmpeg.probe(sample)
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["fps"] == 24.0
    assert info["has_audio"] is True
    assert info["audio_codec"] == "opus"


def test_probe_without_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sample = tmp_path / "video.mp4"
    sample.touch()

    monkeypatch.setattr(ffmpeg, "_ffprobe_path", lambda: "ffprobe")

    video_json = json.dumps({
        "streams": [{
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "30/1",
            "codec_name": "h264",
        }],
        "format": {"duration": "5.0"},
    })
    audio_json = json.dumps({"streams": []})

    def fake_run(cmd, capture_output=True, text=True):
        cmd_str = " ".join(cmd)
        mock_res = MagicMock()
        mock_res.returncode = 0
        if "-select_streams v:0" in cmd_str:
            mock_res.stdout = video_json
        elif "-select_streams a" in cmd_str:
            mock_res.stdout = audio_json
        else:
            mock_res.stdout = "{}"
        return mock_res

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    info = ffmpeg.probe(sample)
    assert info["has_audio"] is False
    assert info["audio_codec"] is None


class FakeProcess:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.stdin = MagicMock()
        self.stderr = MagicMock()
        self.stderr.read.return_value = b""
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0


def _setup_encode_mocks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ffmpeg, "ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(ffmpeg, "nvenc_available", lambda: False)
    monkeypatch.setattr(ffmpeg, "prores_available", lambda: True)

    captured_commands = []

    def fake_popen(cmd, **kwargs):
        captured_commands.append(cmd)
        return FakeProcess(cmd)

    monkeypatch.setattr(ffmpeg.subprocess, "Popen", fake_popen)
    return captured_commands


@pytest.mark.parametrize("codec", ["aac", "mp3", "ac3", "eac3", "AAC", "MP3"])
def test_encode_video_audio_copy_for_compatible_codecs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, codec: str
) -> None:
    captured = _setup_encode_mocks(monkeypatch)
    src = tmp_path / "src.mp4"
    out = tmp_path / "out.mp4"
    frames = [b"\x00" * (10 * 10 * 3)]

    ffmpeg.encode_video(
        frames=frames,
        frame_w=10,
        frame_h=10,
        fps=30.0,
        src_path=src,
        out_path=out,
        use_audio=True,
        audio_codec=codec,
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert "-c:a" in cmd
    idx = cmd.index("-c:a")
    assert cmd[idx + 1] == "copy"


@pytest.mark.parametrize("codec", ["opus", "vorbis", "flac", "pcm_s16le", None])
def test_encode_video_audio_reencode_for_incompatible_or_unknown_codecs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, codec: str | None
) -> None:
    captured = _setup_encode_mocks(monkeypatch)
    monkeypatch.setattr(ffmpeg, "_ffprobe_path", lambda: None)
    src = tmp_path / "src.mkv"
    out = tmp_path / "out.mp4"
    frames = [b"\x00" * (10 * 10 * 3)]

    ffmpeg.encode_video(
        frames=frames,
        frame_w=10,
        frame_h=10,
        fps=30.0,
        src_path=src,
        out_path=out,
        use_audio=True,
        audio_codec=codec,
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert "-c:a" in cmd
    idx = cmd.index("-c:a")
    assert cmd[idx:idx + 4] == ["-c:a", "aac", "-b:a", "320k"]


def test_encode_video_no_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _setup_encode_mocks(monkeypatch)
    src = tmp_path / "src.mp4"
    out = tmp_path / "out.mp4"
    frames = [b"\x00" * (10 * 10 * 3)]

    ffmpeg.encode_video(
        frames=frames,
        frame_w=10,
        frame_h=10,
        fps=30.0,
        src_path=src,
        out_path=out,
        use_audio=False,
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert "-c:a" not in cmd
    assert "-map" in cmd
    assert "1:a:0?" not in cmd


def test_encode_video_alpha_prores_copies_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _setup_encode_mocks(monkeypatch)
    src = tmp_path / "src.mov"
    out = tmp_path / "out.mov"
    frames = [b"\x00" * (10 * 10 * 4)]

    ffmpeg.encode_video(
        frames=frames,
        frame_w=10,
        frame_h=10,
        fps=30.0,
        src_path=src,
        out_path=out,
        use_audio=True,
        alpha=True,
        audio_codec="pcm_s16le",
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert "-c:a" in cmd
    idx = cmd.index("-c:a")
    assert cmd[idx + 1] == "copy"
    assert "prores_ks" in cmd


def test_encode_video_out_size_lanczos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _setup_encode_mocks(monkeypatch)
    src = tmp_path / "src.mp4"
    out = tmp_path / "out.mp4"
    frames = [b"\x00" * (1280 * 720 * 3)]

    ffmpeg.encode_video(
        frames=frames,
        frame_w=1280,
        frame_h=720,
        fps=30.0,
        src_path=src,
        out_path=out,
        use_audio=False,
        out_size=(640, 360),
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert "-vf" in cmd
    idx = cmd.index("-vf")
    assert cmd[idx + 1] == "scale=640:360:flags=lanczos"


def test_encode_video_out_size_rounds_to_even(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _setup_encode_mocks(monkeypatch)
    src = tmp_path / "src.mp4"
    out = tmp_path / "out.mp4"
    frames = [b"\x00" * (1280 * 720 * 3)]

    ffmpeg.encode_video(
        frames=frames,
        frame_w=1280,
        frame_h=720,
        fps=30.0,
        src_path=src,
        out_path=out,
        use_audio=False,
        out_size=(641, 361),
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert "-vf" in cmd
    idx = cmd.index("-vf")
    assert cmd[idx + 1] == "scale=640:360:flags=lanczos"
