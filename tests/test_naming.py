"""Tests for output filename generation and dedupe (suffix-tag scheme)."""

from __future__ import annotations

from pathlib import Path

from video_upscaler.processor import (
    dedup_output_tag,
    slowed_output_tag,
    unique_dedup_output_path,
    unique_interp_output_path,
    unique_output_path,
    upscale_output_tag,
)


def test_upscale_output_tag_by_profile() -> None:
    assert upscale_output_tag("2x_Balanced") == "2x"
    assert upscale_output_tag("3x_Deep") == "3x"
    assert upscale_output_tag("4x_Clean") == "4x"


def test_slowed_output_tag_includes_model() -> None:
    assert slowed_output_tag(2, "AMT-S") == "slowed2x_amt-s"
    assert slowed_output_tag(4, "amt-l") == "slowed4x_amt-l"
    assert slowed_output_tag(8, "AMT-G") == "slowed8x_amt-g"


def test_dedup_output_tag() -> None:
    assert dedup_output_tag(2, "gmfss") == "dedup2x_gmfss"
    assert dedup_output_tag(4, "RIFE") == "dedup4x_rife"


def test_unique_output_path_suffixes_and_keeps_extension(tmp_path: Path) -> None:
    out = unique_output_path(tmp_path, "Scaled_video.mp4", "2x_Balanced")
    assert out == tmp_path / "Scaled_video_2x.mp4"
    assert out.suffix == ".mp4"


def test_unique_output_path_3x_profile(tmp_path: Path) -> None:
    out = unique_output_path(tmp_path, "clip.mkv", "3x_Deep")
    assert out == tmp_path / "clip_3x.mkv"


def test_unique_output_path_never_overwrites(tmp_path: Path) -> None:
    (tmp_path / "Scaled_video_2x.mp4").write_bytes(b"a")
    first = unique_output_path(tmp_path, "Scaled_video.mp4", "2x_Balanced")
    assert first == tmp_path / "Scaled_video_2x_1.mp4"

    first.write_bytes(b"b")
    second = unique_output_path(tmp_path, "Scaled_video.mp4", "2x_Balanced")
    assert second == tmp_path / "Scaled_video_2x_2.mp4"


def test_unique_output_path_existing_input_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "Scaled_video.mp4").write_bytes(b"input")
    out = unique_output_path(tmp_path, "Scaled_video.mp4", "2x_Balanced")
    assert out == tmp_path / "Scaled_video_2x.mp4"
    assert (tmp_path / "Scaled_video.mp4").read_bytes() == b"input"


def test_interp_output_path_includes_model(tmp_path: Path) -> None:
    out = unique_interp_output_path(tmp_path, "video.mp4", 2, "AMT-S")
    assert out == tmp_path / "video_slowed2x_amt-s.mp4"


def test_dedup_output_path_and_unique(tmp_path: Path) -> None:
    out = unique_dedup_output_path(tmp_path, "video.mp4", 2, "gmfss")
    assert out == tmp_path / "video_dedup2x_gmfss.mp4"
    out.touch()

    out2 = unique_dedup_output_path(tmp_path, "video.mp4", 2, "gmfss")
    assert out2 == tmp_path / "video_dedup2x_gmfss_1.mp4"


def test_action_outputs_never_collide_on_same_stem(tmp_path: Path) -> None:
    cugan_out = unique_output_path(tmp_path, "video.mp4", "2x_Balanced")
    amt_out = unique_interp_output_path(tmp_path, "video.mp4", 2, "AMT-S")
    dedup_out = unique_dedup_output_path(tmp_path, "video.mp4", 2, "gmfss")

    assert cugan_out.name == "video_2x.mp4"
    assert amt_out.name == "video_slowed2x_amt-s.mp4"
    assert dedup_out.name == "video_dedup2x_gmfss.mp4"
    assert len({cugan_out.name, amt_out.name, dedup_out.name}) == 3
