"""Tests for 1x profiles and output scale helper."""

from __future__ import annotations

import pytest

from video_upscaler.models import (
    PROFILES,
    default_profile,
    model_for_profile,
    output_scale_for_profile,
    scale_for_model,
)


def test_default_profile_is_still_2x_balanced() -> None:
    assert default_profile() == "2x_Balanced"


def test_1x_profiles_exist_and_map_to_2x_models() -> None:
    expected_variants = ["Balanced", "Clean", "Light", "Deep", "Faithful"]
    for variant in expected_variants:
        profile_1x = f"1x_{variant}"
        profile_2x = f"2x_{variant}"
        assert profile_1x in PROFILES
        # Model file matches the 2x model file
        assert model_for_profile(profile_1x) == model_for_profile(profile_2x)
        # scale_for_model still returns 2 for the model weights
        model_file = model_for_profile(profile_1x)
        assert scale_for_model(model_file) == 2
        # output_scale_for_profile returns 1 for 1x and 2 for 2x
        assert output_scale_for_profile(profile_1x) == 1
        assert output_scale_for_profile(profile_2x) == 2


def test_output_scale_for_all_scales() -> None:
    assert output_scale_for_profile("1x_Clean") == 1
    assert output_scale_for_profile("2x_Clean") == 2
    assert output_scale_for_profile("3x_Clean") == 3
    assert output_scale_for_profile("4x_Clean") == 4


def test_1x_process_videos_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess
    import cv2
    from video_upscaler import config, processor
    ffmpeg_exe = config.ffmpeg_path()
    if not ffmpeg_exe:
        pytest.skip("ffmpeg required for end-to-end test")

    src = tmp_path / "small.mp4"
    subprocess.run([
        ffmpeg_exe, "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=0.2",
        "-pix_fmt", "yuv420p", str(src)
    ], check=True, capture_output=True)

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)

    class StubEngine:
        def enhance(self, frame):
            return cv2.resize(frame, (128, 96), interpolation=cv2.INTER_NEAREST)

    monkeypatch.setattr(processor, "build_engine", lambda *a, **kw: StubEngine())

    results = processor.process_videos([src], "1x_Balanced", lambda *a: None)
    assert len(results["success"]) == 1
    out_file = results["success"][0]
    assert out_file.name == "small_1x.mp4"
    assert out_file.exists()

    probe_info = processor.probe(out_file)
    assert probe_info["width"] == 64
    assert probe_info["height"] == 48


def test_1x_wide_source_emits_oversized_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from video_upscaler import config, processor
    src = tmp_path / "wide.mp4"
    src.touch()

    # source width 4000 -> 4000 * 2 = 8000 > 7680 (MAX_OUTPUT_WIDTH_WARNING)
    fake_info = {
        "width": 4000,
        "height": 2000,
        "fps": 24.0,
        "duration": 0.1,
        "rotation": 0,
        "has_audio": False,
        "audio_codec": None,
        "codec_name": "h264",
    }
    monkeypatch.setattr(processor, "probe", lambda p: fake_info)
    monkeypatch.setattr(processor, "decode_frames", lambda p: iter([]))
    monkeypatch.setattr(processor, "build_engine", lambda *a, **kw: None)
    monkeypatch.setattr(processor, "_enhance_with_progress", lambda *a, **kw: iter([]))
    monkeypatch.setattr(processor, "encode_video", lambda *a, **kw: None)

    processor.process_videos([src], "1x_Balanced", lambda *a: None)
    captured = capsys.readouterr()
    assert "Warning: output width 8000px exceeds the 7680px threshold" in captured.out
