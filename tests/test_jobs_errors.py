"""Job failure reporting surfaces real reasons, not just file names."""

from video_upscaler.web.jobs import _format_failed


def test_includes_names_and_first_reason_line():
    failed = [
        ("a.mp4", "ModuleNotFoundError: No module named 'omegaconf'\ntraceback..."),
        ("b.mp4", "FFmpeg encoding failed"),
    ]
    msg = _format_failed(failed)
    assert msg.startswith("Failed: ")
    assert "a.mp4: ModuleNotFoundError: No module named 'omegaconf'" in msg
    assert "traceback" not in msg  # only the first line is surfaced
    assert "b.mp4: FFmpeg encoding failed" in msg


def test_long_reasons_are_truncated():
    failed = [("x.mp4", "y" * 1000)]
    msg = _format_failed(failed)
    assert len(msg) < 500
    assert msg.endswith("…")


def test_missing_reason_falls_back():
    assert "z.mp4" in _format_failed([("z.mp4", "")])
