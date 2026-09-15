"""Engine-preparation stage callbacks keep the render UI from freezing."""

from __future__ import annotations

from video_upscaler import processor


def test_process_videos_reports_engine_stages(monkeypatch) -> None:
    stages: list[str] = []
    monkeypatch.setattr(processor, "detect_backend", lambda: "torch-cpu")

    def fake_build(profile, backend, stage_cb=None):
        if stage_cb is not None:
            stage_cb("Loading the Real-CUGAN model…")
        return object()

    monkeypatch.setattr(processor, "build_engine", fake_build)

    result = processor.process_videos(
        [], "2x_Balanced", lambda *args: None, stage_cb=stages.append
    )

    assert result == {"success": [], "failed": [], "times": []}
    assert stages[0] == "Detecting the compute backend…"
    assert "Loading the Real-CUGAN model…" in stages
