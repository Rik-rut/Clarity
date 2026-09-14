"""Phase D: Slow-motion web jobs auto-download a missing AMT checkpoint.

Regression tests for the render that hard-failed with
``Required AMT model not found`` when the user picked AMT-G/L while only
the ``essential`` tier (AMT-S) was installed. The Slow-motion branch of
``JobManager._run_job_thread`` must ensure the *requested* model_key's
checkpoint exists before ``process_interpolate`` runs — mirroring the CLI
``_ensure_amt`` / MatAnyone2 first-use recovery.

The hub seam (``modelhub.install_entry``) is stubbed: it records calls
and materializes a fake checkpoint file. No real models are downloaded.
The heavy engine (``process_interpolate``) is stubbed so these tests
exercise job-level ensure logic only.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from video_upscaler import config
from video_upscaler.modelhub import HubError
from video_upscaler.web import jobs as jobs_mod
from video_upscaler.web.jobs import JobInfo, JobManager


def _run_slow_motion_job(tmp_path, monkeypatch, model_key, make_install, process_fake):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(
        "video_upscaler.modelhub.install_entry", make_install(models_dir)
    )
    monkeypatch.setattr(jobs_mod, "process_interpolate", process_fake)

    manager = JobManager()
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"input")
    params = {"model_key": model_key, "factor": 2}
    job = JobInfo(
        job_id="amt1",
        action="Slow-motion",
        video_names=["clip.mp4"],
        params=dict(params),
        output_dir=str(out_dir),
    )
    manager.jobs["amt1"] = job
    manager._run_job_thread("amt1", [src], params, out_dir)
    return job, models_dir


def _record_and_materialize(calls):
    def make(models_dir):
        def fake_install(entry):
            calls.append(entry)
            dest = models_dir / PurePosixPath(entry["dest"]).name
            dest.write_bytes(b"fake-checkpoint")
            return dest

        return fake_install

    return make


def _succeed(process_calls):
    def fake_process(videos, model_key, factor, progress_cb):
        process_calls.append((model_key, factor))
        progress_cb(1, 1, 100)
        return {"success": ["clip_slowed2x_amt.mp4"], "failed": [], "times": [0.1]}

    return fake_process


def test_slow_motion_job_fetches_missing_amt_checkpoint(tmp_path, monkeypatch):
    """Missing AMT-G checkpoint is fetched exactly once for the selected key,
    and the job proceeds past model resolution."""
    install_calls: list = []
    process_calls: list = []
    job, models_dir = _run_slow_motion_job(
        tmp_path,
        monkeypatch,
        "AMT-G",
        _record_and_materialize(install_calls),
        _succeed(process_calls),
    )

    assert len(install_calls) == 1
    assert PurePosixPath(install_calls[0]["dest"]).name == "amt-g.pth"
    assert (models_dir / "amt-g.pth").is_file()
    assert process_calls == [("AMT-G", 2)]
    assert job.status == "completed"
    assert job.error_message is None


def test_slow_motion_job_skips_fetch_when_checkpoint_present(
    tmp_path, monkeypatch
):
    """Present checkpoint (here AMT-S) triggers no hub fetch."""
    install_calls: list = []
    process_calls: list = []

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "amt-s.pth").write_bytes(b"already-installed")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)

    def fail_on_fetch(entry):
        raise AssertionError(f"no fetch expected, got {entry!r}")

    monkeypatch.setattr("video_upscaler.modelhub.install_entry", fail_on_fetch)
    monkeypatch.setattr(jobs_mod, "process_interpolate", _succeed(process_calls))

    manager = JobManager()
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"input")
    params = {"model_key": "AMT-S", "factor": 2}
    job = JobInfo(
        job_id="amt1",
        action="Slow-motion",
        video_names=["clip.mp4"],
        params=dict(params),
        output_dir=str(out_dir),
    )
    manager.jobs["amt1"] = job
    manager._run_job_thread("amt1", [src], params, out_dir)

    assert install_calls == []
    assert process_calls == [("AMT-S", 2)]
    assert job.status == "completed"


def test_slow_motion_job_fetch_failure_surfaces_as_job_error(tmp_path, monkeypatch):
    """A failed checkpoint fetch fails the job with the hub error (no crash,
    no engine call); the ``check_amt`` raise in the engine stays a backstop."""
    process_calls: list = []

    def make_failing(models_dir):
        def fake_install(entry):
            raise HubError("Failed to download amt/amt-g.pth:\nboom")

        return fake_install

    job, _ = _run_slow_motion_job(
        tmp_path, monkeypatch, "AMT-G", make_failing, _succeed(process_calls)
    )

    assert job.status == "failed"
    assert job.error_message is not None
    assert "amt-g.pth" in job.error_message
    assert process_calls == []
