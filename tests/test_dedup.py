"""Tests for MultiPassDedup backend, plan building, validation, and CLI isolation."""

from __future__ import annotations

from pathlib import Path
import pytest

from video_upscaler import config
from video_upscaler.dedup_backend import (
    DEDUP_MODELS,
    check_dedup_weights,
    detect_dedup_device,
    parse_npass,
    validate_model_type,
)
from video_upscaler.dedup import build_dedup_plan, ensure_dedup_weights

GMFSS_FILES = (
    "train_log_pg104/feat.pkl",
    "train_log_pg104/flownet.pkl",
    "train_log_pg104/fusionnet.pkl",
    "train_log_pg104/metric.pkl",
    "train_log_pg104/rife.pkl",
)


def _install_gmfss_weights(tmp_path: Path, only: tuple[str, ...] = GMFSS_FILES) -> None:
    for rel in only:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_validate_model_type() -> None:
    assert validate_model_type("gmfss") == "gmfss"
    assert validate_model_type("GMFSS") == "gmfss"
    assert validate_model_type("rife") == "rife"
    assert validate_model_type("RIFE") == "rife"

    with pytest.raises(ValueError, match="Invalid MultiPassDedup model"):
        validate_model_type("cugan")

    with pytest.raises(ValueError, match="Invalid MultiPassDedup model"):
        validate_model_type("amt-s")

    with pytest.raises(ValueError, match="Invalid MultiPassDedup model"):
        validate_model_type("gimm")


def test_parse_npass_mappings() -> None:
    assert parse_npass("auto") == 0
    assert parse_npass("AUTO") == 0
    assert parse_npass("0") == 0
    assert parse_npass(0) == 0
    assert parse_npass("2") == 2
    assert parse_npass(2) == 2
    assert parse_npass("3") == 3
    assert parse_npass(3) == 3
    assert parse_npass(4) == 4

    with pytest.raises(ValueError, match="Invalid npass"):
        parse_npass("invalid")

    with pytest.raises(ValueError, match="non-negative"):
        parse_npass(-1)

    with pytest.raises(TypeError, match="Expected str or int"):
        parse_npass(None)  # type: ignore[arg-type]


def test_check_dedup_weights_missing_and_present(tmp_path: Path) -> None:
    config.DEDUP_MODELS_DIR = tmp_path
    msg = check_dedup_weights("gmfss")
    assert msg is not None
    assert "train_log_pg104" in msg

    # Regression: a partial install (directory exists, some files missing)
    # must still be reported as incomplete, not silently pass.
    (tmp_path / "train_log_pg104").mkdir()
    _install_gmfss_weights(tmp_path, only=(
        "train_log_pg104/feat.pkl",
        "train_log_pg104/flownet.pkl",
        "train_log_pg104/fusionnet.pkl",
    ))
    msg_partial = check_dedup_weights("gmfss")
    assert msg_partial is not None
    assert "metric.pkl" in msg_partial
    assert "rife.pkl" in msg_partial

    _install_gmfss_weights(tmp_path)
    assert check_dedup_weights("gmfss") is None

    msg_rife = check_dedup_weights("rife")
    assert msg_rife is not None
    assert "rife48.pkl" in msg_rife

    (tmp_path / "rife48.pkl").touch()
    assert check_dedup_weights("rife") is None


def test_ensure_dedup_weights_repairs_only_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partially-installed model downloads only the absent weight files."""
    config.DEDUP_MODELS_DIR = tmp_path
    _install_gmfss_weights(tmp_path, only=(
        "train_log_pg104/feat.pkl",
        "train_log_pg104/flownet.pkl",
        "train_log_pg104/fusionnet.pkl",
    ))

    installed: list[str] = []

    def fake_install(entry):
        installed.append(str(entry["dest"]))

    monkeypatch.setattr("video_upscaler.modelhub.install_entry", fake_install)
    ensure_dedup_weights("gmfss", auto_download=True)

    assert sorted(installed) == [
        "train_log_pg104/metric.pkl",
        "train_log_pg104/rife.pkl",
    ]


def test_ensure_dedup_weights_raises_on_missing(tmp_path: Path) -> None:
    config.DEDUP_MODELS_DIR = tmp_path
    with pytest.raises(SystemExit) as excinfo:
        ensure_dedup_weights("gmfss")
    assert excinfo.value.code == 1


def test_detect_dedup_device_cpu_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PREFERRED_DEVICE", "cpu")
    assert detect_dedup_device() == "cpu"


def test_build_dedup_plan_non_interactive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config.DEDUP_MODELS_DIR = tmp_path
    _install_gmfss_weights(tmp_path)

    monkeypatch.setattr("video_upscaler.dedup._interactive", lambda: False)
    plan = build_dedup_plan()

    assert plan["action_label"] == "Interpolate (MultiPassDedup)"
    assert plan["engine_label"] == "MultiPassDedup (GMFSS)"
    assert plan["model"] == "gmfss"
    assert plan["npass"] == 0
    assert plan["factor"] == 2
    assert len(plan["header_lines"]) >= 4
    assert len(plan["summary_lines"]) >= 4


def test_dedup_and_amt_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify MultiPassDedup execution plan does not mutate AMT configuration."""
    from video_upscaler.interp import select_amt_backend

    config.DEDUP_MODELS_DIR = tmp_path
    _install_gmfss_weights(tmp_path)
    monkeypatch.setattr("video_upscaler.dedup._interactive", lambda: False)

    amt_sel_before = select_amt_backend("AMT-S")
    _ = build_dedup_plan()
    amt_sel_after = select_amt_backend("AMT-S")

    assert amt_sel_before.backend == amt_sel_after.backend
    assert amt_sel_before.precision == amt_sel_after.precision
    assert amt_sel_before.batch_size == amt_sel_after.batch_size
