"""Tests for MultiPassDedup backend, plan building, validation, and CLI isolation."""

from __future__ import annotations

from pathlib import Path
import pytest

from video_upscaler import config
from video_upscaler.dedup_backend import (
    DEDUP_MODELS,
    check_dedup_weights,
    detect_dedup_device,
    get_dedup_weights_path,
    parse_npass,
    validate_model_type,
)
from video_upscaler.dedup import build_dedup_plan, ensure_dedup_weights


def test_validate_model_type() -> None:
    assert validate_model_type("gmfss") == "gmfss"
    assert validate_model_type("GMFSS") == "gmfss"
    assert validate_model_type("rife") == "rife"
    assert validate_model_type("RIFE") == "rife"
    assert validate_model_type("gimm") == "gimm"
    assert validate_model_type("  GIMM  ") == "gimm"

    with pytest.raises(ValueError, match="Invalid MultiPassDedup model"):
        validate_model_type("cugan")

    with pytest.raises(ValueError, match="Invalid MultiPassDedup model"):
        validate_model_type("amt-s")


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


def test_get_dedup_weights_path(tmp_path: Path) -> None:
    config.DEDUP_MODELS_DIR = tmp_path
    assert get_dedup_weights_path("gmfss") == tmp_path / "train_log_pg104"
    assert get_dedup_weights_path("rife") == tmp_path / "rife48.pkl"
    assert get_dedup_weights_path("gimm") == tmp_path / "gimmvfi_r_arb_lpips.pt"


def test_check_dedup_weights_missing_and_present(tmp_path: Path) -> None:
    config.DEDUP_MODELS_DIR = tmp_path
    msg = check_dedup_weights("gmfss")
    assert msg is not None
    assert "train_log_pg104" in msg

    # Create weights directory
    (tmp_path / "train_log_pg104").mkdir()
    assert check_dedup_weights("gmfss") is None

    msg_rife = check_dedup_weights("rife")
    assert msg_rife is not None
    assert "rife48.pkl" in msg_rife

    (tmp_path / "rife48.pkl").touch()
    assert check_dedup_weights("rife") is None


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
    (tmp_path / "train_log_pg104").mkdir()

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
    (tmp_path / "train_log_pg104").mkdir()
    monkeypatch.setattr("video_upscaler.dedup._interactive", lambda: False)

    amt_sel_before = select_amt_backend("AMT-S")
    _ = build_dedup_plan()
    amt_sel_after = select_amt_backend("AMT-S")

    assert amt_sel_before.backend == amt_sel_after.backend
    assert amt_sel_before.precision == amt_sel_after.precision
    assert amt_sel_before.batch_size == amt_sel_after.batch_size
