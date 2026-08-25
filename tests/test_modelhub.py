"""Torch-free tests for the central model hub (manifest, filtering, installs)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_upscaler import config, modelhub


@pytest.fixture()
def manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    payload = {
        "version": 1,
        "repo": "Rik-rut/clarity-models",
        "files": [
            {
                "group": "cugan",
                "path": "cugan/up2x-latest-no-denoise.pth",
                "dest": "up2x-latest-no-denoise.pth",
                "tier": "essential",
                "size": 4,
                "sha256": hashlib.sha256(b"abcd").hexdigest(),
            },
            {
                "group": "amt",
                "path": "amt/amt-s.pth",
                "dest": "amt-s.pth",
                "tier": "essential",
                "size": 3,
                "sha256": hashlib.sha256(b"xyz").hexdigest(),
            },
            {
                "group": "amt",
                "path": "amt/amt-g.pth",
                "dest": "amt-g.pth",
                "tier": "full",
                "size": 5,
                "sha256": hashlib.sha256(b"abcde").hexdigest(),
            },
            {
                "group": "dedup",
                "path": "dedup/train_log_pg104/metric.pkl",
                "dest": "train_log_pg104/metric.pkl",
                "tier": "full",
                "size": 2,
                "sha256": hashlib.sha256(b"ok").hexdigest(),
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(modelhub, "MANIFEST_PATH", manifest_path)
    return payload


def test_hub_base_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLARITY_MODEL_HUB_BASE", raising=False)
    assert modelhub.hub_base() == modelhub.DEFAULT_HUB_BASE
    monkeypatch.setenv("CLARITY_MODEL_HUB_BASE", "https://mirror.example/models")
    assert modelhub.hub_base() == "https://mirror.example/models"


def test_local_dir_base_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLARITY_MODEL_HUB_BASE", str(tmp_path))
    assert modelhub._local_base_dir(modelhub.hub_base()) == tmp_path
    monkeypatch.setenv(
        "CLARITY_MODEL_HUB_BASE", (tmp_path / "missing").as_posix()
    )
    assert modelhub._local_base_dir(modelhub.hub_base()) is None


def test_entries_filtering(manifest: dict) -> None:
    assert len(modelhub.entries()) == 4
    assert {e["dest"] for e in modelhub.entries(group="amt")} == {"amt-s.pth", "amt-g.pth"}
    essential = modelhub.entries(tier="essential")
    assert {e["group"] for e in essential} == {"cugan", "amt"}
    assert all(e["tier"] == "essential" for e in essential)


def test_group_roots(manifest: dict) -> None:
    assert modelhub._group_root("cugan") == config.MODELS_DIR
    assert modelhub._group_root("dedup") == config.DEDUP_MODELS_DIR
    with pytest.raises(modelhub.HubError):
        modelhub._group_root("bogus")


def test_install_entry_from_local_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict,
) -> None:
    hub_dir = tmp_path / "hub"
    source = hub_dir / "amt"
    source.mkdir(parents=True)
    source.joinpath("amt-s.pth").write_bytes(b"xyz")
    monkeypatch.setenv("CLARITY_MODEL_HUB_BASE", str(hub_dir))

    models_dir = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)

    entry = next(e for e in modelhub.entries(group="amt") if e["dest"] == "amt-s.pth")
    installed = modelhub.install_entry(entry)
    assert installed.read_bytes() == b"xyz"

    # Idempotent fast path: existing file with matching size is kept.
    assert modelhub.install_entry(entry) == installed

    # Corrupt local hub file fails hash verification (dest removed so the
    # size fast path does not short-circuit).
    installed.unlink()
    source.joinpath("amt-s.pth").write_bytes(b"bad")
    with pytest.raises(modelhub.HubError, match="integrity"):
        modelhub.install_entry(entry)


def test_missing_entries_detects_absent_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: dict
) -> None:
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    missing = modelhub.missing_entries(modelhub.entries(group="amt"))
    assert [e["dest"] for e in missing] == ["amt-s.pth", "amt-g.pth"]

    target = tmp_path / "models" / "amt-s.pth"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"xyz")
    missing = modelhub.missing_entries(modelhub.entries(group="amt"))
    assert [e["dest"] for e in missing] == ["amt-g.pth"]


def test_load_manifest_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modelhub, "MANIFEST_PATH", Path("/nonexistent.json"))
    with pytest.raises(modelhub.HubError, match="manifest not found"):
        modelhub.load_manifest()


def test_download_size_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: dict
) -> None:
    """A local hub file with wrong content length fails the size gate."""
    hub_dir = tmp_path / "hub"
    source = hub_dir / "amt"
    source.mkdir(parents=True)
    source.joinpath("amt-s.pth").write_bytes(b"wrong-size")
    monkeypatch.setenv("CLARITY_MODEL_HUB_BASE", str(hub_dir))
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    entry = next(e for e in modelhub.entries(group="amt") if e["dest"] == "amt-s.pth")
    with pytest.raises(modelhub.HubError, match="wrong size"):
        modelhub.install_entry(entry)


def test_real_manifest_is_valid_and_complete() -> None:
    """The shipped manifest must describe every profile + dedup weight file."""
    from video_upscaler.dedup_backend import DEDUP_MODEL_WEIGHT_FILES
    from video_upscaler.models import INTERP_MODELS, PROFILES

    entries = modelhub.entries()
    dests = {(e["group"], str(e["dest"])) for e in entries}
    for _, (model_name, _) in PROFILES.items():
        assert ("cugan", model_name) in dests
    for _, (ckpt_name, _) in INTERP_MODELS.items():
        assert ("amt", ckpt_name) in dests
    for key, filename in DEDUP_MODEL_WEIGHT_FILES.items():
        assert any(g == "dedup" and d.startswith(filename) for g, d in dests), key


def test_dedup_entry_mapping(manifest: dict) -> None:
    from video_upscaler.dedup import _dedup_entries_for

    gmfss = [str(e["dest"]) for e in _dedup_entries_for("gmfss")]
    assert gmfss == ["train_log_pg104/metric.pkl"]
    rife = [str(e["dest"]) for e in _dedup_entries_for("rife")]
    assert rife == []
    gimm = sorted(str(e["dest"]) for e in _dedup_entries_for("gimm"))
    assert "train_log_pg104/metric.pkl" not in gimm
