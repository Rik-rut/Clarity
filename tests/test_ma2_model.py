"""Checkpoint resolution and singleton cache for MatAnyone2."""

import numpy as np
import pytest

from video_upscaler import config
from video_upscaler.matanyone2 import model as ma2_model


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    root = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", root)
    return root


class FakeVendorModel:
    def __init__(self, ckpt, device):
        self.ckpt = ckpt
        self.device = device
        self.cfg = {"fake": True}


@pytest.fixture
def fake_loader(monkeypatch):
    calls = []

    def loader(ckpt_path, device):
        calls.append((ckpt_path, device))
        return FakeVendorModel(ckpt_path, device)

    monkeypatch.setattr(ma2_model, "_load_vendor_model", loader)
    return calls


@pytest.fixture(autouse=True)
def clean_cache():
    ma2_model.release_model()
    yield
    ma2_model.release_model()


def test_checkpoint_missing_raises_actionable(models_dir):
    with pytest.raises(ma2_model.MatAnyoneModelMissing) as excinfo:
        ma2_model.checkpoint_path()
    message = str(excinfo.value)
    assert "matanyone2.pth" in message
    assert "--download-models" in message


def test_get_model_caches_singleton(models_dir, fake_loader):
    ckpt_dir = models_dir / "matanyone"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "matanyone2.pth"
    ckpt.write_bytes(b"fake-weights")

    first = ma2_model.get_model("cpu")
    second = ma2_model.get_model("cpu")

    assert first is second
    assert len(fake_loader) == 1


def test_different_device_loads_separately(models_dir, fake_loader):
    ckpt_dir = models_dir / "matanyone"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "matanyone2.pth").write_bytes(b"fake-weights")

    cpu_model = ma2_model.get_model("cpu")
    cuda_model = ma2_model.get_model("cuda")

    assert cpu_model is not cuda_model
    assert len(fake_loader) == 2


def test_release_model_clears_cache(models_dir, fake_loader):
    ckpt_dir = models_dir / "matanyone"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "matanyone2.pth").write_bytes(b"fake-weights")

    first = ma2_model.get_model("cpu")
    ma2_model.release_model()
    second = ma2_model.get_model("cpu")

    assert first is not second
    assert len(fake_loader) == 2


def test_manifest_has_matanyone_group_entry():
    from video_upscaler.modelhub import entries, load_manifest

    manifest = load_manifest()
    group_entries = entries(manifest, group="matanyone")
    assert len(group_entries) == 1
    entry = group_entries[0]
    assert entry["path"] == "matanyone/matanyone2.pth"
    assert entry["dest"] == "matanyone2.pth"
    # MatAnyone2 ships in the essential download set (with SAM) so the
    # Easy Mask tab works after the standard setup.
    assert entry["tier"] == "essential"
    assert int(entry["size"]) > 0
    assert len(entry["sha256"]) == 64


def test_group_root_resolves_matanyone(models_dir):
    from video_upscaler.modelhub import _group_root

    assert _group_root("matanyone") == config.MODELS_DIR / "matanyone"


def test_registry_constants_exist():
    from video_upscaler.models import MATANYONE_CKPT_NAME, MATANYONE_SOURCE_URL

    assert MATANYONE_CKPT_NAME == "matanyone2.pth"
    assert MATANYONE_SOURCE_URL.startswith("https://github.com/pq-yang/MatAnyone2/releases/")
