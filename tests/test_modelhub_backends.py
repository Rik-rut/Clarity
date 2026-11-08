"""Backend selection, fallback and resume behaviour in ``modelhub`` (torch-free)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_upscaler import aria2, config, modelhub


@pytest.fixture()
def manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    payload = {
        "version": 1,
        "repo": "Rikrut/clarity",
        "files": [
            {
                "group": "sam",
                "path": "sam/sam.pth",
                "dest": "sam.pth",
                "tier": "essential",
                "size": 6,
                "sha256": hashlib.sha256(b"abcdef").hexdigest(),
            },
            {
                "group": "amt",
                "path": "amt/amt-s.pth",
                "dest": "amt-s.pth",
                "tier": "essential",
                "size": 3,
                "sha256": hashlib.sha256(b"xyz").hexdigest(),
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(modelhub, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(modelhub, "_noticed", set())
    monkeypatch.delenv("CLARITY_DOWNLOADER", raising=False)
    monkeypatch.delenv("CLARITY_MODEL_HUB_BASE", raising=False)
    return payload


def _entry(manifest: dict, dest: str) -> dict:
    return next(e for e in modelhub.entries() if e["dest"] == dest)


def _dest(entry: dict) -> Path:
    return modelhub._group_root(entry["group"]) / entry["dest"]


def _raise(exc: BaseException):
    def boom(*args, **kwargs):
        raise exc

    return boom


# --- 3.1 selection ---------------------------------------------------------


def test_choice_defaults_to_auto(manifest: dict) -> None:
    assert modelhub.downloader_choice() == "auto"


@pytest.mark.parametrize("value", ["auto", "aria2", "hf", "http", "HTTP"])
def test_choice_accepts_documented_values(
    manifest: dict, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CLARITY_DOWNLOADER", value)
    assert modelhub.downloader_choice() == value.lower()


def test_choice_invalid_warns_and_uses_auto(
    manifest: dict, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("CLARITY_DOWNLOADER", "everything")
    assert modelhub.downloader_choice() == "auto"
    out = capsys.readouterr().out
    assert "Unrecognised CLARITY_DOWNLOADER" in out
    assert "auto|aria2|hf|http" in out


def test_auto_order_prefers_aria2_then_hf_then_http(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))
    monkeypatch.setattr(modelhub, "_hf_downloader", lambda: object())

    big = {"path": "sam/sam.pth", "size": modelhub.HF_MIN_BYTES}
    assert modelhub.download_backends(big) == ["aria2", "hf", "http"]

    small = {"path": "amt/amt-s.pth", "size": modelhub.HF_MIN_BYTES - 1}
    assert modelhub.download_backends(small) == ["aria2", "http"]


def test_auto_order_without_aria2(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aria2, "find_aria2c", lambda: None)
    monkeypatch.setattr(modelhub, "_hf_downloader", lambda: object())
    big = {"path": "sam/sam.pth", "size": modelhub.HF_MIN_BYTES}
    assert modelhub.download_backends(big) == ["hf", "http"]


def test_forced_http_skips_accelerated_backends(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLARITY_DOWNLOADER", "http")
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))
    monkeypatch.setattr(modelhub, "_hf_downloader", lambda: object())
    big = {"path": "sam/sam.pth", "size": modelhub.HF_MIN_BYTES}
    assert modelhub.download_backends(big) == ["http"]


def test_forced_aria2_unavailable_degrades_to_auto_with_notice(
    manifest: dict, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("CLARITY_DOWNLOADER", "aria2")
    monkeypatch.setattr(aria2, "find_aria2c", lambda: None)
    monkeypatch.setattr(modelhub, "_hf_downloader", lambda: object())
    big = {"path": "sam/sam.pth", "size": modelhub.HF_MIN_BYTES}
    assert modelhub.download_backends(big) == ["hf", "http"]
    assert "not installed" in capsys.readouterr().out


def test_forced_hf_unavailable_degrades_to_auto_with_notice(
    manifest: dict, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("CLARITY_DOWNLOADER", "hf")
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))
    monkeypatch.setattr(modelhub, "_hf_downloader", lambda: object())
    small = {"path": "amt/amt-s.pth", "size": 3}
    assert modelhub.download_backends(small) == ["aria2", "http"]
    assert "Hugging Face path is unavailable" in capsys.readouterr().out


def test_forced_hf_keeps_plain_http_as_fallback(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLARITY_DOWNLOADER", "hf")
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))
    monkeypatch.setattr(modelhub, "_hf_downloader", lambda: object())
    big = {"path": "sam/sam.pth", "size": modelhub.HF_MIN_BYTES}
    assert modelhub.download_backends(big) == ["hf", "http"]


def test_local_hub_never_starts_a_network_downloader(
    tmp_path: Path, manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_dir = tmp_path / "hub"
    (hub_dir / "amt").mkdir(parents=True)
    (hub_dir / "amt" / "amt-s.pth").write_bytes(b"xyz")
    monkeypatch.setenv("CLARITY_MODEL_HUB_BASE", str(hub_dir))
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))

    def forbidden(*args, **kwargs):
        raise AssertionError("a network downloader must not run for a local hub")

    monkeypatch.setattr(aria2, "run_download", forbidden)
    monkeypatch.setattr(modelhub, "_download_hf", forbidden)
    monkeypatch.setattr(modelhub, "_download_http", forbidden)

    installed = modelhub.install_entry(_entry(manifest, "amt-s.pth"))
    assert installed.read_bytes() == b"xyz"


# --- 3.2 fallback ----------------------------------------------------------


def test_aria2_failure_falls_back_to_hf(
    manifest: dict, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))
    monkeypatch.setattr(modelhub, "_should_use_hf", lambda entry: True)

    def aria2_boom(*args, **kwargs):
        raise aria2.Aria2Error("aria2c exited with code 22")

    def fake_hf(entry, staging_dir, progress_cb=None):
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged = staging_dir / entry["dest"]
        staged.write_bytes(b"abcdef")
        return staged

    def http_forbidden(*args, **kwargs):
        raise AssertionError("plain HTTP must not run when Xet succeeds")

    monkeypatch.setattr(aria2, "run_download", aria2_boom)
    monkeypatch.setattr(modelhub, "_download_hf", fake_hf)
    monkeypatch.setattr(modelhub, "_download_http", http_forbidden)

    installed = modelhub.install_entry(_entry(manifest, "sam.pth"))
    assert installed.read_bytes() == b"abcdef"
    assert "Accelerated download unavailable" in capsys.readouterr().out


def test_aria2_and_hf_fail_falls_back_to_http(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))
    monkeypatch.setattr(modelhub, "_should_use_hf", lambda entry: True)

    def aria2_boom(*args, **kwargs):
        raise aria2.Aria2Error("network problem")

    def hf_boom(entry, staging_dir, progress_cb=None):
        raise RuntimeError("xet unavailable")

    def fake_http(url, dest, label, expected_size, progress_cb=None):
        dest.write_bytes(b"abcdef")

    monkeypatch.setattr(aria2, "run_download", aria2_boom)
    monkeypatch.setattr(modelhub, "_download_hf", hf_boom)
    monkeypatch.setattr(modelhub, "_download_http", fake_http)

    installed = modelhub.install_entry(_entry(manifest, "sam.pth"))
    assert installed.read_bytes() == b"abcdef"


def test_every_backend_fails_reports_actionable_error_and_installs_nothing(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))
    monkeypatch.setattr(modelhub, "_should_use_hf", lambda entry: False)
    monkeypatch.setattr(aria2, "run_download", _raise(aria2.Aria2Error("nope")))
    monkeypatch.setattr(modelhub, "_download_http", _raise(OSError("connection reset")))

    entry = _entry(manifest, "sam.pth")
    with pytest.raises(modelhub.HubError, match="sam/sam.pth"):
        modelhub.install_entry(entry)

    dest = _dest(entry)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
    assert not dest.with_name(dest.name + ".part.aria2").exists()


# --- 3.3 resume ------------------------------------------------------------


def test_interrupted_aria2_keeps_partial_for_resume(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))
    monkeypatch.setattr(modelhub, "_should_use_hf", lambda entry: False)
    monkeypatch.setenv("CLARITY_DOWNLOADER", "aria2")

    entry = _entry(manifest, "sam.pth")
    dest = _dest(entry)
    part = dest.with_name(dest.name + ".part")
    control = dest.with_name(dest.name + ".part.aria2")

    def interrupted(url, part_path, label, expected_size, progress_cb=None, **kwargs):
        part_path.write_bytes(b"abc")  # partial data, aria2 would leave .aria2 too
        control.write_bytes(b"control")
        raise aria2.Aria2Error("interrupted")

    monkeypatch.setattr(aria2, "run_download", interrupted)
    monkeypatch.setattr(modelhub, "_download_http", _raise(OSError("down")))

    with pytest.raises(modelhub.HubError):
        modelhub.install_entry(entry)

    assert part.read_bytes() == b"abc"
    assert control.is_file()
    assert not dest.exists()

    # Second attempt resumes (aria2 continues from the partial) and installs.
    resumed_calls: list[int] = []

    def resumed(url, part_path, label, expected_size, progress_cb=None, **kwargs):
        resumed_calls.append(part_path.stat().st_size)
        part_path.write_bytes(b"abcdef")
        control.unlink(missing_ok=True)

    monkeypatch.setattr(aria2, "run_download", resumed)
    installed = modelhub.install_entry(entry)
    assert resumed_calls == [3], "aria2 should see the partial data on retry"
    assert installed.read_bytes() == b"abcdef"
    assert not part.exists()
    assert not control.exists()


def test_corrupt_resumed_partial_is_deleted(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))
    monkeypatch.setenv("CLARITY_DOWNLOADER", "aria2")

    entry = _entry(manifest, "sam.pth")
    dest = _dest(entry)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    control = dest.with_name(dest.name + ".part.aria2")
    part.write_bytes(b"zzzzzz")
    control.write_bytes(b"control")

    def resumes_but_corrupt(url, part_path, label, expected_size, progress_cb=None, **kwargs):
        assert part_path.stat().st_size == 6, "aria2 should resume the existing partial"
        part_path.write_bytes(b"zzzzzz")  # same corrupt content, "completed"

    monkeypatch.setattr(aria2, "run_download", resumes_but_corrupt)

    with pytest.raises(modelhub.HubError, match="integrity"):
        modelhub.install_entry(entry)

    assert not part.exists(), "a corrupt partial must not survive"
    assert not control.exists()
    assert not dest.exists()


def test_wrong_size_deletes_aria2_partial(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aria2, "find_aria2c", lambda: Path("aria2c.exe"))
    monkeypatch.setenv("CLARITY_DOWNLOADER", "aria2")

    entry = _entry(manifest, "sam.pth")
    dest = _dest(entry)
    part = dest.with_name(dest.name + ".part")
    control = dest.with_name(dest.name + ".part.aria2")

    def short(url, part_path, label, expected_size, progress_cb=None, **kwargs):
        part_path.write_bytes(b"ab")
        control.write_bytes(b"control")

    monkeypatch.setattr(aria2, "run_download", short)

    with pytest.raises(modelhub.HubError, match="wrong size"):
        modelhub.install_entry(entry)
    assert not part.exists()
    assert not control.exists()


def test_http_and_hf_cleanup_is_unchanged(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-aria2 backends still delete their temp file and staging dir."""
    monkeypatch.setattr(aria2, "find_aria2c", lambda: None)

    def fake_http(url, dest, label, expected_size, progress_cb=None):
        dest.write_bytes(b"xyz")

    monkeypatch.setattr(modelhub, "_download_http", fake_http)
    installed = modelhub.install_entry(_entry(manifest, "amt-s.pth"))
    assert installed.read_bytes() == b"xyz"
    assert not installed.with_name(installed.name + ".part").exists()
    assert not installed.with_name(installed.name + ".part.aria2").exists()
    assert not installed.with_name(installed.name + ".staging").exists()

    # A failing http download leaves nothing behind.
    installed.unlink()
    monkeypatch.setattr(modelhub, "_download_http", _raise(OSError("reset")))
    entry = _entry(manifest, "amt-s.pth")
    dest = _dest(entry)
    with pytest.raises(modelhub.HubError):
        modelhub.install_entry(entry)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
