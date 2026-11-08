"""Torch-free unit tests for the aria2c backend (no network, no real aria2)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from video_upscaler import aria2

# tests/conftest.py disables aria2 discovery for every test; discovery tests
# below need the genuine implementation.
_REAL_FIND = aria2.find_aria2c


@pytest.fixture()
def real_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aria2, "find_aria2c", _REAL_FIND)


def _fake_exe(tmp_path: Path, body: str, code: int = 0) -> Path:
    """A stand-in for aria2c: runs ``body`` (python source) and exits ``code``."""
    script = tmp_path / "fake_aria2.py"
    exit_line = f"sys.exit({code})\n" if code else ""
    script.write_text(body + exit_line, encoding="utf-8")
    if os.name == "nt":
        bat = tmp_path / "fake_aria2.bat"
        bat.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return bat
    sh = tmp_path / "fake_aria2.sh"
    sh.write_text(f'#!/bin/sh\n"{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    sh.chmod(0o755)
    return sh


def _make_aria2_file(tmp_path: Path, name: str) -> Path:
    exe = tmp_path / (f"{name}.exe" if os.name == "nt" else name)
    exe.write_text("", encoding="utf-8")
    if os.name != "nt":
        exe.chmod(0o755)
    return exe


# --- 2.1 find_aria2c -------------------------------------------------------


def test_find_aria2c_from_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_discovery: None
) -> None:
    exe = _make_aria2_file(tmp_path, "aria2c")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv(aria2.ARIA2_ENV, raising=False)
    assert aria2.find_aria2c() == exe


def test_find_aria2c_not_found(
    monkeypatch: pytest.MonkeyPatch, real_discovery: None
) -> None:
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv(aria2.ARIA2_ENV, raising=False)
    assert aria2.find_aria2c() is None


def test_find_aria2c_env_override_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_discovery: None
) -> None:
    on_path = _make_aria2_file(tmp_path, "aria2c")
    elsewhere = tmp_path / "custom-aria2c.exe"
    elsewhere.write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv(aria2.ARIA2_ENV, str(elsewhere))
    assert aria2.find_aria2c() == elsewhere


def test_find_aria2c_env_override_missing_falls_back_to_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_discovery: None
) -> None:
    on_path = _make_aria2_file(tmp_path, "aria2c")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv(aria2.ARIA2_ENV, str(tmp_path / "nope.exe"))
    assert aria2.find_aria2c() == on_path


def test_find_aria2c_env_override_absent_when_path_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_discovery: None
) -> None:
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv(aria2.ARIA2_ENV, str(tmp_path / "nope.exe"))
    assert aria2.find_aria2c() is None


# --- 2.2 build_command + input-file writer --------------------------------


def test_build_command_flags() -> None:
    argv = aria2.build_command(
        Path("aria2c.exe"), Path("req.txt"), Path("models"), timeout_s=60
    )
    assert argv[:3] == ["aria2c.exe", "--input-file", "req.txt"]
    assert argv[3:5] == ["-d", "models"]
    assert "-x16" in argv and "-s16" in argv and "-k1M" in argv
    assert "--min-split-size=1M" in argv
    assert "-c" in argv
    assert "--allow-overwrite=true" in argv
    assert "--auto-file-renaming=false" in argv
    assert "--file-allocation=none" in argv
    assert "--connect-timeout=60" in argv
    assert "--max-tries=3" in argv
    assert "--retry-wait=1" in argv
    assert "--console-log-level=warn" in argv
    assert "--summary-interval=1" in argv
    assert "--download-result=hide" in argv
    # No --check-integrity: modelhub verifies sha256 for every backend.
    assert not any(a.startswith("--check-integrity") for a in argv)
    # The name comes from the input file's out= line, not from argv.
    assert not any(a.startswith("-o") or a.startswith("--out") for a in argv)


def test_build_input_text_carries_name_and_token() -> None:
    text = aria2.build_input_text(
        "https://huggingface.co/x/resolve/main/CLARITY_MODELS/a.pth",
        "a.pth.part",
        "secret-token",
    )
    assert text.splitlines()[0] == "https://huggingface.co/x/resolve/main/CLARITY_MODELS/a.pth"
    assert text.splitlines()[1] == "  out=a.pth.part"
    assert text.splitlines()[2] == "  header=Authorization: Bearer secret-token"


def test_token_never_in_argv_only_in_input_file(tmp_path: Path) -> None:
    part = tmp_path / "models" / "sam.pth.part"
    part.parent.mkdir()
    input_file = part.with_name(part.name + ".aria2-req")
    token = "secret-token"
    with aria2.temp_input_file(
        input_file, "https://huggingface.co/x/resolve/main/CLARITY_MODELS/sam.pth",
        part.name, token,
    ):
        argv = aria2.build_command(Path("aria2c.exe"), input_file, part.parent)
        assert token not in " ".join(argv)
        assert token in input_file.read_text(encoding="utf-8")
    assert not input_file.exists()


def test_token_not_sent_to_non_huggingface_host(tmp_path: Path) -> None:
    text = aria2.build_input_text(
        "https://mirror.example/models/sam.pth", "sam.pth.part", "secret-token"
    )
    assert "Authorization" not in text
    assert "secret-token" not in text
    assert "  out=sam.pth.part" in text


def test_hf_token_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in aria2.TOKEN_ENVS:
        monkeypatch.delenv(name, raising=False)
    assert aria2.hf_token() is None
    monkeypatch.setenv("HF_TOKEN", "tok-from-hf")
    assert aria2.hf_token() == "tok-from-hf"
    monkeypatch.setenv("CLARITY_HF_TOKEN", "tok-first")
    assert aria2.hf_token() == "tok-first"


# --- 2.3 parse_progress ----------------------------------------------------


CAPTURED = (
    "\r[#03d4fa 0B/0B CN:1 DL:0B]"
    "\r[#03d4fa 122MiB/357MiB(34%) CN:16 DL:9.5MiB ETA:24s]"
    "\r[#03d4fa 352MiB/357MiB(98%) CN:13 DL:27MiB]"
    "\n*** Download Progress Summary as of Sun Nov 08 04:10:40 2026 ***\n"
    "FILE: models/sam_vit_b_01ec64.pth\n"
)


def test_parse_progress_mib_units() -> None:
    assert aria2.parse_progress(CAPTURED) == [
        (0, None),
        (122 * 1024**2, 357 * 1024**2),
        (352 * 1024**2, 357 * 1024**2),
    ]


def test_parse_progress_gib_and_kib_units() -> None:
    assert aria2.parse_progress("[#a 1.5GiB/4.5GiB(33%) CN:8 DL:2MiB]") == [
        (int(1.5 * 1024**3), int(4.5 * 1024**3))
    ]
    assert aria2.parse_progress("[#a 512KiB/1MiB]") == [(512 * 1024, 1024**2)]


def test_parse_progress_ignores_garbage() -> None:
    noise = (
        "aria2 will resume download if the transfer is restarted.\n"
        "Download Results:\ngid   |stat|avg speed  |path/URI\n"
        "abcdef|ERR |       0B/s|https://example/x\n"
        "[#feedface not-a-number/4MiB]\n"
    )
    assert aria2.parse_progress(noise) == []


def test_parse_progress_strips_ansi() -> None:
    line = "\x1b[1;36mINFO\x1b[0m\r[#ab 8MiB/16MiB(50%) CN:4 DL:8MiB]"
    assert aria2.parse_progress(line) == [(8 * 1024**2, 16 * 1024**2)]


# --- 2.4 run_download ------------------------------------------------------


def test_run_download_success(tmp_path: Path) -> None:
    part = tmp_path / "models" / "sam.pth.part"
    part.parent.mkdir()
    exe = _fake_exe(
        tmp_path,
        "import sys\n"
        "sys.stdout.write(chr(13) + '[#a 10MiB/40MiB(25%) CN:4]'"
        " + chr(13) + '[#a 40MiB/40MiB(100%)]' + chr(10))\n"
        "sys.stdout.flush()\n",
    )

    seen: list[tuple[int, int | None]] = []
    printed: list[tuple[str, int, int]] = []
    aria2.run_download(
        "https://huggingface.co/x/resolve/main/CLARITY_MODELS/sam.pth",
        part,
        "Downloading sam.pth",
        40 * 1024**2,
        progress_cb=lambda done, total: seen.append((done, total)),
        exe=exe,
        print_cb=lambda label, done, total: printed.append((label, done, total)),
    )

    assert seen == [
        (10 * 1024**2, 40 * 1024**2),
        (40 * 1024**2, 40 * 1024**2),
        (40 * 1024**2, 40 * 1024**2),
    ]
    assert [p[1] for p in printed] == sorted(p[1] for p in printed)
    assert printed[0][0] == "Downloading sam.pth"
    # The temporary request file is removed even on success.
    assert not part.with_name(part.name + ".aria2-req").exists()


def test_run_download_nonzero_exit(tmp_path: Path) -> None:
    part = tmp_path / "models" / "sam.pth.part"
    part.parent.mkdir()
    exe = _fake_exe(
        tmp_path,
        "import sys\n"
        "sys.stdout.write(chr(13) + '[#a 5MiB/40MiB(12%) CN:4]'"
        " + chr(10) + 'Download aborted.' + chr(10))\n",
        7,
    )

    with pytest.raises(aria2.Aria2Error, match="exited with code 7"):
        aria2.run_download(
            "https://huggingface.co/x/resolve/main/a.pth",
            part,
            "Downloading a.pth",
            40 * 1024**2,
            exe=exe,
            print_cb=lambda *args: None,
        )
    assert not part.with_name(part.name + ".aria2-req").exists()


def test_run_download_unparseable_output_still_completes(tmp_path: Path) -> None:
    part = tmp_path / "models" / "sam.pth.part"
    part.parent.mkdir()
    exe = _fake_exe(tmp_path, "import sys\nprint('some banner')\nprint('not a readout line')\n")

    seen: list[tuple[int, int | None]] = []
    aria2.run_download(
        "https://huggingface.co/x/resolve/main/a.pth",
        part,
        "Downloading a.pth",
        1000,
        progress_cb=lambda done, total: seen.append((done, total)),
        exe=exe,
        print_cb=lambda *args: None,
    )
    # No readout parsed, but the file is reported as done when it finishes.
    assert seen == [(1000, 1000)]


def test_run_download_requires_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = tmp_path / "models" / "sam.pth.part"
    part.parent.mkdir()
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv(aria2.ARIA2_ENV, raising=False)
    with pytest.raises(aria2.Aria2Error, match="not installed"):
        aria2.run_download(
            "https://huggingface.co/x/resolve/main/a.pth",
            part,
            "Downloading a.pth",
            10,
            print_cb=lambda *args: None,
        )


def test_run_download_missing_input_never_leaks_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The request file exists only while the process runs, and argv has no token."""
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    part = tmp_path / "models" / "sam.pth.part"
    part.parent.mkdir()
    exe = _fake_exe(tmp_path, "import sys\nprint(chr(13) + '[#a 1MiB/2MiB]' + chr(10))\n")

    aria2.run_download(
        "https://huggingface.co/x/resolve/main/CLARITY_MODELS/sam.pth",
        part,
        "Downloading sam.pth",
        2 * 1024**2,
        exe=exe,
        print_cb=lambda *args: None,
    )
    request = part.with_name(part.name + ".aria2-req")
    assert not request.exists()
    leaked = [
        p for p in tmp_path.rglob("*")
        if p.is_file() and "secret-token" in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert leaked == []


def test_run_download_request_file_carries_out_line_during_run(tmp_path: Path) -> None:
    """aria2 gets the name through the input file's ``out=`` line, not argv."""
    part = tmp_path / "models" / "sam.pth.part"
    part.parent.mkdir()
    snapshot = tmp_path / "seen_request.txt"
    exe = _fake_exe(
        tmp_path,
        "import sys, pathlib\n"
        "args = sys.argv\n"
        "path = pathlib.Path(args[args.index('--input-file') + 1])\n"
        "path.with_name(path.name + '.seen').write_text(path.read_text(encoding='utf-8'))\n"
        "print(chr(13) + '[#a 1MiB/2MiB]' + chr(10))\n",
    )
    request = part.with_name(part.name + ".aria2-req")
    aria2.run_download(
        "https://huggingface.co/x/resolve/main/CLARITY_MODELS/sam.pth",
        part,
        "Downloading sam.pth",
        2 * 1024**2,
        exe=exe,
        print_cb=lambda *args: None,
    )
    text = request.with_name(request.name + ".seen").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "https://huggingface.co/x/resolve/main/CLARITY_MODELS/sam.pth"
    assert "  out=sam.pth.part" in text
    assert "-o" not in text and "--out" not in text
    assert not request.exists()
