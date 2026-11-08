"""aria2c backend: locate the binary, build its command, run it, read progress.

Torch-free and prompt-free so unit tests can exercise it without heavy imports.
``modelhub.py`` stays the only place that decides *which* backend runs; this
module only knows how to talk to aria2.

Two behaviours learned from the spike (see
``openspec/changes/speed-up-model-download/spike-results.md``):

- aria2 ignores a command-line ``-o`` for URIs read from ``--input-file``, so the
  output name is an ``out=`` line inside the entry (``-d`` is honoured).
- aria2 forwards custom headers across a cross-host redirect, so a token is only
  attached for Hugging Face hosts.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional
from urllib.parse import urlparse

ARIA2_NAME = "aria2c"
ARIA2_ENV = "CLARITY_ARIA2C"
TOKEN_ENVS = ("CLARITY_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# 16 is aria2's practical maximum; the spike measured 23.3 MB/s at 16 versus
# 12.0 MB/s at 8 against the Hugging Face CDN, with no throttling or 429s.
MAX_CONNECTIONS = 16
MIN_SPLIT_SIZE = "1M"
MAX_TRIES = 3
RETRY_WAIT_S = 1

HF_HOSTS = frozenset({"huggingface.co", "hf.co"})

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# e.g. "[#a1b2c3 120MiB/357MiB(33%) CN:16 DL:25MiB ETA:9s]"
READOUT_RE = re.compile(
    r"\[#[0-9a-f]+ (?P<done>[\d.]+)(?P<du>[KMG]?i?B)/"
    r"(?P<total>[\d.]+)(?P<tu>[KMG]?i?B)",
    re.IGNORECASE,
)

UNITS = {
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
}


class Aria2Error(RuntimeError):
    """aria2c could not be started or exited with an error."""


def find_aria2c() -> Optional[Path]:
    """Return the aria2c executable: ``CLARITY_ARIA2C`` first, then PATH.

    Desktop provisioning prepends the bundled ``resources`` directory to PATH
    (``SetupContext.child_env``), so a bundled ``aria2c.exe`` is found here with
    no extra plumbing. Returns ``None`` when nothing usable is installed.
    """
    override = os.environ.get(ARIA2_ENV, "").strip()
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
    found = shutil.which(ARIA2_NAME)
    return Path(found) if found else None


def hf_token() -> Optional[str]:
    """Return a configured Hugging Face token, or ``None``."""
    for name in TOKEN_ENVS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _is_hf_host(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host in HF_HOSTS or host.endswith(".huggingface.co")


def build_input_text(url: str, out_name: str, token: Optional[str] = None) -> str:
    """Build the ``--input-file`` body for one download.

    ``out=`` carries the file name because aria2 ignores a command-line ``-o``
    for input-file URIs. A token is written here (never into argv) and only for
    Hugging Face hosts, since aria2 forwards headers across redirects.
    """
    lines = [url, f"  out={out_name}"]
    if token and _is_hf_host(url):
        lines.append(f"  header=Authorization: Bearer {token}")
    return "\n".join(lines) + "\n"


def build_command(
    exe: Path,
    input_file: Path,
    dest_dir: Path,
    *,
    connections: int = MAX_CONNECTIONS,
    timeout_s: int = 60,
) -> list[str]:
    """Build aria2's argv. The token lives in ``input_file``, not here."""
    return [
        str(exe),
        "--input-file",
        str(input_file),
        "-d",
        str(dest_dir),
        f"-x{connections}",
        f"-s{connections}",
        "-k1M",
        f"--min-split-size={MIN_SPLIT_SIZE}",
        "-c",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--file-allocation=none",
        f"--max-tries={MAX_TRIES}",
        f"--retry-wait={RETRY_WAIT_S}",
        f"--connect-timeout={timeout_s}",
        "--console-log-level=warn",
        "--summary-interval=1",
        "--download-result=hide",
    ]


@contextmanager
def temp_input_file(
    path: Path, url: str, out_name: str, token: Optional[str] = None
) -> Iterator[Path]:
    """Write the ``--input-file`` entry and always remove it afterwards."""
    try:
        path.write_text(build_input_text(url, out_name, token), encoding="utf-8")
        yield path
    finally:
        path.unlink(missing_ok=True)


def _to_bytes(value: str, unit: str) -> int | None:
    scale = UNITS.get(unit.lower())
    if scale is None:
        return None
    try:
        return int(float(value) * scale)
    except ValueError:
        return None


def parse_progress(text: str) -> list[tuple[int, Optional[int]]]:
    """Parse aria2 readout lines into ``(done, total)`` byte counts.

    Splits on ``\\r`` and ``\\n`` (aria2 redraws progress with ``\\r``), ignores
    anything it cannot parse, and returns nothing for lines without a byte pair.
    """
    seen: list[tuple[int, Optional[int]]] = []
    for chunk in re.split(r"[\r\n]+", ANSI_RE.sub("", text)):
        match = READOUT_RE.search(chunk)
        if not match:
            continue
        done = _to_bytes(match.group("done"), match.group("du"))
        if done is None:
            continue
        total = _to_bytes(match.group("total"), match.group("tu"))
        seen.append((done, total if total else None))
    return seen


def run_download(
    url: str,
    part_path: Path,
    label: str,
    expected_size: int,
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
    *,
    exe: Optional[Path] = None,
    timeout_s: int = 60,
    connections: int = MAX_CONNECTIONS,
    print_cb: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    """Download ``url`` into ``part_path`` with aria2, reporting progress.

    ``part_path`` keeps its name across attempts so ``-c`` resumes; the matching
    ``<part_path>.aria2`` control file is what makes that work. Leaves both in
    place on failure — ``modelhub`` decides when to delete them.

    Progress is best effort: an unreadable readout never fails the download.
    """
    binary = exe if exe is not None else find_aria2c()
    if binary is None:
        raise Aria2Error("aria2c is not installed")

    if print_cb is None:
        from video_upscaler.modelhub import _print_progress as print_cb  # noqa: F811

    input_file = part_path.with_name(part_path.name + ".aria2-req")
    argv = build_command(binary, input_file, part_path.parent, connections=connections, timeout_s=timeout_s)

    done = 0
    total = expected_size
    try:
        with temp_input_file(input_file, url, part_path.name, hf_token()):
            proc = subprocess.Popen(  # noqa: S106 - no shell, argv only
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            assert proc.stdout is not None
            for chunk in proc.stdout:
                for parsed_done, parsed_total in parse_progress(chunk):
                    if parsed_total:
                        total = parsed_total
                    if parsed_done >= done:
                        done = parsed_done
                        print_cb(label, done, total)
                        if progress_cb is not None:
                            progress_cb(done, total)
            code = proc.wait()
    except OSError as exc:
        raise Aria2Error(f"aria2c could not start: {exc}") from exc

    if code != 0:
        raise Aria2Error(f"aria2c exited with code {code}")
    if progress_cb is not None:
        progress_cb(total, total)
