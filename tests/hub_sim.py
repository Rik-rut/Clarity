"""Test helper: a throttled local model hub plus a scaled-down manifest.

The hub reproduces the two properties that make the real download slow:

- ``/CLARITY_MODELS/<path>`` answers ``302`` and the target serves the bytes,
  like Hugging Face's ``resolve/`` -> CDN behaviour.
- bandwidth is capped **per connection**, which is the real bottleneck: one
  stream cannot fill a link, while parallel streams multiply the throughput.

Sizes and sha256 are real, so the manifest verification path is exercised.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Tuple

CHUNK = 64 * 1024
RANGE_RE = re.compile(r"bytes=(\d+)-(\d+)?")


class _HubHandler(BaseHTTPRequestHandler):
    hub_dir: Path
    cap_bps: float
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # keep test output clean
        return

    @property
    def served(self) -> list[str]:
        return self.server.served  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        self._handle(body=True)

    def do_HEAD(self) -> None:
        self._handle(body=False)

    def _handle(self, body: bool) -> None:
        path = self.path.lstrip("/")
        if path.startswith("cdn/"):
            self._serve(path[len("cdn/") :], body)
            return
        if path.startswith("CLARITY_MODELS/"):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_address[1]}/cdn/{path[len('CLARITY_MODELS/') :]}")  # type: ignore[attr-defined]
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_error(404)

    def _serve(self, rel: str, body: bool) -> None:
        file_path = self.hub_dir / rel
        if not file_path.is_file():
            self.send_error(404)
            return
        size = file_path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = RANGE_RE.match(range_header)
            if not match:
                self.send_error(400)
                return
            start = int(match.group(1))
            if match.group(2):
                end = min(int(match.group(2)), size - 1)
            if start > end:
                self.send_error(416)
                return
        self.served.append(rel)
        self.send_response(200 if start == 0 and end == size - 1 else 206)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if start or end != size - 1:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not body:
            return

        t0 = time.perf_counter()
        served_bytes = 0
        with open(file_path, "rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(CHUNK, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)
                served_bytes += len(chunk)
                # Pace this connection so its rate stays near ``cap_bps``.
                budget = served_bytes / self.cap_bps
                elapsed = time.perf_counter() - t0
                if elapsed < budget:
                    time.sleep(budget - elapsed)


class ThrottledHub:
    """Local hub server serving ``hub_dir`` with a per-connection rate cap."""

    def __init__(self, hub_dir: Path, cap_bps: float) -> None:
        self.hub_dir = hub_dir
        self.cap_bps = cap_bps
        self.served: list[str] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        handler = type(
            "HubHandler", (_HubHandler,), {"hub_dir": self.hub_dir, "cap_bps": self.cap_bps}
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.served = self.served  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}/CLARITY_MODELS"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def content(seed: str, size: int) -> bytes:
    """Deterministic bytes of exactly ``size``, varying per ``seed``."""
    block = hashlib.sha256(seed.encode()).digest()
    return (block * (size // len(block) + 1))[:size]


def build_simulated_hub(
    root: Path,
    *,
    big_size: int,
    small_size: int,
    small_count: int,
    cap_bps: float,
) -> Tuple[Path, dict, ThrottledHub]:
    """Write a scaled-down essential tier + manifest into ``root``."""
    hub_dir = root / "hub" / "CLARITY_MODELS"
    entries: Iterable[Tuple[str, str, str, str, int]] = [
        ("sam/sam_vit_b_01ec64.pth", "sam", "sam_vit_b_01ec64.pth", "sam", big_size),
    ]
    labels = [
        "up2x-latest-no-denoise",
        "up2x-latest-conservative",
        "up2x-latest-denoise1x",
        "up2x-latest-denoise2x",
        "up2x-latest-denoise3x",
        "up3x-latest-no-denoise",
        "up3x-latest-conservative",
        "up3x-latest-denoise3x",
        "up4x-latest-no-denoise",
        "up4x-latest-conservative",
        "up4x-latest-denoise3x",
        "up4x-latest-denoise1x",
    ][:small_count]
    for label in labels:
        entries.append((f"cugan/{label}.pth", "cugan", f"{label}.pth", "cugan", small_size))

    files = []
    for rel, group, dest, seed, size in entries:
        data = content(seed, size)
        path = hub_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append(
            {
                "group": group,
                "path": rel,
                "dest": dest,
                "tier": "essential",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    manifest = {"version": 1, "repo": "Rikrut/clarity", "files": files}
    return hub_dir, manifest, ThrottledHub(hub_dir, cap_bps)


def fetch(url: str, dest: Path) -> int:
    """Fetch one url into ``dest`` with a single connection; return byte count."""
    request = urllib.request.Request(url, headers={"User-Agent": "clarity-test"})
    with urllib.request.urlopen(request, timeout=30) as response:
        total = 0
        with open(dest, "wb") as handle:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
        return total
