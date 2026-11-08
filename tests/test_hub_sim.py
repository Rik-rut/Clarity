"""Unit tests for the throttled hub helper (task 4.1) — fast, no aria2."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hub_sim import ThrottledHub, build_simulated_hub, fetch

CAP = 2 * 1024 * 1024


@pytest.fixture()
def hub(tmp_path: Path) -> tuple[ThrottledHub, str, Path]:
    _, manifest, server = build_simulated_hub(
        tmp_path, big_size=4 * 1024 * 1024, small_size=1 * 1024 * 1024,
        small_count=4, cap_bps=CAP,
    )
    base = server.start()
    yield server, base, tmp_path
    server.stop()


def test_single_connection_stays_near_the_cap(hub) -> None:
    server, base, tmp_path = hub
    url = f"{base}/cugan/up2x-latest-no-denoise.pth"
    t0 = time.perf_counter()
    total = fetch(url, tmp_path / "single.bin")
    elapsed = time.perf_counter() - t0
    rate = total / elapsed
    assert total == 1 * 1024 * 1024
    assert 0.6 * CAP <= rate <= 1.35 * CAP, f"rate {rate / 1024 / 1024:.2f} MB/s, cap {CAP / 1024 / 1024:.0f} MB/s"


def test_parallel_connections_scale(hub) -> None:
    server, base, tmp_path = hub
    names = [
        "up2x-latest-no-denoise",
        "up2x-latest-conservative",
        "up2x-latest-denoise1x",
        "up2x-latest-denoise2x",
    ]

    def one(name: str) -> int:
        return fetch(f"{base}/cugan/{name}.pth", tmp_path / f"{name}.bin")

    results: list[int] = []
    threads = [threading.Thread(target=lambda name=n: results.append(one(name))) for n in names]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    aggregate = sum(results) / elapsed
    assert sum(results) == 4 * 1024 * 1024
    assert len(results) == 4
    assert aggregate >= 3 * CAP, f"aggregate {aggregate / 1024 / 1024:.2f} MB/s, cap {CAP / 1024 / 1024:.0f} MB/s"
    assert server.served.count("cugan/up2x-latest-no-denoise.pth") == 1


def test_hub_supports_range_requests(hub) -> None:
    import urllib.request

    server, base, tmp_path = hub
    url = f"{base}/sam/sam_vit_b_01ec64.pth"
    request = urllib.request.Request(url, headers={"Range": "bytes=100-199"})
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 206
        assert response.headers.get("Accept-Ranges") == "bytes"
        assert response.headers.get("Content-Range") == "bytes 100-199/4194304"
        assert len(response.read()) == 100


def test_hub_redirects_like_the_real_one(hub) -> None:
    server, base, tmp_path = hub
    url = f"{base}/cugan/up2x-latest-no-denoise.pth"
    assert fetch(url, tmp_path / "r.bin") == 1 * 1024 * 1024
    # The redirect target is what served the bytes.
    assert server.served == ["cugan/up2x-latest-no-denoise.pth"]
