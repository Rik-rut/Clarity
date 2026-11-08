"""Shared test setup: keep unit tests offline and free of a real aria2c.

``modelhub`` tries aria2 whenever the binary is on PATH, which would make
ordinary unit tests start real downloads. Tests that want the real binary (the
slow simulated-install benchmark) set ``CLARITY_TEST_REAL_ARIA2=1``.
"""

from __future__ import annotations

import os

import pytest

from video_upscaler import aria2


@pytest.fixture(autouse=True)
def _no_real_aria2(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("CLARITY_TEST_REAL_ARIA2") == "1":
        return
    monkeypatch.setattr(aria2, "find_aria2c", lambda: None)
