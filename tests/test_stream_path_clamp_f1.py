"""Follow-up F1: streamed video paths must clamp to the media roots for ALL forms.

- Absolute path outside the roots (existing file) -> 403 (never serve).
- Drive-less rooted escape (``/../...`` on Windows) -> 403.
- Drive-less rooted in-root missing file -> 404 (historical behavior kept).
- Absolute path in-root (existing file) -> 200 (output-click/scan flows).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from video_upscaler.web.server import create_app

PAYLOAD = b"f1-clamp-bytes" * 100


def _patch_roots(monkeypatch, tmp_path):
    from video_upscaler import config

    data_dir = tmp_path / "data"
    out_dir = data_dir / "output"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    return data_dir, out_dir


def test_absolute_path_outside_roots_is_forbidden(tmp_path, monkeypatch):
    data_dir, _ = _patch_roots(monkeypatch, tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(PAYLOAD)
    assert data_dir.resolve() not in outside.resolve().parents

    client = TestClient(create_app())
    resp = client.get("/api/stream/video", params={"path": str(outside.resolve())})

    assert resp.status_code == 403


def test_rooted_escape_is_forbidden(tmp_path, monkeypatch):
    data_dir, _ = _patch_roots(monkeypatch, tmp_path)
    secret = data_dir.parent / "secret.mp4"
    secret.write_bytes(PAYLOAD)

    client = TestClient(create_app())
    resp = client.get("/api/stream/video", params={"path": "/../secret.mp4"})

    assert resp.status_code == 403


def test_rooted_in_root_missing_is_not_found(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)

    client = TestClient(create_app())
    resp = client.get("/api/stream/video", params={"path": "/output/missing.mp4"})

    assert resp.status_code == 404


def test_absolute_path_in_root_streams(tmp_path, monkeypatch):
    _, out_dir = _patch_roots(monkeypatch, tmp_path)
    clip = out_dir / "render_result.mp4"
    clip.write_bytes(PAYLOAD)

    client = TestClient(create_app())
    resp = client.get("/api/stream/video", params={"path": str(clip.resolve())})

    assert resp.status_code == 200
    assert int(resp.headers["content-length"]) == len(PAYLOAD)
