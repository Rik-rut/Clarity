"""Contract tests for the web UI, the desktop setup wizard and the Tauri shell.

These assert the *shape* of the desktop architecture: the studio page is a plain
web app, the first-run wizard is served over HTTP by the Python provisioning
server, and the Tauri window boots a dedicated shell that owns start / error /
retry.
"""

from pathlib import Path

import json
from fastapi.testclient import TestClient

from video_upscaler.web.server import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "src" / "video_upscaler" / "web" / "static"
TAURI_DIR = REPO_ROOT / "src-tauri"


def _read(path: Path) -> str:
    assert path.is_file(), f"expected file at {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------- studio page

def test_studio_page_has_no_desktop_boot_splash():
    """Boot status belongs to the shell.

    The splash used to live here and polled /api/system/info forever with no
    timeout, error state, or retry — so any backend problem left the user
    staring at a spinner that covered the app.
    """
    content = _read(STATIC_DIR / "index.html")
    assert "desktop-startup-splash" not in content
    assert "api/system/info" not in content
    assert 'id="drag-overlay"' in content


def test_app_js_contains_desktop_notification_bridge():
    content = _read(STATIC_DIR / "js" / "app.js")
    assert "window.__TAURI__" in content
    assert "initDesktopBridge" in content
    assert "sendDesktopNotification" in content
    assert "plugin:notification|notify" in content
    assert "Clarity — Render Complete" in content


def test_app_js_drag_overlay_cannot_stick_over_the_ui():
    """#drag-overlay covers the viewport with pointer-events: all while active."""
    content = _read(STATIC_DIR / "js" / "app.js")
    assert "function hideDragOverlay()" in content
    assert "dragWatchdog" in content
    # A drag released over a native dialog never delivers drop/dragleave.
    assert "addEventListener('blur', hideDragOverlay)" in content
    assert "addEventListener('dragend', hideDragOverlay)" in content


def test_app_js_prefers_the_native_folder_dialog_in_desktop_mode():
    content = _read(STATIC_DIR / "js" / "app.js")
    assert "'plugin:dialog|open', {" in content
    # The tkinter-backed route remains the browser-mode fallback.
    assert "/api/directories/browse" in content


# ------------------------------------------------------------------- wizard

def test_setup_html_exists_and_contains_core_elements():
    content = _read(STATIC_DIR / "setup.html")

    # Brand & theme
    assert "CLARITY" in content
    assert "DESKTOP SETUP" in content or "First-Time Setup" in content
    assert "clarity.jpg" in content

    # Stepper elements
    for step in ("step-python", "step-venv", "step-dependencies", "step-models"):
        assert f'id="{step}"' in content

    # Progress & metrics
    for element in (
        "setup-progress-bar",
        "setup-percent-text",
        "setup-stage-title",
        "setup-status-message",
        "setup-speed-text",
        "setup-eta-text",
    ):
        assert f'id="{element}"' in content

    # Error & retry, success, log drawer
    for element in (
        "setup-error-banner",
        "btn-retry-setup",
        "setup-error-message",
        "setup-success-banner",
        "logs-accordion",
        "btn-toggle-logs",
        "logs-terminal",
    ):
        assert f'id="{element}"' in content

    # Absolute asset path: the wizard is served by the backend, like the studio.
    assert 'src="/static/js/setup.js"' in content


def test_setup_js_drives_the_http_wizard():
    content = _read(STATIC_DIR / "js" / "setup.js")
    for endpoint in (
        "/api/setup/status",
        "/api/setup/progress",
        "/api/setup/detect-gpu",
        "/api/setup/runtime",
        "/api/setup/models",
        "/api/setup/verify",
        "/api/setup/complete",
        "/api/setup/retry",
    ):
        assert endpoint in content, f"setup.js must call {endpoint}"
    assert "window.__ClaritySetup" in content
    # Provisioning state must not travel over Tauri IPC. The wizard used to
    # listen for setup-progress / setup-complete / setup-error events on a page
    # served by another origin, where the event bridge is not connected, and the
    # progress bar then sat at 0% forever.
    assert "__TAURI__.event" not in content
    assert "start_setup" not in content
    assert "'setup-complete'" not in content


# -------------------------------------------------------------------- shell

def test_desktop_shell_is_the_tauri_frontend():
    config = json.loads(_read(TAURI_DIR / "tauri.conf.json"))
    assert config["build"]["frontendDist"] == "./shell"

    html = _read(TAURI_DIR / "shell" / "index.html")
    js = _read(TAURI_DIR / "shell" / "shell.js")

    for element in ("id=\"status\"", 'id="error-message"', 'id="btn-retry"', 'src="shell.js"'):
        assert element in html

    assert "invoke('retry_boot')" in js
    assert "'boot-status'" in js
    assert "'boot-error'" in js
    # Retry must still work when IPC is unavailable, and a failure must be
    # displayable without any event bridge at all.
    assert "window.location.reload()" in js
    assert "params.get('error')" in js
    # The shell is inert: it talks to no server, so it can never hang on a
    # backend that is not up yet.
    assert "fetch(" not in html + js
    assert "/api/" not in html + js


def test_desktop_shell_renders_provisioning_progress():
    html = _read(TAURI_DIR / "shell" / "index.html")
    js = _read(TAURI_DIR / "shell" / "shell.js")

    for element in ('id="progress"', 'id="progress-bar"', 'id="progress-percent"',
                    'id="progress-phase"', 'id="progress-log"'):
        assert element in html, f"shell must render {element}"

    assert "'boot-progress'" in js, "the shell must listen for progress events"
    assert "showProgress" in js
    # The shell still owns no product logic: no API calls, no fetch.
    assert "/api/" not in js
    assert "fetch(" not in js


def test_window_uses_html5_drag_and_drop():
    """Tauri's native drag-drop handler swallows HTML5 drag events on WebView2."""
    config = json.loads(_read(TAURI_DIR / "tauri.conf.json"))
    assert config["app"]["windows"][0]["dragDropEnabled"] is False


def test_capabilities_grant_loopback_pages_only_what_the_ui_calls():
    caps = {p.name: json.loads(_read(p)) for p in sorted((TAURI_DIR / "capabilities").glob("*.json"))}

    shell = caps["shell.json"]
    assert shell["windows"] == ["main"]
    assert shell["local"] is True
    assert "core:default" in shell["permissions"]
    # Local-only: the shell must not be reachable from the backend origin.
    assert "remote" not in shell

    backend = caps["local-backend.json"]
    # Without local:false plus these remote URL patterns, window.__TAURI__ is
    # simply undefined on the navigated page and every native call throws.
    assert backend["local"] is False
    assert backend["remote"]["urls"] == ["http://127.0.0.1:*", "http://localhost:*"]
    for permission in ("core:event:allow-listen", "notification:default", "dialog:allow-open"):
        assert permission in backend["permissions"]
    # A page served by the local backend must not gain filesystem or shell access.
    for forbidden in ("fs:", "shell:", "opener:", "process:", "window:"):
        assert not any(forbidden in str(p) for p in backend["permissions"])


# ------------------------------------------------------------------- served

def test_server_serves_studio_and_wizard_assets():
    client = TestClient(create_app())

    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "desktop-startup-splash" not in response.text

    response = client.get("/static/setup.html")
    assert response.status_code == 200
    assert "setup-progress-bar" in response.text

    response_js = client.get("/static/js/setup.js")
    assert response_js.status_code == 200
    assert "/api/setup/status" in response_js.text

    response_app = client.get("/static/js/app.js")
    assert response_app.status_code == 200
    assert "sendDesktopNotification" in response_app.text


def test_health_endpoint_is_cheap_and_ready_for_tauri():
    """Tauri must not pay for device detection on every readiness poll."""
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
