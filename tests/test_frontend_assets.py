"""Tests for Frontend Setup Wizard and Desktop IPC Bridge static assets."""

from pathlib import Path
from fastapi.testclient import TestClient
from video_upscaler.web.server import create_app


def test_setup_html_exists_and_contains_core_elements():
    static_dir = Path(__file__).resolve().parent.parent / "src" / "video_upscaler" / "web" / "static"
    setup_file = static_dir / "setup.html"
    assert setup_file.is_file(), f"setup.html must exist at {setup_file}"

    content = setup_file.read_text(encoding="utf-8")

    # Brand & Theme
    assert "CLARITY" in content
    assert "DESKTOP SETUP" in content or "First-Time Setup" in content
    assert "clarity.jpg" in content

    # Stepper elements
    assert 'id="step-python"' in content
    assert 'id="step-venv"' in content
    assert 'id="step-dependencies"' in content
    assert 'id="step-models"' in content

    # Progress & Metrics
    assert 'id="setup-progress-bar"' in content
    assert 'id="setup-percent-text"' in content
    assert 'id="setup-stage-title"' in content
    assert 'id="setup-status-message"' in content
    assert 'id="setup-speed-text"' in content
    assert 'id="setup-eta-text"' in content

    # Error & Retry
    assert 'id="setup-error-banner"' in content
    assert 'id="btn-retry-setup"' in content
    assert 'id="setup-error-message"' in content

    # Success Banner
    assert 'id="setup-success-banner"' in content

    # Technical Log Drawer
    assert 'id="logs-accordion"' in content
    assert 'id="btn-toggle-logs"' in content
    assert 'id="logs-terminal"' in content

    # Script tags
    assert 'src="js/setup.js"' in content


def test_setup_js_exists_and_contains_tauri_listeners():
    static_dir = Path(__file__).resolve().parent.parent / "src" / "video_upscaler" / "web" / "static"
    js_file = static_dir / "js" / "setup.js"
    assert js_file.is_file(), f"setup.js must exist at {js_file}"

    content = js_file.read_text(encoding="utf-8")
    assert "setup-progress" in content
    assert "setup-complete" in content
    assert "setup-error" in content
    assert "start_setup" in content
    assert "retry_setup" in content
    assert "window.__TAURI__" in content
    assert "window.__ClaritySetup" in content


def test_app_js_contains_desktop_notification_bridge():
    static_dir = Path(__file__).resolve().parent.parent / "src" / "video_upscaler" / "web" / "static"
    app_js = static_dir / "js" / "app.js"
    assert app_js.is_file()

    content = app_js.read_text(encoding="utf-8")
    assert "window.__TAURI__" in content
    assert "initDesktopBridge" in content
    assert "sendDesktopNotification" in content
    assert "sendNotification" in content
    assert "Clarity — Render Complete" in content


def test_server_serves_setup_wizard_assets():
    app = create_app()
    client = TestClient(app)

    response = client.get("/static/setup.html")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Clarity" in response.text
    assert "setup-progress-bar" in response.text

    response_js = client.get("/static/js/setup.js")
    assert response_js.status_code == 200
    assert "setup-progress" in response_js.text

    response_app = client.get("/static/js/app.js")
    assert response_app.status_code == 200
    assert "sendDesktopNotification" in response_app.text
