"""The web server must start without importing torch.

The Tauri shell polls /api/health before opening the studio window, and the
first response is only possible once the module import chain finishes. torch's
cold import costs ~10s, so any module-level torch import in that chain turns
into a visible "Starting…" delay on every launch.
"""

from __future__ import annotations

import subprocess
import sys

_IMPORT_CODE = (
    "import sys; import video_upscaler.web.server as s; "
    "assert 'torch' not in sys.modules, 'torch was imported at server import time'; "
    "print('OK')"
)


def test_server_import_chain_is_torch_free() -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-c", _IMPORT_CODE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
