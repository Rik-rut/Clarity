"""Vendored MatAnyone2 stays isolated: lazily loaded, torch-free wrappers."""

import subprocess
import sys
from pathlib import Path

VENDOR_DIR = (
    Path(__file__).resolve().parents[1]
    / "src" / "video_upscaler" / "matanyone2" / "vendor"
)


def test_vendor_tree_present():
    core = VENDOR_DIR / "matanyone2" / "inference" / "inference_core.py"
    utils = VENDOR_DIR / "matanyone2" / "utils" / "get_default_model.py"
    license_file = VENDOR_DIR / "LICENSE.txt"
    assert core.is_file(), f"missing {core}"
    assert utils.is_file(), f"missing {utils}"
    assert license_file.is_file(), "upstream LICENSE.txt must ship with vendor"


def test_wrapper_package_import_is_torch_free():
    code = (
        "import sys; import video_upscaler.matanyone2; "
        "assert 'torch' not in sys.modules, 'torch leaked at package import'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
