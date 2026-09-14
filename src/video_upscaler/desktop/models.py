"""Model weights provisioning.

Delegates to the app's own ``--download-models`` path so the manifest, sha256
verification and atomic writes in :mod:`video_upscaler.modelhub` stay the only
implementation. That code needs its dependencies, so this runs with the venv
interpreter created by the runtime step rather than the bootstrap one.
"""

from __future__ import annotations

from typing import Tuple

from video_upscaler.desktop.context import SetupContext
from video_upscaler.desktop.progress import ProgressTracker
from video_upscaler.desktop.runtime import run_streamed

TIERS: Tuple[str, ...] = ("essential", "all")
DEFAULT_TIER = "essential"


def install_models(
    context: SetupContext,
    tracker: ProgressTracker,
    tier: str = DEFAULT_TIER,
) -> None:
    """Download the requested weight tier into ``<data>/models``."""
    if tier not in TIERS:
        raise ValueError(f"Invalid model tier {tier!r}. Choose from: {', '.join(TIERS)}.")
    python = context.venv_python
    if not python.is_file():
        raise FileNotFoundError(f"Provisioned interpreter is missing: {python}")

    entry = context.main_py if context.main_py.is_file() else None
    command = (
        [str(python), str(entry), "--download-models", tier]
        if entry
        else [str(python), "-m", "video_upscaler.cli", "--download-models", tier]
    )
    tracker.line(f"Downloading model weights ({tier})…")
    run_streamed(
        command,
        tracker,
        context=context,
        cwd=str(context.app_dir),
        stage="model download",
    )
