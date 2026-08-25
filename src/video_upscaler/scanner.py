"""Input directory scanning and supported-video filtering."""

from __future__ import annotations

from pathlib import Path

from video_upscaler.config import SUPPORTED_EXTENSIONS


def scan_videos(input_dir: Path) -> list[Path]:
    """Return supported video files in ``input_dir``, sorted alphabetically.

    - Returns ``[]`` when the directory does not exist.
    - Filters to SUPPORTED_EXTENSIONS (case-insensitive suffix check).
    - Skips hidden files (name starts with ".") and directories.
    - Sorts case-insensitively for consistent menu ordering.
    """
    if not input_dir.is_dir():
        return []

    videos = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(videos, key=lambda path: path.name.lower())
