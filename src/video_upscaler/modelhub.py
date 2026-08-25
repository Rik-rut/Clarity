"""Central model hub: manifest-driven downloads from one canonical source.

Every downloadable artifact (Real-CUGAN weights, AMT checkpoints,
MultiPassDedup weights) is described in ``data/manifest.json`` with its
hub-relative path, destination, byte size, and sha256. Files are fetched
from a single Hugging Face repo (``Rikrut/clarity``, ``CLARITY_MODELS``
folder) so end users never chase scattered release pages.

``CLARITY_MODEL_HUB_BASE`` overrides the source. It may be an https base
URL or a local directory (or ``file://`` URL), which enables offline
installs from the staged ``CLARITY_MODELS/`` folder.

This module stays torch-free and prompt-free so unit tests can exercise it
without heavy imports; consent prompting lives in the CLI layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

DEFAULT_HUB_REPO = "Rikrut/clarity"
DEFAULT_HUB_BASE = (
    f"https://huggingface.co/{DEFAULT_HUB_REPO}/resolve/main/CLARITY_MODELS"
)

MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "manifest.json"

DOWNLOAD_TIMEOUT_S = 60
_CHUNK = 1 << 20  # 1 MiB


class HubError(RuntimeError):
    """Hub metadata or download failure with an actionable message."""


def hub_base() -> str:
    """Return the configured hub source (https base, local dir, or file:// URL)."""
    value = os.environ.get("CLARITY_MODEL_HUB_BASE", "").strip()
    return value if value else DEFAULT_HUB_BASE


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    """Load and validate data/manifest.json."""
    if not MANIFEST_PATH.is_file():
        raise HubError(
            f"Model manifest not found:\n{MANIFEST_PATH}\n\n"
            "Reinstall Clarity or regenerate it with:\n"
            "uv run tools/package_models.py"
        )
    try:
        # utf-8-sig tolerates a BOM (e.g. manifests edited in Notepad).
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HubError(f"Model manifest unreadable: {exc}") from exc
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise HubError("Model manifest has no 'files' list.")
    for entry in files:
        for key in ("group", "path", "dest", "size", "sha256"):
            if key not in entry:
                raise HubError(f"Manifest entry missing {key!r}: {entry}")
    return manifest


def _group_root(group: str) -> Path:
    """Destination root directory for a manifest group (env-overridable)."""
    from video_upscaler import config

    if group in ("cugan", "amt"):
        return config.MODELS_DIR
    if group == "dedup":
        return config.DEDUP_MODELS_DIR
    if group in ("matanyone", "sam"):
        return config.MODELS_DIR / group
    raise HubError(f"Unknown manifest group: {group}")


def entries(
    manifest: dict[str, Any] | None = None,
    group: str | None = None,
    tier: str | None = None,
) -> list[dict[str, Any]]:
    """Filtered manifest entries (group and/or tier: 'essential' | 'full')."""
    manifest = manifest if manifest is not None else load_manifest()
    selected = []
    for entry in manifest["files"]:
        if group is not None and entry["group"] != group:
            continue
        entry_tier = entry.get("tier", "full")
        if tier == "essential" and entry_tier != "essential":
            continue
        selected.append(entry)
    return selected


def total_bytes(items: list[dict[str, Any]]) -> int:
    """Sum of entry sizes (for progress messaging)."""
    return sum(int(entry["size"]) for entry in items)


def _local_base_dir(base: str) -> Path | None:
    """Return base as a filesystem dir when CLARITY_MODEL_HUB_BASE is local."""
    if base.startswith("file://"):
        parsed = urlparse(base)
        return Path(unquote(parsed.path))
    candidate = Path(base)
    if candidate.is_dir():
        return candidate
    return None


def _entry_source_url(base: str, rel_path: str) -> str:
    return f"{base.rstrip('/')}/{rel_path}"


def _format_mb(size_bytes: float) -> str:
    return f"{size_bytes / (1 << 20):.1f} MB"


def _verify_hash(path: Path, expected_sha256: str) -> bool:
    return sha256_file(path) == expected_sha256


def _download_http(url: str, dest: Path, label: str, expected_size: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "clarity-upscaler"})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else expected_size
        done = 0
        with open(dest, "wb") as handle:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                _print_progress(label, done, total)
    print()


def _print_progress(label: str, done: int, total: int) -> None:
    percent = min(100, int(done * 100 / total)) if total else 0
    message = f"{label}: {percent}% ({_format_mb(done)} / {_format_mb(total)})"
    print(f"\r{message:<70}", end="", flush=True)


def install_entry(entry: dict[str, Any], quiet_existing: bool = False) -> Path:
    """Install one manifest entry into its group root (idempotent).

    Fast path: an existing file with the expected size is kept as-is.
    Otherwise the file is downloaded/copied to a temp name, hash-verified,
    and atomically renamed into place.
    """
    dest_root = _group_root(entry["group"])
    dest = dest_root / PurePosixPath(entry["dest"])
    label = f"Downloading {PurePosixPath(entry['path']).name}"

    if dest.is_file() and dest.stat().st_size == int(entry["size"]):
        if not quiet_existing:
            print(f"Already installed: {dest.name}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    base = hub_base()
    local_dir = _local_base_dir(base)
    temp_path = dest.with_name(dest.name + ".part")

    try:
        if local_dir is not None:
            source = local_dir / PurePosixPath(entry["path"])
            if not source.is_file():
                raise HubError(f"Missing in local hub ({local_dir}): {source}")
            print(f"Copying {source.name} from local hub...")
            shutil.copyfile(source, temp_path)
        else:
            url = _entry_source_url(base, entry["path"])
            try:
                _download_http(url, temp_path, label, int(entry["size"]))
            except (urllib.error.URLError, OSError) as exc:
                raise HubError(
                    f"Failed to download {entry['path']}:\n{exc}\n\n"
                    f"Check your network connection, or set CLARITY_MODEL_HUB_BASE\n"
                    f"to a mirror/local folder containing the models."
                ) from exc

        actual_size = temp_path.stat().st_size
        if actual_size != int(entry["size"]):
            temp_path.unlink(missing_ok=True)
            raise HubError(
                f"Downloaded {dest.name} has wrong size "
                f"({_format_mb(actual_size)} != {_format_mb(int(entry['size']))})."
            )
        if not _verify_hash(temp_path, entry["sha256"]):
            temp_path.unlink(missing_ok=True)
            raise HubError(
                f"Downloaded {dest.name} failed its integrity check (sha256).\n"
                "Please retry; if it keeps failing the hub copy may be corrupt."
            )
        os.replace(temp_path, dest)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Installed: {dest}")
    return dest


def missing_entries(
    items: list[dict[str, Any]], verify_hashes: bool = False
) -> list[dict[str, Any]]:
    """Entries whose destination file is absent (or corrupt when hashing)."""
    missing = []
    for entry in items:
        dest = _group_root(entry["group"]) / PurePosixPath(entry["dest"])
        if not dest.is_file():
            missing.append(entry)
        elif verify_hashes and not _verify_hash(dest, entry["sha256"]):
            missing.append(entry)
    return missing


def install_tier(tier: str | None = None, group: str | None = None) -> int:
    """Install all entries at a tier ('essential' or None=all); returns count.

    Raises HubError summarizing any failures after attempting every file.
    """
    items = entries(group=group, tier=tier)
    if not items:
        return 0
    payload = total_bytes(items)
    scope = group or ("essential set" if tier == "essential" else "all models")
    print(f"Fetching {len(items)} file(s) ({_format_mb(payload)}) — {scope}")
    failures: list[str] = []
    for entry in items:
        try:
            install_entry(entry)
        except HubError as exc:
            failures.append(str(exc))
    if failures:
        raise HubError("\n\n".join(failures))
    return len(items)


def describe_local_base() -> str:
    """Human-readable hub source for startup output/tests."""
    base = hub_base()
    local_dir = _local_base_dir(base)
    return f"local: {local_dir}" if local_dir else base
