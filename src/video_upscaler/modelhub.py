"""Central model hub: manifest-driven downloads from one canonical source.

Every downloadable artifact (Real-CUGAN weights, AMT checkpoints,
MultiPassDedup weights) is described in ``data/manifest.json`` with its
hub-relative path, destination, byte size, and sha256. Files are fetched
from a single Hugging Face repo (``Rikrut/clarity``, ``CLARITY_MODELS``
folder) so end users never chase scattered release pages.

For an https source the backends run in order: ``aria2c`` (multi-connection,
resumable), then ``huggingface_hub`` (Xet), then plain ``urllib``. A file is
installed only after its size and sha256 match the manifest, whichever backend
fetched it. ``CLARITY_DOWNLOADER`` forces a single backend for troubleshooting.

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
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse

from video_upscaler import aria2, config

DEFAULT_HUB_REPO = "Rikrut/clarity"
DEFAULT_HUB_BASE = (
    f"https://huggingface.co/{DEFAULT_HUB_REPO}/resolve/main/CLARITY_MODELS"
)

MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "manifest.json"

DOWNLOAD_TIMEOUT_S = 60
_CHUNK = 1 << 20  # 1 MiB

# Xet (huggingface_hub's accelerated transport) parallelises chunks and pays
# off on the big weights. Small files are faster over one plain connection.
HF_MIN_BYTES = 16 << 20

# Download backends for an https source, in the order ``auto`` tries them.
# aria2 leads: the spike measured 23.3 MB/s against Xet's 14.0 MB/s and the
# plain single connection's 1.7 MB/s (spike-results.md).
DOWNLOADERS = ("auto", "aria2", "hf", "http")
PART_SUFFIX = ".part"
ARIA2_CONTROL_SUFFIX = ".aria2"
ARIA2_REQUEST_SUFFIX = ".aria2-req"

_noticed: set[str] = set()


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


def _entry_base(entry: dict[str, Any]) -> str:
    """Hub base for one entry — optional per-entry 'repo' override.

    Entries may set "repo": "<owner>/<name>" to fetch from a different
    Hugging Face repo (e.g. a separate matting-models repo) while the rest
    of the manifest keeps using the default hub.
    """
    repo = entry.get("repo")
    if repo:
        return f"https://huggingface.co/{repo}/resolve/main"
    return hub_base()


def _entry_source_url(base: str, rel_path: str) -> str:
    return f"{base.rstrip('/')}/{rel_path}"


def _format_mb(size_bytes: float) -> str:
    return f"{size_bytes / (1 << 20):.1f} MB"


def _verify_hash(path: Path, expected_sha256: str) -> bool:
    return sha256_file(path) == expected_sha256


def _hf_downloader():
    """Import ``hf_hub_download``, keeping every HF cache inside the app dir."""
    os.environ.setdefault("HF_HOME", str(config.CACHE_DIR / "huggingface"))
    from huggingface_hub import hf_hub_download

    return hf_hub_download


def _hf_repo_and_filename(entry: dict[str, Any]) -> tuple[str, str]:
    """Map a manifest entry onto ``hf_hub_download``'s repo/filename pair.

    The default hub base already includes the ``CLARITY_MODELS`` folder, so the
    repo-relative filename needs that prefix. Entries that point at their own
    repo carry a bare repo-relative path.
    """
    repo = entry.get("repo")
    if repo:
        return str(repo), str(entry["path"])
    manifest = load_manifest()
    repo_id = str(manifest.get("repo") or DEFAULT_HUB_REPO)
    path = str(entry["path"])
    prefix = "CLARITY_MODELS/"
    if path.startswith(prefix):
        path = path[len(prefix) :]
    return repo_id, f"{prefix}{path}"


def _should_use_hf(entry: dict[str, Any]) -> bool:
    """True when the Xet path applies: default hub, big file, hub importable."""
    if hub_base() != DEFAULT_HUB_BASE and not entry.get("repo"):
        return False  # custom mirror or local dir keeps the plain HTTP path
    if int(entry["size"]) < HF_MIN_BYTES:
        return False
    try:
        _hf_downloader()
    except Exception:
        return False
    return True


def _notice(message: str) -> None:
    """Print a one-off notice; backends are resolved per file, so don't repeat."""
    if message not in _noticed:
        _noticed.add(message)
        print(message)


def downloader_choice() -> str:
    """The backend forced by ``CLARITY_DOWNLOADER``, or ``auto``.

    An unrecognised value is ignored with a warning.
    """
    value = os.environ.get("CLARITY_DOWNLOADER", "").strip().lower()
    if not value:
        return "auto"
    if value not in DOWNLOADERS:
        _notice(
            f"Unrecognised CLARITY_DOWNLOADER={value!r}; expected "
            f"{'|'.join(DOWNLOADERS)}. Using the automatic order."
        )
        return "auto"
    return value


def download_backends(entry: dict[str, Any]) -> list[str]:
    """Ordered backends to try for one entry.

    ``auto`` is aria2 (when installed) -> Xet (when it applies) -> plain HTTP.
    A forced backend that is not available prints a notice and degrades to
    ``auto``. A local hub never reaches this: it copies instead.
    """
    auto: list[str] = []
    if aria2.find_aria2c() is not None:
        auto.append("aria2")
    if _should_use_hf(entry):
        auto.append("hf")
    auto.append("http")

    choice = downloader_choice()
    if choice == "auto":
        return auto
    if choice == "aria2":
        if "aria2" in auto:
            return ["aria2", *auto[1:]]
        _notice(
            "CLARITY_DOWNLOADER=aria2 but aria2c is not installed; "
            "using the automatic order."
        )
        return auto
    if choice == "hf":
        if "hf" in auto:
            return ["hf", "http"]
        _notice(
            "CLARITY_DOWNLOADER=hf but the Hugging Face path is unavailable for "
            "this file; using the automatic order."
        )
        return auto
    return ["http"]


def _aria2_side_files(temp_path: Path) -> tuple[Path, Path]:
    """aria2's control file (resume) and its temporary request file."""
    return (
        temp_path.with_name(temp_path.name + ARIA2_CONTROL_SUFFIX),
        temp_path.with_name(temp_path.name + ARIA2_REQUEST_SUFFIX),
    )


def _progress_tqdm(progress_cb: Optional[Callable[[int, Optional[int]], None]]):
    """Build a tqdm subclass that forwards chunk progress to ``progress_cb``."""
    from tqdm.auto import tqdm

    class _ClarityTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True  # the caller owns the visible progress line
            super().__init__(*args, **kwargs)
            self._clarity_seen = int(kwargs.get("initial") or 0)

        def update(self, n=1):
            self._clarity_seen += n
            if progress_cb is not None:
                total = int(self.total) if self.total else None
                progress_cb(self._clarity_seen, total)
            return True

    return _ClarityTqdm


def _download_hf(
    entry: dict[str, Any],
    staging_dir: Path,
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
) -> Path:
    """Download one entry through huggingface_hub (Xet-accelerated)."""
    hf_hub_download = _hf_downloader()
    repo_id, filename = _hf_repo_and_filename(entry)
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(staging_dir),
        cache_dir=str(config.CACHE_DIR / "huggingface"),
        tqdm_class=_progress_tqdm(progress_cb),
    )
    return Path(staged)


def _try_download_hf(
    entry: dict[str, Any],
    staging_dir: Path,
    progress_cb: Optional[Callable[[int, Optional[int]], None]],
) -> Optional[Path]:
    """Return the staged file, or None so the caller falls back to plain HTTP."""
    try:
        return _download_hf(entry, staging_dir, progress_cb)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to HTTP
        print(f"Accelerated download unavailable ({exc}); using direct download…")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return None


def _download_http(
    url: str,
    dest: Path,
    label: str,
    expected_size: int,
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
) -> None:
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
                if progress_cb is not None:
                    progress_cb(done, total)
    print()


def _print_progress(label: str, done: int, total: int) -> None:
    percent = min(100, int(done * 100 / total)) if total else 0
    message = f"{label}: {percent}% ({_format_mb(done)} / {_format_mb(total)})"
    print(f"\r{message:<70}", end="", flush=True)


def _stage_download(
    entry: dict[str, Any],
    temp_path: Path,
    staging_dir: Path,
    label: str,
    expected_size: int,
    progress_cb: Optional[Callable[[int, Optional[int]], None]],
    aria2_used: list[bool],
) -> None:
    """Fetch one entry into ``temp_path``, trying each backend in order.

    ``aria2_used`` is appended to as soon as aria2 runs, so the caller still
    knows a partial file may be worth resuming when this raises.

    Raises ``HubError`` only when every applicable backend failed.
    """
    url = _entry_source_url(_entry_base(entry), entry["path"])
    for backend in download_backends(entry):
        if backend == "aria2":
            aria2_used.append(True)
            try:
                aria2.run_download(url, temp_path, label, expected_size, progress_cb)
                return
            except aria2.Aria2Error as exc:
                print(f"Accelerated download unavailable ({exc}); trying another method…")
        elif backend == "hf":
            staged = _try_download_hf(entry, staging_dir, progress_cb)
            if staged is not None:
                os.replace(staged, temp_path)
                return
        else:
            try:
                _download_http(url, temp_path, label, expected_size, progress_cb)
            except (urllib.error.URLError, OSError) as exc:
                raise HubError(
                    f"Failed to download {entry['path']}:\n{exc}\n\n"
                    f"Check your network connection, or set CLARITY_MODEL_HUB_BASE\n"
                    f"to a mirror/local folder containing the models."
                ) from exc
            return
    raise HubError(
        f"Failed to download {entry['path']}: no download method succeeded.\n"
        "Set CLARITY_MODEL_HUB_BASE to a mirror/local folder containing the models, "
        "or retry."
    )


def install_entry(
    entry: dict[str, Any],
    quiet_existing: bool = False,
    *,
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
) -> Path:
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
    temp_path = dest.with_name(dest.name + PART_SUFFIX)
    staging_dir = dest.with_name(dest.name + ".staging")

    verified = False
    verification_failed = False
    aria2_used: list[bool] = []
    try:
        if local_dir is not None:
            source = local_dir / PurePosixPath(entry["path"])
            if not source.is_file():
                raise HubError(f"Missing in local hub ({local_dir}): {source}")
            print(f"Copying {source.name} from local hub...")
            shutil.copyfile(source, temp_path)
            if progress_cb is not None:
                total = int(entry["size"])
                progress_cb(total, total)
        else:
            _stage_download(
                entry, temp_path, staging_dir, label, int(entry["size"]), progress_cb, aria2_used
            )

        actual_size = temp_path.stat().st_size
        if actual_size != int(entry["size"]):
            verification_failed = True
            raise HubError(
                f"Downloaded {dest.name} has wrong size "
                f"({_format_mb(actual_size)} != {_format_mb(int(entry['size']))})."
            )
        if not _verify_hash(temp_path, entry["sha256"]):
            verification_failed = True
            raise HubError(
                f"Downloaded {dest.name} failed its integrity check (sha256).\n"
                "Please retry; if it keeps failing the hub copy may be corrupt."
            )
        verified = True
        os.replace(temp_path, dest)
    finally:
        # aria2 keeps its partial file + control file so a retry resumes; a
        # corrupt or unused partial is removed. hf/http cleanup is unchanged.
        control, request = _aria2_side_files(temp_path)
        if verified or verification_failed or not aria2_used:
            temp_path.unlink(missing_ok=True)
            control.unlink(missing_ok=True)
        request.unlink(missing_ok=True)
        shutil.rmtree(staging_dir, ignore_errors=True)

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
