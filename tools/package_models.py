"""Compile the CLARITY_MODELS/ staging directory for the Hugging Face hub.

Collects every model file Clarity can download (Real-CUGAN weights, AMT
checkpoints, MultiPassDedup weights) from the local ``models/`` trees,
hashes them, copies them into a hub-layout staging folder, and writes the
app-facing manifest to both the staging root and
``src/video_upscaler/data/manifest.json``.

Usage:
    uv run tools/package_models.py                 # build + verify
    uv run tools/package_models.py --out DIR       # custom output folder
    uv run tools/package_models.py --verify        # re-hash an existing build

After running, upload the contents of the staging folder to the root of
https://huggingface.co/Rik-rut/clarity-models (keeping the folder layout).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from video_upscaler import config  # noqa: E402
from video_upscaler.modelhub import MANIFEST_PATH, sha256_file  # noqa: E402
from video_upscaler.models import INTERP_MODELS, PROFILES  # noqa: E402

CUGAN_FILES = sorted({model for model, _ in PROFILES.values()})
AMT_FILES = sorted(ckpt for ckpt, _ in INTERP_MODELS.values())
DEDUP_TOP_FILES = [
    "rife48.pkl",
]
GMFSS_FOLDER = "train_log_pg104"

ESSENTIAL_AMT = {"amt-s.pth"}


def _dedup_sources() -> list[tuple[Path, str]]:
    """(source path under DEDUP_MODELS_DIR, hub-relative path) pairs."""
    pairs: list[tuple[Path, str]] = []
    dedup_dir = config.DEDUP_MODELS_DIR
    for name in DEDUP_TOP_FILES:
        pairs.append((dedup_dir / name, f"dedup/{name}"))
    gmfss_dir = dedup_dir / GMFSS_FOLDER
    if gmfss_dir.is_dir():
        for source in sorted(gmfss_dir.rglob("*")):
            if source.is_file():
                rel = source.relative_to(dedup_dir).as_posix()
                pairs.append((source, f"dedup/{rel}"))
    return pairs


def collect_inventory() -> list[dict[str, object]]:
    """Full manifest inventory resolved from local model directories."""
    items: list[dict[str, object]] = []
    missing: list[str] = []

    for name in CUGAN_FILES:
        source = config.MODELS_DIR / name
        if not source.is_file():
            missing.append(str(source))
            continue
        items.append(
            {
                "group": "cugan",
                "path": f"cugan/{name}",
                "dest": name,
                "tier": "essential",
                "_source": source,
            }
        )

    for name in AMT_FILES:
        source = config.MODELS_DIR / name
        if not source.is_file():
            missing.append(str(source))
            continue
        items.append(
            {
                "group": "amt",
                "path": f"amt/{name}",
                "dest": name,
                "tier": "essential" if name in ESSENTIAL_AMT else "full",
                "_source": source,
            }
        )

    for source, rel in _dedup_sources():
        if not source.is_file():
            missing.append(str(source))
            continue
        items.append(
            {
                "group": "dedup",
                "path": rel,
                "dest": source.relative_to(config.DEDUP_MODELS_DIR).as_posix(),
                "tier": "full",
                "_source": source,
            }
        )

    if missing:
        raise SystemExit(
            "Missing model files expected under models/:\n  "
            + "\n  ".join(missing)
            + "\nInstall them first (run the app and answer [Y] to download),\n"
            "then re-run the packager."
        )
    return items


def build(out_dir: Path) -> None:
    items = collect_inventory()
    print(f"Packing {len(items)} files from {config.MODELS_DIR} ...")

    staged: list[dict[str, object]] = []
    total = 0
    for entry in items:
        source = Path(entry["_source"])  # type: ignore[arg-type]
        target = out_dir / str(entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        size = target.stat().st_size
        digest = sha256_file(target)
        staged.append(
            {
                "group": entry["group"],
                "path": entry["path"],
                "dest": entry["dest"],
                "tier": entry["tier"],
                "size": size,
                "sha256": digest,
            }
        )
        total += size
        print(f"  {entry['path']}  ({size / (1 << 20):.1f} MB)")

    manifest = {
        "version": 1,
        "repo": "Rikrut/clarity",
        "files": staged,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    staged_manifest = out_dir / "manifest.json"
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"Wrote {staged_manifest}")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Total: {len(staged)} files, {total / (1 << 20):.1f} MB")


def verify(out_dir: Path) -> int:
    if not out_dir.is_dir():
        raise SystemExit(f"Staging directory does not exist: {out_dir}")
    failures = 0
    for entry in collect_inventory():
        target = out_dir / str(entry["path"])
        source = Path(entry["_source"])  # type: ignore[arg-type]
        if not target.is_file():
            print(f"MISSING  {target}")
            failures += 1
        elif sha256_file(target) != sha256_file(source):
            print(f"MISMATCH {target}")
            failures += 1
        else:
            print(f"OK       {entry['path']}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=BASE_DIR / "CLARITY_MODELS",
        help="staging directory to fill (default: ./CLARITY_MODELS)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="only verify an existing staging directory against models/",
    )
    args = parser.parse_args()

    if args.verify:
        failures = verify(args.out)
        if failures:
            raise SystemExit(f"{failures} file(s) failed verification.")
        print("All staged files verified.")
        return 0

    build(args.out)
    failures = verify(args.out)
    if failures:
        raise SystemExit(f"{failures} file(s) failed post-build verification.")
    print("Post-build verification passed.")
    print(
        "\nNext step: upload the CONTENTS of "
        f"{args.out} to https://huggingface.co/Rikrut/clarity"
        "\n(keep the folder layout; manifest.json goes to the repo root)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
