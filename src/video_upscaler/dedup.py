"""Interactive prompts, plan building, and execution orchestration for MultiPassDedup."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import questionary

from video_upscaler import config
from video_upscaler.dedup_backend import (
    DEDUP_MODEL_NAMES,
    DEDUP_MODELS,
    check_dedup_weights,
    detect_dedup_device,
    parse_npass,
    validate_model_type,
)


def _interactive() -> bool:
    """Return True when stdin and stdout are interactive TTYs."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_choice(title: str, options: dict[str, str]) -> str:
    """Arrow-key single select; returns the selected option key."""
    if not _interactive():
        return next(iter(options))
    try:
        chosen = questionary.select(title, choices=list(options.values())).ask()
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(0)
    if chosen is None:
        raise SystemExit(0)
    return next(key for key, label in options.items() if label == chosen)


def _prompt_confirm(title: str) -> bool:
    """Yes/no confirmation; False when non-interactive."""
    if not _interactive():
        return False
    try:
        return bool(questionary.confirm(title, default=False).ask())
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(0)


def _cadence_options() -> dict[str, str]:
    return {
        "auto": "Auto-detect (recommended) — automatic cadence detection",
        "2": "On Twos — animation on 2s (e.g. 12 fps on 24 fps)",
        "3": "On Threes — animation on 3s (e.g. 8 fps on 24 fps)",
    }


def _model_options() -> dict[str, str]:
    return {
        "gmfss": "GMFSS (Fortuna) | Best quality (default)",
        "rife": "RIFE (Practical-RIFE) | Faster",
    }


def _factor_options() -> dict[str, str]:
    return {
        "2": "2x — double frame rate (e.g. 24 -> 48 fps)",
        "4": "4x — quadruple frame rate (e.g. 24 -> 96 fps)",
        "8": "8x — octuple frame rate (e.g. 24 -> 192 fps)",
    }


def _dedup_entries_for(model_type: str) -> list[dict]:
    """Manifest entries required by one MultiPassDedup model type."""
    from video_upscaler import modelhub

    entries = modelhub.entries(group="dedup")
    if model_type == "gmfss":
        return [e for e in entries if str(e["dest"]).startswith("train_log_pg104/")]
    if model_type == "rife":
        return [e for e in entries if str(e["dest"]) == "rife48.pkl"]
    raise ValueError(f"Unknown MultiPassDedup model: {model_type}")


def ensure_dedup_weights(model_type: str, auto_download: bool = False) -> None:
    """Verify weights exist, offering a hub download on first use.

    Only entries whose destination file is absent are downloaded, so a
    partially-installed model gets repaired instead of crashing inference.
    Non-interactive runs raise SystemExit(1) when weights are missing unless
    auto_download=True.
    """
    model_type = validate_model_type(model_type)
    missing = check_dedup_weights(model_type)
    if not missing:
        return

    from video_upscaler import modelhub
    needed = modelhub.missing_entries(_dedup_entries_for(model_type))

    if auto_download and needed:
        for entry in needed:
            modelhub.install_entry(entry)
        return

    if _interactive():
        size_mb = sum(int(e["size"]) for e in needed) / (1 << 20)
        label = DEDUP_MODEL_NAMES.get(model_type, model_type.upper())
        if _prompt_confirm(
            f"MultiPassDedup weights for {label} are not installed. "
            f"Download them (~{size_mb:.0f} MB) into models/multipassdedup/?"
        ):
            try:
                for entry in needed:
                    modelhub.install_entry(entry)
                return
            except modelhub.HubError as exc:
                print()
                print(str(exc))
                raise SystemExit(1) from exc

    print()
    print(missing)
    raise SystemExit(1)


def build_dedup_plan() -> dict:
    """Prompt for MultiPassDedup parameters and construct an execution plan."""
    if _interactive():
        cadence_key = _prompt_choice("Duplicate cadence:", _cadence_options())
        model_key = _prompt_choice("Interpolation model:", _model_options())
        factor_key = _prompt_choice("Interpolation multiplier (frame rate):", _factor_options())
    else:
        cadence_key = config.DEDUP_NPASS_DEFAULT
        model_key = config.DEDUP_MODEL_DEFAULT
        factor_key = "2"

    model_type = validate_model_type(model_key)
    npass = parse_npass(cadence_key)
    factor = int(factor_key)

    ensure_dedup_weights(model_type)

    device = detect_dedup_device()
    device_label = "CUDA (torch)" if device == "cuda" else "CPU (torch)"
    model_label = DEDUP_MODEL_NAMES.get(model_type, model_type.upper())
    cadence_label = f"Auto (np=0)" if npass == 0 else f"On {'Twos' if npass == 2 else 'Threes'} (np={npass})"

    return {
        "action_label": "Interpolate (MultiPassDedup)",
        "engine_label": f"MultiPassDedup ({model_type.upper()})",
        "header_lines": [
            f"Model: {model_label}",
            f"Cadence: {cadence_label}",
            f"Factor: {factor}x",
            f"Device: {device_label}",
            f"Scale: {config.DEDUP_SCALE_DEFAULT}",
            f"Scene Detect: {'on' if config.DEDUP_SCDET_DEFAULT else 'off'}",
        ],
        "summary_lines": [
            f"\nAction:\n  Interpolate (MultiPassDedup)",
            f"\nModel:\n  {model_label}",
            f"\nCadence:\n  {cadence_label}",
            f"\nFactor:\n  {factor}x",
            f"\nDevice:\n  {device_label}",
        ],
        "model": model_type,
        "npass": npass,
        "factor": factor,
    }


def run_dedup_infer(
    video_in: Path,
    video_out: Path,
    model_type: str,
    npass: int,
    factor: int,
    scale: float = 1.0,
    enable_scdet: bool = True,
    scdet_threshold: float = 0.3,
    hwaccel: bool = False,
    progress_cb=None,
) -> None:
    """Execute MultiPassDedup inference on a video file via infer.py."""
    model_type = validate_model_type(model_type)
    ensure_dedup_weights(model_type, auto_download=True)
    script_path = config.BASE_DIR / "src" / "video_upscaler" / "multipass_dedup" / "infer.py"
    weights_dir = config.DEDUP_MODELS_DIR

    if not script_path.exists():
        raise FileNotFoundError(f"MultiPassDedup infer.py not found at {script_path}")

    # Temporary directory for intermediate output
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_out = Path(tmpdir) / f"temp_{video_in.name}"
        
        env = os.environ.copy()
        pythonpath_parts = [
            str(config.BASE_DIR / "src" / "video_upscaler" / "multipass_dedup"),
            str(config.BASE_DIR / "src"),
        ]
        if "PYTHONPATH" in env:
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        cmd = [
            sys.executable,
            str(script_path),
            "-i", str(video_in.resolve()),
            "-o", str(temp_out.resolve()),
            "-np", str(npass),
            "-t", str(factor),
            "-m", str(model_type),
            "-scale", str(scale),
            "-st", str(scdet_threshold),
            "-w", str(weights_dir.resolve()),
        ]
        if enable_scdet:
            cmd.append("-s")
        if hwaccel:
            cmd.append("-hw")


        # Run process
        proc = subprocess.Popen(
            cmd,
            cwd=str(config.BASE_DIR / "src" / "video_upscaler" / "multipass_dedup"),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Stream output
        output_lines = []
        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                output_lines.append(line)
        proc.wait()

        if proc.returncode != 0:
            error_msg = "".join(output_lines).strip()
            raise RuntimeError(f"MultiPassDedup failed (code {proc.returncode}):\n{error_msg}")

        # If infer.py created temp_out or output file, move it to video_out
        if temp_out.exists():
            video_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_out), str(video_out))
        else:
            # Check if an output was generated in temp directory
            generated = list(Path(tmpdir).glob("*"))
            if generated:
                shutil.move(str(generated[0]), str(video_out))
            else:
                raise RuntimeError(f"MultiPassDedup did not generate an output video.")
