"""Interactive CLI front end: banner, menus, selection, confirmation,
progress display, and the final summary.

Module top level imports only pure modules (config/scanner); the
heavy processor/cugan/ffmpeg imports happen inside run() so the unit
tests never load torch.

Menus use arrow-key navigation via questionary. When stdin is not a TTY
(piped input, smoke tests) the flow falls back to safe defaults instead of
prompting.
"""

from __future__ import annotations

import sys
import time

import questionary

from video_upscaler import config
from video_upscaler.scanner import scan_videos

BANNER = (
    '██████╗██╗      █████╗ ██████╗ ██╗████████╗██╗   ██╗\n██╔════╝██║     ██╔══██╗██╔══██╗██║╚══██╔══╝╚██╗ ██╔╝\n██║     ██║     ███████║██████╔╝██║   ██║    ╚████╔╝ \n██║     ██║     ██╔══██║██╔══██╗██║   ██║     ╚██╔╝  \n╚██████╗███████╗██║  ██║██║  ██║██║   ██║      ██║   \n ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝      ╚═╝   '
)

TITLE = "Clarity — Video Upscaler"


def _profile_options() -> dict[str, str]:
    """Map profile name -> menu label ("2x_Clean | description")."""
    from video_upscaler.models import PROFILES

    return {name: f"{name} | {description}" for name, (_, description) in PROFILES.items()}


def _interactive() -> bool:
    """Return True when we can prompt (real terminal, not piped input)."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def select_videos(videos: list) -> list:
    """Choose videos via arrow keys; auto-select when only one exists.

    Nothing is pre-selected — the user toggles videos with space. An empty
    selection re-prompts (an empty batch is never accepted silently);
    Esc/Ctrl-C quits (SystemExit).
    """
    print("\nVideos found in input/:\n")
    for index, video in enumerate(videos, start=1):
        print(f"[{index}] {video.name}")

    if len(videos) == 1:
        print("\n1 video found — proceeding with it automatically.")
        return videos

    if not _interactive():
        print("\nNon-interactive mode — selecting all videos.")
        return videos

    names = [video.name for video in videos]
    warned = False
    while True:
        try:
            selection = questionary.checkbox(
                "Select videos (space to toggle, enter to confirm, Esc to quit):",
                choices=names,
            ).ask()
        except (KeyboardInterrupt, EOFError):
            raise SystemExit(0)
        if selection is None:  # user cancelled
            raise SystemExit(0)
        if not selection:
            if warned:
                print("\nNo videos selected — quitting.")
                raise SystemExit(0)
            print("\nNo videos selected — please choose at least one "
                  "(or press Enter again to quit).\n")
            warned = True
            continue
        return [video for video in videos if video.name in selection]


def _prompt_choice(title: str, options: dict[str, str]) -> str:
    """Arrow-key single-select; returns the option key."""
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
    """Arrow-key y/N confirmation."""
    if not _interactive():
        return True
    try:
        return questionary.confirm(title, default=False).ask()
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(0)


def _prompt_download_cugan() -> bool:
    """Ask the user whether to auto-download Real-CUGAN weights (~51 MB)."""
    if not _interactive():
        return False
    try:
        return questionary.confirm(
            "Real-CUGAN weights are not installed. Download them (~51 MB, "
            "official bilibili release) into models/?",
            default=False,
        ).ask()
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(0)


def _ensure_cugan(model_name: str) -> None:
    """Auto-download Real-CUGAN weights on first anime use (user consent)."""
    from video_upscaler.cugan import check_cugan, download_cugan_models

    missing = check_cugan(model_name)
    if not missing:
        return
    if not _prompt_download_cugan():
        print()
        print(missing)
        raise SystemExit(1)
    try:
        download_cugan_models()
    except RuntimeError as exc:
        print()
        print(str(exc))
        raise SystemExit(1) from exc


def _ensure_ncnn(tool: str) -> None:
    """Auto-download the ncnn Vulkan tool on non-NVIDIA fallback (consent)."""
    from video_upscaler.ncnn import check_ncnn, download_ncnn

    missing = check_ncnn(tool)
    if not missing:
        return
    if not _prompt_confirm(
        f"The ncnn Vulkan tool for {tool} is not installed. Download the "
        "portable package (~40 MB, official GitHub release) into tools/ncnn/?"
    ):
        print()
        print(missing)
        raise SystemExit(1)
    try:
        download_ncnn(tool)
    except RuntimeError as exc:
        print()
        print(str(exc))
        raise SystemExit(1) from exc


def _ensure_amt(model_key: str) -> None:
    """Auto-download an AMT checkpoint on first interpolate use (consent)."""
    from video_upscaler.interp import check_amt, download_amt_model

    missing = check_amt(model_key)
    if not missing:
        return
    if not _prompt_confirm(
        f"The AMT checkpoint for {model_key} is not installed. Download it "
        "from the official Hugging Face mirror (~tens of MB) into models/?"
    ):
        print()
        print(missing)
        raise SystemExit(1)
    try:
        download_amt_model(model_key)
    except RuntimeError as exc:
        print()
        print(str(exc))
        raise SystemExit(1) from exc


def _build_upscale_plan() -> dict:
    """Prompt for an upscale profile and ensure its engine assets exist."""
    from video_upscaler.backend import backend_label, detect_backend
    from video_upscaler.models import model_for_profile
    from video_upscaler.processor import effective_backend

    profile = _prompt_choice("Select a profile:", _profile_options())
    model_name = model_for_profile(profile)
    backend = detect_backend()
    backend = effective_backend(backend, profile)

    if _needs_torch_weights(backend):
        _ensure_cugan(model_name)
    else:
        _ensure_ncnn("realcugan")

    device_label = backend_label(backend)
    return {
        "action_label": "Upscale (Real-CUGAN)",
        "engine_label": "Real-CUGAN",
        "header_lines": [
            f"Profile: {profile}",
            f"Model: {model_name}",
            f"Backend: {device_label}",
        ],
        "summary_lines": [
            f"\nProfile:\n  {profile}",
            f"\nModel:\n  {model_name}",
            f"\nBackend:\n  {device_label}",
        ],
        "profile": profile,
    }


def _build_interp_plan() -> dict:
    """Prompt for interpolation factor + AMT model and ensure weights exist."""
    from video_upscaler.interp import select_amt_backend
    from video_upscaler.models import (
        INTERP_MODELS,
        DEFAULT_INTERP_MODEL,
        description_for_interp_model,
        ckpt_for_interp_model,
    )

    factor = int(
        _prompt_choice(
            "Interpolation factor (slow motion):",
            {
                "2": "2x — half speed",
                "4": "4x — quarter speed",
                "8": "8x — eighth speed",
            },
        )
    )
    # Non-interactive runs default to the first model (AMT-S).
    model_opts = {k: f"{k} | {description_for_interp_model(k)}" for k in INTERP_MODELS}
    model_key = _prompt_choice("Select an AMT model:", model_opts) if _interactive() else DEFAULT_INTERP_MODEL
    _ensure_amt(model_key)

    selection = select_amt_backend(model_key)
    if selection.backend == "tensorrt":
        device_label = "TensorRT (fp16)"
        engine_label = "AMT TensorRT static-spatial profile"
    else:
        from video_upscaler.cugan import detect_device

        device = detect_device()
        device_label = {
            "cuda": "CUDA (torch)",
            "mps": "Metal (MPS)",
        }.get(device, "CPU (torch)")
        engine_label = "AMT PyTorch"
    lines = [
        f"\nFactor:\n  {factor}x (slow motion)",
        f"\nModel:\n  {model_key} ({ckpt_for_interp_model(model_key)})",
        f"\nBackend:\n  {device_label}",
        f"\nPrecision:\n  {selection.precision}",
        f"\nBatch size:\n  {selection.batch_size}",
        f"\nEngine/profile:\n  {engine_label}",
    ]
    if selection.fallback_reason:
        lines.append(f"\nFallback:\n  {selection.fallback_reason}")
    return {
        "action_label": "Slow-motion (AMT)",
        "engine_label": "AMT",
        "header_lines": [
            f"Factor: {factor}x",
            f"Model: {model_key}",
            f"Backend: {device_label}",
            f"Precision: {selection.precision}",
            f"Batch: {selection.batch_size}",
            f"Engine/profile: {engine_label}",
            f"Window: {config.AMT_SEGMENT_FRAMES} source frames",
        ],
        "summary_lines": lines,
        "model_key": model_key,
        "factor": factor,
        "amt_selection": selection,
    }



def _needs_torch_weights(backend: str) -> bool:
    """Return True when the run needs torch model weights.

    Every profile has an ncnn Vulkan model, so only the ncnn backend skips
    the torch weight checks/downloads.
    """
    return backend != "ncnn"


_BACKEND_CHOICES = ("auto", "torch", "ncnn", "tensorrt")

_DOWNLOAD_MODEL_TIERS = ("essential", "all")


def _parse_download_models_arg(argv: list[str]) -> str | None:
    """Parse ``--download-models essential|all``; returns the tier or None."""
    if "--download-models" not in argv:
        return None
    index = argv.index("--download-models")
    if index + 1 >= len(argv):
        raise SystemExit(
            "--download-models requires a value: "
            + ", ".join(_DOWNLOAD_MODEL_TIERS) + "."
        )
    value = argv[index + 1].lower()
    if value not in _DOWNLOAD_MODEL_TIERS:
        raise SystemExit(
            f"Invalid --download-models {value!r}. Choose: "
            + ", ".join(_DOWNLOAD_MODEL_TIERS) + "."
        )
    return value


def _run_download_models(tier: str) -> None:
    """Install hub models non-interactively, then report and exit cleanly."""
    from video_upscaler import modelhub

    print(f"Model source: {modelhub.describe_local_base()}")
    try:
        count = modelhub.install_tier(tier=None if tier == "all" else tier)
    except modelhub.HubError as exc:
        print()
        print(str(exc))
        raise SystemExit(1) from exc
    print()
    print(f"Done — {count} model file(s) installed.")
    print("Start Clarity with run.bat (Windows) or: uv run main.py")


def _apply_cli_backend(argv: list[str]) -> None:
    """Parse ``--backend <choice>`` into config.BACKEND_PREF.

    Runs before the interactive workflow so ``uv run main.py --backend ncnn``
    behaves like ``CLARITY_BACKEND=ncnn``. Raises SystemExit on bad input.
    """
    if "--backend" not in argv:
        return
    index = argv.index("--backend")
    if index + 1 >= len(argv):
        raise SystemExit(
            "--backend requires a value: " + ", ".join(_BACKEND_CHOICES) + "."
        )
    value = argv[index + 1].lower()
    if value not in _BACKEND_CHOICES:
        raise SystemExit(
            f"Invalid --backend {value!r}. Choose: "
            + ", ".join(_BACKEND_CHOICES) + "."
        )
    config.BACKEND_PREF = value


def run() -> None:
    """Main entry point: banner first, then the interactive workflow."""
    # Windows: when stdout is piped/redirected Python falls back to cp1252,
    # which cannot encode the banner's box-drawing characters (or em-dashes
    # in benchmark progress output). Force UTF-8 before any subcommand.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    # Benchmark subcommand: `uv run main.py benchmark [flags]`. Runs before
    # the banner so piped benchmark output stays clean and non-interactive.
    if sys.argv[1:] and sys.argv[1] == "benchmark":
        from video_upscaler.benchmark import run_cli

        run_cli(sys.argv[2:])
        return

    # Optional backend override: `uv run main.py --backend ncnn`.
    _apply_cli_backend(sys.argv[1:])

    # 1-2. Banner and title — the very first terminal output.
    print(BANNER)
    print(TITLE)

    # Non-interactive model pre-download (used by setup scripts):
    # `uv run main.py --download-models essential|all`
    tier = _parse_download_models_arg(sys.argv[1:])
    if tier is not None:
        config.ensure_directories()
        _run_download_models(tier)
        return

    # 3. Create runtime directories if missing.
    config.ensure_directories()

    # 4. FFmpeg availability check.
    from video_upscaler.ffmpeg import check_ffmpeg

    missing_ffmpeg = check_ffmpeg()
    if missing_ffmpeg:
        print()
        print(missing_ffmpeg)
        raise SystemExit(1)

    # 5. Scan input/.
    videos = scan_videos(config.INPUT_DIR)
    if not videos:
        print("No supported videos were found in input/.\n")
        print("Add video files to:")
        print("input/")
        return  # graceful exit

    # 6-7. Video selection (arrow keys; auto-select when only one).
    selected = select_videos(videos)

    # 8. Action menu: Upscale (Real-CUGAN), Slow-motion (AMT), or Interpolate (MultiPassDedup).
    action = _prompt_choice(
        "Select an action:",
        {
            "Upscale": "Upscale (Real-CUGAN) — increase resolution",
            "Slow-motion": "Slow-motion (AMT) — smooth slow motion 2x / 4x / 8x",
            "Interpolate": "Interpolate (MultiPassDedup) — anime duplicate compensation 2x / 4x / 8x",
        },
    )

    if action == "Upscale":
        plan = _build_upscale_plan()
    elif action == "Slow-motion":
        plan = _build_interp_plan()
    else:
        from video_upscaler.dedup import build_dedup_plan

        plan = build_dedup_plan()

    # 9. Processing summary + confirmation.
    print("\n========================================")
    print("PROCESSING SUMMARY")
    print("========================================")
    print("\nVideos:")
    for video in selected:
        print(f"  {video.name}")
    print(f"\nAction:\n  {plan['action_label']}")
    for line in plan["summary_lines"]:
        print(line)
    print(f"\nOutput:\n  {config.OUTPUT_DIR}")
    if not _prompt_confirm("\nContinue?"):
        print("\nCancelled.")
        return

    # 10. Process with an in-place progress line per file.
    from video_upscaler.processor import (
        process_videos,
        process_interpolate,
        process_dedup,
        format_duration,
    )

    def progress_cb(file_index: int, file_count: int, percent: int) -> None:
        nonlocal last_header_index, _eta_start
        if file_index != last_header_index:
            print()  # blank line before the next file's header block
            print(f"[{file_index}/{file_count}] Processing: {selected[file_index - 1].name}")
            print(f"Engine: {plan['engine_label']}")
            for line in plan["header_lines"]:
                print(line)
            print()
            last_header_index = file_index
            _eta_start = time.perf_counter()
        width = 20
        filled = round(percent / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        eta = ""
        if percent > 0 and percent < 100 and _eta_start is not None:
            elapsed = time.perf_counter() - _eta_start
            if elapsed > 0 and percent > 2:
                remaining = elapsed / percent * (100 - percent)
                eta = f"  ETA {format_duration(remaining)}"
        print(f"\rProgress: {bar} {percent:3d}%{eta}", end="", flush=True)
        if percent >= 100:
            print()

    last_header_index = 0
    _eta_start = None
    if action == "Upscale":
        results = process_videos(selected, plan["profile"], progress_cb)
    elif action == "Slow-motion":
        results = process_interpolate(
            selected, plan["model_key"], plan["factor"], progress_cb
        )
    else:
        results = process_dedup(
            selected, plan["model"], plan["npass"], plan["factor"], progress_cb
        )


    # 13. Final summary.
    from video_upscaler.processor import format_duration

    print("\n========================================")
    print("PROCESSING COMPLETE")
    print("========================================")
    print(f"\nSuccessful:\n{len(results['success'])}")
    print(f"\nFailed:\n{len(results['failed'])}")
    print(f"\nOutput directory:\n{config.OUTPUT_DIR}")
    if results["success"]:
        print("\nTiming:")
        for video, elapsed in zip(selected, results["times"]):
            if elapsed is not None:
                print(f"  {video.name}: {format_duration(elapsed)}")
        print(f"\nTotal: {format_duration(sum(t for t in results['times'] if t is not None))}")
    for name, reason in results["failed"]:
        print(f"\nFailed: {name}\nReason: {reason}")
    if results["failed"]:
        raise SystemExit(1)
