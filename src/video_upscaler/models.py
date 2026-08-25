"""Processing profiles (scale + model variant + engine).

Each profile pairs a name with the exact Real-CUGAN model file it runs and
a short description shown in the CLI menu. The first entry is the default
for non-interactive/scripted runs. TensorRT covers 2x models on NVIDIA;
3x/4x always use the torch engine; ncnn Vulkan covers every profile.

The interpolation section below describes the AMT frame-interpolation
models used by the "Interpolate" action (slow motion). It is metadata only
(maps, descriptions); the actual torch model construction lives in
``interp.py`` so this module stays import-safe for torch-free unit tests.
"""

from __future__ import annotations

# profile name -> (model filename, description)
PROFILES: dict[str, tuple[str, str]] = {
    "2x_Balanced": (
        "up2x-latest-denoise2x.pth",
        "2x with moderate denoising — the everyday default",
    ),
    "2x_Clean": (
        "up2x-latest-no-denoise.pth",
        "2x pure upscale without denoising — preserves original grain and texture exactly",
    ),
    "2x_Light": (
        "up2x-latest-denoise1x.pth",
        "2x with light denoising — gently cleans noise while keeping most detail",
    ),
    "2x_Deep": (
        "up2x-latest-denoise3x.pth",
        "2x with strong denoising — best for heavily compressed or noisy sources",
    ),
    "2x_Faithful": (
        "up2x-latest-conservative.pth",
        "2x with minimal processing — preserves the original art style and avoids AI artifacts",
    ),
    "3x_Clean": (
        "up3x-latest-no-denoise.pth",
        "3x pure upscale without denoising — preserves original grain and texture exactly",
    ),
    "3x_Deep": (
        "up3x-latest-denoise3x.pth",
        "3x with strong denoising — best for heavily compressed or noisy sources",
    ),
    "3x_Faithful": (
        "up3x-latest-conservative.pth",
        "3x with minimal processing — preserves the original art style and avoids AI artifacts",
    ),
    "4x_Clean": (
        "up4x-latest-no-denoise.pth",
        "4x pure upscale without denoising — preserves original grain and texture exactly",
    ),
    "4x_Deep": (
        "up4x-latest-denoise3x.pth",
        "4x with strong denoising — best for heavily compressed or noisy sources",
    ),
    "4x_Faithful": (
        "up4x-latest-conservative.pth",
        "4x with minimal processing — preserves the original art style and avoids AI artifacts",
    ),
}

# ncnn Vulkan port args: CUGAN denoise level (-1 no-denoise, 0 conservative,
# 1..3 denoise). Every profile has an ncnn model.
_CUGAN_NCNN_NOISE = {
    "no-denoise": -1,
    "conservative": 0,
    "denoise1x": 1,
    "denoise2x": 2,
    "denoise3x": 3,
}


def default_profile() -> str:
    """Return the default profile name (first entry)."""
    return next(iter(PROFILES))


def model_for_profile(profile: str) -> str:
    """Return the model filename for a profile."""
    return PROFILES[profile][0]


def description_for_profile(profile: str) -> str:
    """Return the CLI description for a profile."""
    return PROFILES[profile][1]


def scale_for_model(model_name: str) -> int:
    """Derive the upscale factor from a model name (up2x.. -> 2)."""
    for scale in (2, 3, 4):
        if f"up{scale}x" in model_name:
            return scale
    raise ValueError(f"Could not determine scale from model name: {model_name}")


def ncnn_args_for_profile(profile: str) -> dict:
    """Return ncnn-vulkan CLI args for a profile: {"s": scale, "n": noise}."""
    model = model_for_profile(profile)
    scale = scale_for_model(model)
    for suffix, noise in _CUGAN_NCNN_NOISE.items():
        if suffix in model:
            return {"s": scale, "n": noise}
    raise ValueError(f"Profile {profile} has no ncnn noise mapping for {model}")


# --------------------------------------------------------------------------
# Interpolation (AMT frame interpolation -> slow motion)
# --------------------------------------------------------------------------
# Slow-motion factor -> AMT ``niters`` (recursive 2x passes).
# niters=1 -> 2x, niters=2 -> 4x, niters=3 -> 8x.  See AMT demo_2x.py.
INTERP_FACTORS: dict[int, int] = {2: 1, 4: 2, 8: 3}

# model key -> (checkpoint filename, CLI description)
INTERP_MODELS: dict[str, tuple[str, str]] = {
    "AMT-S": (
        "amt-s.pth",
        "AMT-S: lightweight, fastest — best for long clips and low VRAM",
    ),
    "AMT-L": (
        "amt-l.pth",
        "AMT-L: balanced quality and speed (recommended default)",
    ),
    "AMT-G": (
        "amt-g.pth",
        "AMT-G: highest quality, heaviest and slowest",
    ),
}

DEFAULT_INTERP_MODEL = "AMT-S"

# Hugging Face mirror for the official AMT pretrained checkpoints.
AMT_CKPT_BASE = "https://huggingface.co/lalala125/AMT/resolve/main"


def default_interp_model() -> str:
    """Return the default interpolation model key (first entry)."""
    return DEFAULT_INTERP_MODEL


def ckpt_for_interp_model(model_key: str) -> str:
    """Return the checkpoint filename for an interpolation model key."""
    return INTERP_MODELS[model_key][0]


def description_for_interp_model(model_key: str) -> str:
    """Return the CLI description for an interpolation model key."""
    return INTERP_MODELS[model_key][1]


def niters_for_factor(factor: int) -> int:
    """Map a slow-motion factor (2/4/8) to AMT ``niters`` (1/2/3)."""
    return INTERP_FACTORS[factor]


# --------------------------------------------------------------------------
# Matting (MatAnyone2) — metadata only; loading lives in matanyone2/model.py
# --------------------------------------------------------------------------
MATANYONE_CKPT_NAME = "matanyone2.pth"

# Official release used to seed the hub copy (CLARITY_MODELS/matanyone/).
MATANYONE_SOURCE_URL = (
    "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
)
