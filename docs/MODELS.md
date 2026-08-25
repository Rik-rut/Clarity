# Model Guide — `models/`

Clarity uses two model families:

- **Real-CUGAN** (bilibili AILab) — an anime super-resolution U-Net used for **Upscale** (MIT-licensed). Runs on TensorRT/CUDA/ncnn Vulkan (TensorRT for 2x on NVIDIA, torch for 3x/4x, ncnn Vulkan on other GPUs).
- **AMT** (MCG-NKU) — a frame-interpolation network used for **Interpolate / slow motion** (CC BY-NC 4.0, non-commercial only). AMT-S, AMT-L, and AMT-G run on TensorRT FP16 or the PyTorch backend via vendored code in `src/video_upscaler/amt/`.

## Real-CUGAN

All weights are an anime super-resolution U-Net trained on a million-scale anime patch dataset. It supports 2x, 3x, and 4x upscaling, each with a few variants that differ in how aggressively they clean up the source.

Every model ships three behaviors (except 2x, which has a bonus mid denoise level):

| Variant | What it does | Best for |
|---------|--------------|----------|
| `no-denoise` | Pure restoration/upscaling, no noise reduction | Already-clean, high-bitrate sources; keeps original grain |
| `denoise` | Restoration + noise/artifact removal | Compressed, noisy, or low-quality sources |
| `conservative` | Mild restoration that preserves the original look | When you don't want visible AI processing |

---

## 2x models (scale 2)

| File | Profile name | Description |
|------|--------------|-------------|
| `up2x-latest-no-denoise.pth` | **Clean** | 2x pure upscale without denoising — preserves original grain and texture exactly |
| `up2x-latest-denoise1x.pth` | **Light** | 2x with light denoising — gently cleans noise while keeping most detail |
| `up2x-latest-denoise2x.pth` | **Balanced** ⭐ default | 2x with moderate denoising — the everyday default; good all-rounder |
| `up2x-latest-denoise3x.pth` | **Deep** | 2x with strong denoising — best for heavily compressed or noisy sources |
| `up2x-latest-conservative.pth` | **Faithful** | 2x with minimal processing — preserves the original art style and avoids AI artifacts |

## 3x models (scale 3)

| File | Profile name | Description |
|------|--------------|-------------|
| `up3x-latest-no-denoise.pth` | **Clean** | 3x pure upscale without denoising |
| `up3x-latest-denoise3x.pth` | **Deep** | 3x with strong denoising |
| `up3x-latest-conservative.pth` | **Faithful** | 3x minimal processing, preserves original look |

## 4x models (scale 4)

| File | Profile name | Description |
|------|--------------|-------------|
| `up4x-latest-no-denoise.pth` | **Clean** | 4x pure upscale without denoising |
| `up4x-latest-denoise3x.pth` | **Deep** ⭐ default | 4x with strong denoising — the default for 4x mode |
| `up4x-latest-conservative.pth` | **Faithful** | 4x minimal processing, preserves original look |

---

## Profile names at a glance

| Profile | Meaning | File suffix |
|---------|---------|-------------|
| **Clean** | No denoising — pure upscale, keeps grain | `no-denoise` |
| **Light** | Light denoising (2x only) | `denoise1x` |
| **Balanced** | Moderate denoising — default for 2x | `denoise2x` |
| **Deep** | Strong denoising — default for 4x | `denoise3x` |
| **Faithful** | Conservative — minimal AI processing | `conservative` |

## How Clarity uses them

Every model is a selectable **profile** in the CLI menu, shown as `Name | description` — e.g. `2x_Clean | 2x pure upscale without denoising — preserves original grain and texture exactly`. The first profile (`2x_Balanced`) is the default for non-interactive runs.

- 2x profiles: `2x_Clean` · `2x_Light` · `2x_Balanced` · `2x_Deep` · `2x_Faithful`
- 3x profiles: `3x_Clean` · `3x_Deep` · `3x_Faithful`
- 4x profiles: `4x_Clean` · `4x_Deep` · `4x_Faithful`

---

## ncnn Vulkan models (`tools/ncnn/`)

On non-CUDA backends Clarity runs the portable ncnn Vulkan tool (auto-downloaded with consent from the official GitHub releases). Real-CUGAN has full ncnn support for every profile.

| Tool | Source repo | Models used |
|------|-------------|-------------|
| `realcugan-ncnn-vulkan.exe` | [nihui/realcugan-ncnn-vulkan](https://github.com/nihui/realcugan-ncnn-vulkan) | Real-CUGAN, all scales |

Real-CUGAN denoise levels are mapped from the profile's model suffix:

| Model suffix | Profile variant | ncnn `-n` level |
|--------------|-----------------|-----------------|
| `no-denoise` | Clean | `-1` |
| `conservative` | Faithful | `0` |
| `denoise1x` | Light | `1` |
| `denoise2x` | Balanced | `2` |
| `denoise3x` | Deep | `3` |

## TensorRT engines (`models/tensorrt/`)

On NVIDIA with the `tensorrt` extra, Clarity exports each Real-CUGAN model to ONNX and builds a cached fp16 engine:

| Artifact | Purpose |
|----------|---------|
| `<model>_x2_fp16_<gpu>.engine` | Serialized TensorRT engine (built once, keyed by model hash / GPU / TRT version) |
| `metadata.json` | Engine cache metadata; engines rebuild automatically when stale |
| `<model>_x2.onnx` | Intermediate ONNX export (kept for debugging) |

TensorRT covers 2x profiles only; 3x/4x profiles use the torch engine (auto mode falls back automatically).

---

## AMT (Interpolate / slow motion)

[AMT](https://github.com/mcg-nku/amt) (All-Pairs Multi-Field Transforms, CVPR 2023) synthesizes in-between frames to slow video down. It is **CC BY-NC 4.0 — non-commercial use only**. The action keeps the original resolution and frame rate, inserting interpolated frames so a 2x/4x/8x factor yields half/quarter/eighth-speed playback.

| File | Model key | Notes |
|------|-----------|-------|
| `amt-s.pth` | **AMT-S** ⭐ default | Lightweight, fastest — best for long clips and low VRAM |
| `amt-l.pth` | **AMT-L** | Balanced quality and speed |
| `amt-g.pth` | **AMT-G** | Highest quality, heaviest and slowest |

The recursive 2x passes (`niters`) map to the slow-motion factor: 2x→1, 4x→2, 8x→3. Checkpoints are auto-downloaded on first use from the Hugging Face mirror (`CLARITY_AMT_CKPT_BASE`) into `models/`.

### AMT-S / AMT-L / AMT-G TensorRT engine (`models/amt/tensorrt/`)

On NVIDIA with the `tensorrt` extra, AMT-S, AMT-L, and AMT-G interpolation use a cached TensorRT **FP16** engine. The engine is built from a static-spatial ONNX export of the pair graph (input `frame_a`/`frame_b`, output `output`, fixed padded spatial dims, dynamic batch up to `CLARITY_AMT_BATCH`), keyed by checkpoint hash, GPU, CUDA/TensorRT versions, padded shape, scale, precision, batch, and opset. Engines rebuild automatically when any key changes.

| Artifact | Purpose |
|----------|---------|
| `AMT-<S\|L\|G>_<H>x<W>_s<scale>_fp16_b<batch>_o<opset>_e<version>_<gpu>.engine` | Serialized TensorRT FP16 engine (built once per profile) |
| `AMT-<S\|L\|G>_<H>x<W>_...onnx` | Intermediate static-spatial ONNX export (kept for debugging) |
| `*.json` | Cache metadata; engines rebuild automatically when stale |

The profile uses a **static spatial shape** (the AMT adaptive downscale plus 16-divisible padding at the source resolution), so engines are per-resolution. The internal AMT adaptive scale is fixed into the graph (e.g. 0.64x for 1080p on the reference RTX 3050). PyTorch remains the automatic fallback when the runtime or a compatible engine is unavailable. Auto batch defaults are model-aware: AMT-S defaults to 2 on CUDA, AMT-L and AMT-G to 1 (they are heavier and fit the 6 GB card more safely at batch 1); `CLARITY_AMT_BATCH` overrides any of them.
