# ✨ Clarity — AI Video Upscaler & Interpolation Studio

> **Transform low-resolution, choppy videos into crisp, high-fps masterpieces with a single click.**

Clarity is an all-in-one AI video enhancement studio for **Windows, Linux, and macOS**. It comes with a modern **Web Studio interface** where you can drag and drop videos, compare Before & After results side-by-side with interactive zoom and scrubbing, and batch process videos using state-of-the-art AI models.

---

## 🚀 Quick Start (Easy 3-Step Setup)

No coding or complex setup needed. Clarity handles everything automatically!

### 1. Download Clarity
- Click the green **Code** button at the top right of this page and select **Download ZIP**.
- Extract the ZIP folder to anywhere on your computer (e.g., your Desktop or Documents).

### 2. Run the One-Time Setup
- **Windows:** Double-click **`Setup.bat`**
- **Mac / Linux:** Open a terminal in the folder and run `bash setup.sh`

> 💡 *The setup assistant will automatically install Python, the required AI libraries, video decoders (FFmpeg), and NVIDIA GPU acceleration (TensorRT).*

### 3. Launch Clarity
- **Windows:** Double-click **`run.bat`**
- **Mac / Linux:** Run `bash run.sh`

Your web browser will automatically open with the **Clarity Web Studio**!

---

## 🎨 What Can You Do With Clarity?

Clarity provides four powerful AI tools in one clean interface:

| Mode | What It Does | Best For |
| :--- | :--- | :--- |
| 🔍 **Upscale** | Increases video resolution (**2x, 3x, or 4x**) up to crisp 4K while removing blur and compression artifacts. | Anime, animations, retro cartoons, and vintage clips. |
| ⏳ **Slow-Motion** | Generates brand-new AI intermediate frames to create ultra-smooth **2x, 4x, or 8x** slow-motion. | Action shots, sports replays, smooth motion clips. |
| 🎞️ **Interpolate** | Replaces duplicate/choppy frames with newly synthesized motion (**2x, 4x, or 8x** fps multiplier). | Smoothing 24fps anime to silky 48fps or 60fps+ fluid playback. |
| 🎭 **MatAnyone2** | Clicks or paints a subject out of a video in one pass, producing a **green-screen composite**, an **alpha matte**, and an optional **transparent WebM** alongside the original. | Removing or replacing backgrounds, rotoscoping people and objects. |

---

## 🖥️ How to Use the Web Studio

```
 ┌───────────────────────────┬─────────────────────────────────────────────────┐
 │       INPUT VIDEOS        │               INTERACTIVE PLAYER                │
 │  ┌─────────────────────┐  │  ┌──────────────────────┬────────────────────┐  │
 │  │ Drag & Drop Videos  │  │  │                      │                    │  │
 │  └─────────────────────┘  │  │      [BEFORE]        │      [AFTER]       │  │
 │  • video1.mp4  [...]      │  │      Original        │    AI Enhanced     │  │
 │  • video2.mp4  [...]      │  │                      │                    │  │
 │  [Clear Videos]           │  └──────────────────────┴────────────────────┘  │
 ├───────────────────────────┤  [◀◀] [▶ / ⏸] [▶▶] [⏮] ────⚪────────── [100%]   │
 │   UPSCALE / SLOWMO / MAT  │                                                 │
 │  • Multiplier: [2x] [4x]  │  ┌───────────────────────────────────────────┐  │
 │  • Model: Balanced / Deep │  │ Output Destination: /output/       [ ⌵ ]  │  │
 ├───────────────────────────┤  │ 📂 Output Videos & Batch Queue            │  │
 │      [ RENDER VIDEO ]     │  └───────────────────────────────────────────┘  │
 └───────────────────────────┴─────────────────────────────────────────────────┘
```

1. **Add Your Videos**:
   - Simply drag and drop video files (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, etc.) onto the left panel or player area.
   - You can also paste videos directly into the `input/` folder.
2. **Choose Your AI Action**:
   - Click **Upscale**, **Slow-motion**, **Interpolate**, or **MatAnyone2** at the top left.
   - Choose your scale/multiplier (`2x`, `3x`, `4x`, `8x`).
   - Pick a model card that suits your video (e.g. *Balanced* for everyday videos, *Deep* for noisy clips).
3. **Compare & Inspect**:
   - Preview original and enhanced videos side-by-side.
   - Use the **1-Window / 2-Window** toggle button to switch viewing modes.
   - Scroll your mouse wheel to zoom in (up to 1000%) and drag to pan across details.
   - Use the synchronized timeline scrubber and frame-by-frame step buttons to inspect individual frames.
4. **Batch Processing (Multiple Videos)**:
   - Select multiple videos just like in Windows File Explorer: hold **Ctrl + Click** to select individual files, or **Shift + Click** to select a range.
   - Click the **Render Queue** button at the bottom left to see the full queue.
5. **Hit Render**:
   - Click **Render Video**. A live progress bar will show the render stage, percent, elapsed time, and ETA.
   - When finished, your output video is saved to the `output/` folder and loaded into the player for instant review!

---

## 🎭 MatAnyone2: Object Matting

MatAnyone2 is Clarity's object-matting model — mark a person or object on the
first frame and MatAnyone2 tracks it through the entire clip. The tab uses a
**4-window workspace**: input video, first-frame target editor, green-screen
result, and alpha-matte result, with all controls in the left card. Each
render can produce three files in your output folder: a
**`*_greenscreen.mp4`** where the subject is composited over a green screen
(for background replacement or chroma keying), a **`*_matte.mp4`** containing
the alpha matte (the real transparency mask), and an optional
**`*_transparent.mov`** (ProRes 4444) with true per-pixel alpha, so the
original video stays untouched.

1. Open the **MatAnyone2** tab and select your clip.
2. Click **Select Target on Frame 1**, then either click **Auto-Detect** and
   simply click the subject (SAM-powered, same interaction as the official
   demo — click again to refine the selection), or paint a rough mask —
   you can **Add** / **Remove**, **Undo / Redo**, and the magenta **Preview**
   overlay is on by default.
3. Fine-tune the **Dilate** / **Erode** settings if the matte edge should grow
   or shrink, pick the **Precision / Backend** for your hardware, and choose
   whether to keep the **Alpha Matte**, the **Green Screen**, and/or the
   **Transparent ProRes**.
4. Click **RENDER MATANYONE2**. The first frame's mask is mirrored through the
   whole clip automatically, and both result windows fill with the outputs.

> 💡 The ~141 MB matting model and the ~375 MB SAM click-segmentation model
> download on first use (or run
> `uv run --all-extras main.py --download-models all` to fetch the full model
> set upfront). Keep the painted mask roughly inside the subject — the model
> refines and tracks the exact boundary itself. On GPUs with 6 GB or less,
> inference internally runs at up to 720p shortest-side and upscales the
> matte back to native resolution (override with
> `CLARITY_MA2_INTERNAL_SIZE`; `0` disables the cap).

---

## 🧩 AI Models Explained (In Plain English)

### 1. Upscale Models (Real-CUGAN)
- **Balanced (Recommended)**: The best everyday setting. Sharpens details while cleaning up standard compression artifacts.
- **Clean**: Upscales purely without altering fine textures or film grain.
- **Deep Denoise**: Strong AI cleaning for heavily compressed, blurry, or low-bitrate web videos.
- **Faithful**: Gentle enhancement designed to preserve original artistic styles.

### 2. Slow-Motion Models (AMT)
- **AMT-S (Fast & Crisp)**: Lightweight, fast, and great for general slow-motion video.
- **AMT-L (High Quality)**: Deeper AI analysis for complex, fast-moving scenes.
- **AMT-G (Maximum Smoothness)**: Flagship model with the highest motion detail.

### 3. Interpolation Models (Anime Dedup)
- **GMFSS (Fortuna)**: Industry-standard anime frame interpolation. Ideal for 2D animation with duplicate frame cadences.
- **RIFE (Practical-RIFE)**: High-speed real-time interpolation with great motion flow.
- **GIMM (GIMM-VFI)**: Advanced INR interpolation that produces sharp transitions on fast movements.

### 4. Matting Model (MatAnyone2)
- **MatAnyone2**: State-of-the-art object matting. Click a subject or paint a
  rough mask on the first frame — it is tracked through the clip, outputting a
  green-screen composite, an alpha matte, and an optional transparent WebM.

---

## ⚡ Hardware Acceleration

Clarity automatically detects your computer's hardware and selects the fastest acceleration available:

- **NVIDIA GeForce / RTX GPUs**: Runs with **TensorRT (FP16)** and **CUDA** for blazing fast performance.
- **Apple Silicon Macs (M1/M2/M3/M4)**: Fully accelerated with Apple **Metal (MPS)**.
- **Intel & AMD GPUs**: Supported via **Vulkan (ncnn)**.
- **CPU**: Works on any computer even without a dedicated graphics card (processing will take longer).

---

## 💻 Optional: Interactive Terminal Mode (CLI)

Prefer using the command line? Clarity includes a full keyboard-driven terminal interface:

```bash
uv run main.py --cli
```

- Navigate menus with the **Arrow Keys**, press **Space** to toggle, and **Enter** to confirm.
- Run automated benchmarks with `uv run main.py benchmark`.

---

## ❓ Frequently Asked Questions & Troubleshooting

<details>
<summary><b>Where are my converted videos saved?</b></summary>

By default, all rendered videos are placed in the `output/` folder inside the Clarity directory. You can also click the folder path under **Output Destination** in the Web UI to pick any custom export folder on your computer.
</details>

<details>
<summary><b>Do I need an internet connection while rendering?</b></summary>

Only the first time you use a specific AI model (Clarity will automatically download and verify the model weights). Once downloaded, Clarity runs 100% offline on your machine.
</details>

<details>
<summary><b>It says "Setup.bat did not find uv" or won't launch</b></summary>

Close your terminal/command prompt window, open a new one, and run `Setup.bat` again. Windows requires a terminal restart to recognize newly installed environment paths.
</details>

<details>
<summary><b>How do I delete or clear videos?</b></summary>

- Click the **3 dots (`...`)** next to any video in the Input or Output list to **Rename** or **Delete** it.
- Click the **Clear Videos** button at the bottom of the list to remove all videos in that folder at once.
</details>

<details>
<summary><b>Why is my video taking a long time?</b></summary>

Higher multipliers (`4x` upscale, `8x` slowmo) and larger videos (like 4K) require more computation. If you do not have an NVIDIA GPU or Apple Silicon chip, your computer will process frames using the CPU, which is slower. For the fastest speeds, use `2x` on the *Balanced* or *AMT-S* profile.
</details>

---

## 📜 License & Acknowledgements

- **Clarity Core Application**: Released under the [MIT License](LICENSE).
- **AI Models & Engines**: Third-party models carry their respective licenses (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
  - [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) by bilibili AI Lab.
  - [AMT](https://github.com/mcg-nku/amt) by MCG-NKU (Non-Commercial use).
  - [GMFSS / MultiPassDedup / RIFE / GIMM-VFI](https://github.com/AlexWortega/MultiPassDedup).
  - [MatAnyone2](https://github.com/pq-yang/MatAnyone2) by pq-yang / S-Lab (Non-Commercial use).
