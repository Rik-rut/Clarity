# ✨ Clarity — AI Video Upscaler & Interpolation Studio

> **Transform low-resolution, choppy videos into crisp, high-fps masterpieces with a single click.**

Clarity is an all-in-one AI video enhancement studio for **Windows, Linux, and macOS**. It comes with a modern **Web Studio interface** where you can drag and drop videos, compare Before & After results side-by-side with interactive zoom and scrubbing, extract subjects with **Easy Mask**, and batch process videos using state-of-the-art AI models.

---

## 🚀 Quick Start (1-Click Launch)

No coding, complex installation, or terminal commands needed. Clarity handles everything automatically!

### 1. Download Clarity
- Click the green **Code** button at the top right of this page and select **Download ZIP**.
- Extract the ZIP folder anywhere on your computer (e.g., your Desktop or Documents).

### 2. Launch Clarity
- **Windows:** Double-click **`run.bat`**
- **Mac / Linux:** Open a terminal in the folder and run `bash run.sh` (or `./run.sh`)

> 💡 *On your very first run, Clarity will automatically configure Python, install required AI libraries, video decoders (FFmpeg), and NVIDIA GPU acceleration (TensorRT). Once finished, your browser will immediately open with the **Clarity Web Studio**!*
> 
> *On daily runs, double-clicking `run.bat` launches the studio in under a second.*

---

## 🎨 What Can You Do With Clarity?

Clarity provides four powerful AI tools in one unified studio:

| Mode | What It Does | Best For |
| :--- | :--- | :--- |
| 🔍 **Upscale** | Increases video resolution (**2x, 3x, or 4x**) up to crisp 4K while removing blur, compression artifacts, and noise. | Anime, animations, retro cartoons, and vintage clips. |
| ⏳ **Slow-Motion** | Generates brand-new AI intermediate frames to create ultra-smooth **2x, 4x, or 8x** slow-motion. | Action shots, sports replays, smooth motion clips. |
| 🎞️ **Interpolate** | Replaces duplicate/choppy frames with newly synthesized motion (**2x, 4x, or 8x** fps multiplier). | Smoothing 24fps anime to silky 48fps or 60fps+ fluid playback. |
| 🎭 **Easy Mask** | 1-click subject extraction and background separation. Outputs a **green-screen composite**, an **alpha matte**, and optional **transparent video**. | Background replacement, VFX compositing, rotoscoping subjects. |

---

## 🖥️ How to Use the Web Studio

![Clarity Web Studio Interface](src/video_upscaler/web/Screenshot%202026-08-26%20193733.png)

1. **Add Your Videos**:
   - Simply drag and drop video files (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, etc.) onto the left panel or player area.
   - You can also paste videos directly into the `input/` folder.
2. **Choose Your AI Action**:
   - Click **Upscale**, **Slow-motion**, **Interpolate**, or **Easy Mask** in the top navigation bar.
   - Choose your scale or speed multiplier (`2x`, `3x`, `4x`, `8x`).
   - Pick a model card that suits your video (e.g. *Balanced* for everyday videos, *Deep* for noisy clips).
3. **Compare & Inspect**:
   - Preview original and enhanced videos side-by-side.
   - Use the **1-Window / 2-Window** toggle button to switch viewing modes.
   - Scroll your mouse wheel to zoom in (up to 1000%) and drag to pan across fine details.
   - Use the synchronized timeline scrubber and frame-by-frame step buttons to inspect individual frames.
4. **Batch Processing (Multiple Videos)**:
   - Select multiple videos just like in Windows File Explorer: hold **Ctrl + Click** to select individual files, or **Shift + Click** to select a range.
   - Click the **Render Queue** button at the bottom left to inspect queued jobs.
5. **Hit Render**:
   - Click **Render Video**. A live progress bar will show the render stage, percent, elapsed time, and ETA.
   - When finished, your output video is saved to the `output/` folder and loaded into the player for instant review!
6. **Reset App Anytime**:
   - Click the **Reset App** button at the top right to instantly purge GPU VRAM caches and restore the studio to its pristine initial state.

---

## 🎭 Easy Mask: Subject & Background Separation

Easy Mask is Clarity's AI-powered video matting tool — mark a person or object on the first frame, and Easy Mask tracks and segments it cleanly through the entire clip. 

The workspace features a **4-window stage**:
1. **Input Video**: Original footage with timeline synchronization.
2. **Target Mask**: First-frame interactive painter and auto-detector.
3. **Green Screen Result**: Subject composited over pure chroma green (`#00FF00`).
4. **Alpha Matte Result**: High-contrast black & white luminance transparency mask.

### How to use Easy Mask:
1. Open the **Easy Mask** tab and select a video from your list.
2. Click **Select Target on Frame 1**.
3. Choose **Auto-Detect** (✨) and click directly on the subject (powered by Segment Anything). You can also use **Add** (+) or **Erase** (-) brush tools to paint or touch up edges.
4. Customize outputs: choose whether to generate **Green Screen**, **Alpha Matte**, or **Transparent ProRes 4444 video**.
5. Click **Render Easy Mask** at the bottom. When finished, all preview viewports update with the keyed results!

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

### 4. Matting & Segmentation Models (Easy Mask)
- **Easy Mask (MatAnyone2)**: State-of-the-art video object matting. Accurately tracks transparent hair, edges, and motion blur through entire video clips.
- **SAM (Segment Anything)**: High-precision point-and-click subject detection on frame 1.

---

## ⚡ Hardware Acceleration

Clarity automatically detects your computer's hardware and selects the fastest acceleration available:

- **NVIDIA GeForce / RTX GPUs**: Runs with dedicated **TensorRT (FP16)** CUDA streams for maximum GPU throughput.
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
<summary><b>How do I re-run setup or repair packages?</b></summary>

- **Windows:** Run `run.bat --setup` in command prompt.
- **Mac / Linux:** Run `bash run.sh --setup`.
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
  - [GMFSS / MultiPassDedup / RIFE](https://github.com/AlexWortega/MultiPassDedup).
  - [MatAnyone2](https://github.com/pq-yang/MatAnyone2) by pq-yang / S-Lab (Non-Commercial use).
