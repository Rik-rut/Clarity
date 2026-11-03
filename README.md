# Clarity - AI Video Upscaler and Interpolation Studio

> Enhance low-resolution or low-frame-rate videos into clear, high-definition videos with a clean, easy-to-use interface.

Clarity is an all-in-one video enhancement desktop application for Windows, macOS, and Linux. It provides a visual studio interface where you can drag and drop videos, compare before-and-after results side-by-side, cut out subjects with automatic masking, and process multiple videos at once using modern AI models.

---

## Quick Start

### Option 1: Desktop Installer (Windows - Recommended)

1. Go to the [Releases](https://github.com/Rik-rut/Clarity/releases) page.
2. Download the latest installer (`Clarity_x.x.x_x64-setup.exe`).
3. Run the installer and launch Clarity from your Start menu or desktop shortcut.

Automatic updates are built into the desktop app, so future updates can be installed with a single click.

### Option 2: Portable Launch (No Installation Required)

1. Download the repository by clicking the green **Code** button and selecting **Download ZIP**.
2. Extract the ZIP file to a folder of your choice.
3. Launch the app:
   - **Windows:** Double-click `run.bat`
   - **macOS / Linux:** Open a terminal in the folder and run `bash run.sh`

> Note: On your first launch, Clarity will automatically configure Python, download necessary media components (FFmpeg), and detect your graphics card. Once ready, the studio opens automatically in your browser or desktop window. On subsequent launches, the studio opens in seconds.

---

## Core Features

Clarity provides four dedicated video tools in one unified studio:

| Tool | What It Does | Recommended Use |
| :--- | :--- | :--- |
| **Upscale** | Increases video resolution (**1x, 2x, 3x, or 4x**) up to 4K while sharpening details and removing noise and blur. (1x mode cleans and sharpens without changing original dimensions). | Animated content, older videos, web clips, and compressed footage. |
| **Slow-Motion** | Generates new frames between existing ones to create smooth slow-motion (**2x, 4x, or 8x** slower). | Action shots, sports highlights, and cinematic b-roll. |
| **Interpolate** | Replaces duplicate or choppy frames with synthesized fluid motion (**2x, 4x, or 8x** frame rate). | Converting 24 FPS animation to fluid 48 FPS or 60+ FPS motion. |
| **Easy Mask** | Automatically isolates a person or object across an entire video clip. Generates a green screen version, a black-and-white alpha matte, or a transparent video. | Background replacement, video editing, and visual effects. |

---

## How to Use the Studio

![Clarity Studio Interface](src/video_upscaler/web/Screenshot%202026-08-26%20193733.png)

1. **Add Your Videos**:
   - Drag and drop video files (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`) into the file panel on the left.
   - You can also place files directly inside the `input/` folder in the Clarity directory.

2. **Choose a Tool and Settings**:
   - Select **Upscale**, **Slow-motion**, **Interpolate**, or **Easy Mask** from the top menu.
   - Choose your scale multiplier or speed factor.
   - Select a model preset:
     - *Balanced*: Best for general everyday videos.
     - *Clean*: Preserves original film grain and subtle textures.
     - *Deep Denoise*: Ideal for heavily compressed, blurry, or low-quality clips.
     - *Faithful*: Mild enhancement focused on preserving artistic style.

3. **Compare Results Side-by-Side**:
   - Preview original and processed clips side-by-side or in a single synced view.
   - Zoom in up to 1000% using your mouse wheel and drag to inspect fine details.
   - Scrub the synchronized timeline or use the step buttons to compare specific frames.

4. **Batch Processing (Multiple Videos)**:
   - Select multiple files from the input list using **Ctrl + Click** or **Shift + Click**.
   - Open the **Render Queue** panel at the bottom left to view all queued tasks.

5. **Render and Save**:
   - Click **Render Video**. A live progress bar shows progress percentage, elapsed time, and estimated time remaining.
   - Completed videos are saved in the `output/` folder and can be opened or played immediately.

6. **Recent Activity**:
   - Click the **Bell icon** in the upper header to view recent jobs, completion times, and direct file path links.

---

## Easy Mask: Subject and Background Separation

Easy Mask allows you to isolate a moving person or object from a video without manual frame-by-frame rotoscoping:

1. Open the **Easy Mask** tab and select a video.
2. Click **Select Target on Frame 1**.
3. Use **Auto-Detect** to click on the subject you want to isolate, or use the **Add** (+) and **Erase** (-) brush tools to paint or adjust the selection.
4. Select your preferred output formats:
   - **Green Screen**: Subject placed over a pure green chroma background.
   - **Alpha Matte**: High-contrast black-and-white mask for video editors.
   - **Transparent Video**: Transparent video file (ProRes 4444 `.mov`) ready for drag-and-drop into editing software like Premiere, DaVinci Resolve, or Final Cut.
5. Click **Render Easy Mask** to process the entire clip.

---

## Hardware Support

Clarity runs on virtually any modern computer and automatically selects the fastest acceleration available:

- **NVIDIA GeForce / RTX GPUs**: Runs with dedicated TensorRT acceleration and CUDA for maximum performance. Older GTX cards and modern RTX 50-series cards are supported automatically.
- **Apple Silicon (M1, M2, M3, M4)**: Accelerated using Apple Metal (MPS).
- **AMD & Intel GPUs**: Supported via Vulkan (ncnn).
- **CPU Mode**: Runs on standard processors if a dedicated graphics card is not present (processing will take longer).

---

## Command Line Interface (Advanced)

For users who prefer working from a terminal, Clarity also includes an interactive keyboard-driven command-line interface:

```bash
uv run main.py --cli
```

- Navigate options with arrow keys and press Enter to select.
- Run automated benchmarks with `uv run main.py benchmark`.

---

## Frequently Asked Questions

<details>
<summary><b>Where are my converted videos saved?</b></summary>

All completed videos are saved in the `output/` folder inside the Clarity directory by default. You can change this destination anytime by clicking the folder path under **Output Destination** in the studio.
</details>

<details>
<summary><b>Do I need an active internet connection to use Clarity?</b></summary>

An internet connection is only needed once when an AI model is used for the very first time so Clarity can download the model weights. After that, Clarity runs completely offline.
</details>

<details>
<summary><b>How do updates work?</b></summary>

If you installed Clarity using the desktop installer, the app will periodically check for updates. When an update is available, you will receive an in-app notification. You can review the update notes and install it with a single click, keeping your downloaded models and settings intact.
</details>

<details>
<summary><b>How do I repair or re-configure my installation?</b></summary>

- On Windows, run `run.bat --setup`.
- On macOS or Linux, run `bash run.sh --setup`.
</details>

---

## License and Acknowledgements

- **Clarity Core Application**: Released under the [MIT License](LICENSE).
- **AI Models and Components**: Third-party models are subject to their respective licenses (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
  - [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) by bilibili AI Lab.
  - [AMT](https://github.com/mcg-nku/amt) by MCG-NKU (Non-Commercial use).
  - [GMFSS / MultiPassDedup / RIFE](https://github.com/AlexWortega/MultiPassDedup).
  - [MatAnyone2](https://github.com/pq-yang/MatAnyone2) by pq-yang / S-Lab (Non-Commercial use).
