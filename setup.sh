#!/usr/bin/env bash
# Clarity setup — one-time installation assistant (Linux / macOS).
set -u

cd "$(dirname "$0")"

echo "=========================================="
echo "           CLARITY  -  SETUP"
echo "       One-time installation assistant"
echo "=========================================="
echo
echo "This will prepare everything Clarity needs:"
echo "  1. uv          (Python environment manager)"
echo "  2. Libraries   (PyTorch and friends)"
echo "  3. FFmpeg      (video decoder/encoder)"
echo "  4. GPU booster (TensorRT, only if you have an NVIDIA GPU)"
echo "  5. AI models   (optional pre-download)"
echo

fail() {
    echo
    echo "=========================================="
    echo "             SETUP FAILED"
    echo "=========================================="
    echo "Check the messages above, fix the issue,"
    echo "then run:  bash setup.sh"
    exit 1
}

# ---- [1/5] uv -------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    echo "[1/5] uv is already installed."
else
    echo "[1/5] Installing uv (Python environment manager)..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || fail
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh || fail
    else
        echo "  Neither curl nor wget was found. Install one and retry."
        fail
    fi
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo
        echo "  uv did not appear after installation."
        echo "  Close this terminal, open a NEW one, and run:  bash setup.sh"
        fail
    fi
fi

# ---- [2/5] Libraries ------------------------------------------------------
echo
echo "[2/5] Installing libraries (first time only - this downloads a few GB)..."
uv sync --all-extras || fail

# ---- [3/5] FFmpeg ---------------------------------------------------------
echo
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
    echo "[3/5] FFmpeg is already installed."
else
    echo "[3/5] FFmpeg is required but was not found."
    printf "      Try installing FFmpeg automatically? [y/N] "
    read -r answer
    case "$answer" in
        y|Y)
            if command -v brew >/dev/null 2>&1; then
                brew install ffmpeg || echo "  brew install failed - install FFmpeg manually from https://ffmpeg.org"
            elif command -v apt-get >/dev/null 2>&1; then
                sudo apt-get update && sudo apt-get install -y ffmpeg \
                    || echo "  apt install failed - run: sudo apt install ffmpeg"
            elif command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y ffmpeg \
                    || echo "  dnf install failed - see https://rpmfusion.org for FFmpeg on RHEL/Fedora"
            elif command -v pacman >/dev/null 2>&1; then
                sudo pacman -S --noconfirm ffmpeg \
                    || echo "  pacman install failed - run: sudo pacman -S ffmpeg"
            else
                echo "  No supported package manager found."
                echo "  Install FFmpeg manually from https://ffmpeg.org/download.html"
            fi
            ;;
        *)
            echo "      Skipping. Install FFmpeg later from https://ffmpeg.org/download.html"
            ;;
    esac
fi

# ---- [4/5] TensorRT (NVIDIA only) -----------------------------------------
echo
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "[4/5] NVIDIA GPU detected."
    echo "      TensorRT makes upscaling several times faster (~3 GB download)."
    printf "      Install the TensorRT booster now? [y/N] "
    read -r answer
    case "$answer" in
        y|Y)
            uv sync --all-extras || echo "      TensorRT install failed - continuing without it. Clarity will still use your GPU via CUDA."
            ;;
        *)
            echo "      Skipping TensorRT. Clarity will still use your GPU via CUDA."
            ;;
    esac
else
    echo "[4/5] No NVIDIA GPU detected - skipping the TensorRT speed booster."
fi

# ---- [5/5] Models ----------------------------------------------------------
chmod +x run.sh 2>/dev/null || true
echo
echo "[5/5] AI models are downloaded on first use. You can also get them now:"
echo "      1 Essential set  (~560 MB, upscaling + slow-mo + Easy Mask)"
echo "      2 Everything     (~820 MB, all models including interpolation)"
echo "      3 Skip           (Clarity will ask when it first needs one)"
printf "      Choose [1], [2] or [3]: "
read -r answer
case "$answer" in
    1) uv run --all-extras main.py --download-models essential ;;
    2) uv run --all-extras main.py --download-models all ;;
    *) echo "      Skipping model pre-download." ;;
esac

echo
echo "=========================================="
echo "              SETUP COMPLETE"
echo "=========================================="
echo
echo "  How to use Clarity:"
echo "    1. Copy video files into the  input  folder"
echo "    2. Start Clarity:   bash run.sh   (or ./run.sh)"
echo "    3. Follow the menus - results appear in  output"
echo
