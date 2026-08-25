@echo off
setlocal
cd /d "%~dp0"
title Clarity Setup

echo ==========================================
echo            CLARITY  -  SETUP
echo        One-time installation assistant
echo ==========================================
echo.
echo This will prepare everything Clarity needs:
echo   1. uv          (Python environment manager)
echo   2. Libraries   (PyTorch and friends)
echo   3. FFmpeg      (video decoder/encoder)
echo   4. GPU booster (TensorRT, only if you have an NVIDIA GPU)
echo   5. AI models   (optional pre-download)
echo.

REM ---- [1/5] uv -----------------------------------------------------------
where uv >nul 2>nul
if %errorlevel%==0 (
    echo [1/5] uv is already installed.
    goto sync
)
echo [1/5] Installing uv (Python environment manager)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if not exist "%USERPROFILE%\.local\bin\uv.exe" (
    echo.
    echo   uv did not appear after installation.
    echo   Please close this window, open a NEW terminal, and run Setup.bat again.
    goto fail
)
set "PATH=%PATH%;%USERPROFILE%\.local\bin"

:sync
echo.
echo [2/5] Installing libraries (first time only - this downloads a few GB)...
uv sync --all-extras
if errorlevel 1 goto fail

REM ---- [3/5] FFmpeg -------------------------------------------------------
echo.
where ffmpeg >nul 2>nul
if %errorlevel%==0 (
    echo [3/5] FFmpeg is already installed.
    goto gpu_check
)
echo [3/5] FFmpeg is required but was not found.
choice /C YN /M "      Try installing FFmpeg automatically"
if errorlevel 2 (
    echo.
    echo      Install it later from https://www.gyan.dev/ffmpeg/builds/
    echo      and make sure ffmpeg.exe is on your PATH.
    goto gpu_check
)
where winget >nul 2>nul
if not %errorlevel%==0 (
    echo.
    echo      winget is unavailable on this PC.
    echo      Install FFmpeg manually from https://www.gyan.dev/ffmpeg/builds/
    goto gpu_check
)
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
where ffmpeg >nul 2>nul
if not %errorlevel%==0 (
    echo.
    echo      FFmpeg was installed but is not on PATH yet.
    echo      Close this window, open a NEW terminal, and run Setup.bat again.
)

REM ---- [4/5] TensorRT (NVIDIA only) ---------------------------------------
:gpu_check
echo.
nvidia-smi -L >nul 2>nul
if not %errorlevel%==0 (
    echo [4/5] No NVIDIA GPU detected - skipping the TensorRT speed booster.
    goto models_menu
)
echo [4/5] NVIDIA GPU detected.
echo       TensorRT makes upscaling several times faster (~3 GB download).
choice /C YN /M "      Install the TensorRT booster now"
if errorlevel 2 (
    echo      Skipping TensorRT. Clarity will still use your GPU via CUDA.
    goto models_menu
)
uv sync --all-extras
if errorlevel 1 (
    echo.
    echo      TensorRT install failed - continuing without it.
    echo      Clarity will still use your GPU via CUDA.
)

REM ---- [5/5] Models -------------------------------------------------------
:models_menu
echo.
echo [5/5] AI models are downloaded on first use. You can also get them now:
echo       1 Essential set  (~75 MB, enough for upscaling + basic slow-mo)
echo       2 Everything     (~540 MB, all upscale/slow-mo/interpolation models)
echo       3 Skip           (Clarity will ask when it first needs one)
choice /C 123 /N /M "      Choose [1], [2] or [3]"
if errorlevel 3 goto done
if errorlevel 2 (
    uv run --all-extras main.py --download-models all
    goto done
)
if errorlevel 1 (
    uv run --all-extras main.py --download-models essential
    goto done
)

:done
echo.
echo ==========================================
echo               SETUP COMPLETE
echo ==========================================
echo.
echo   How to use Clarity:
echo     1. Copy video files into the  input  folder
echo     2. Double-click  run.bat
echo     3. Follow the menus - results appear in  output
echo.
pause
exit /b 0

:fail
echo.
echo ==========================================
echo              SETUP FAILED
echo ==========================================
echo Check the messages above, fix the issue,
echo then run Setup.bat again.
echo.
pause
exit /b 1
