@echo off
setlocal
cd /d "%~dp0"
title Clarity

where uv >nul 2>nul
if not %errorlevel%==0 (
    echo uv was not found.
    echo Please run setup.bat first, then start Clarity again.
    echo.
    pause
    exit /b 1
)

uv run --all-extras main.py

echo.
echo Clarity closed.
pause
