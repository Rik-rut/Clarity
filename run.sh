#!/usr/bin/env bash
# Clarity launcher (Linux / macOS).
set -u

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found."
    echo "Please run 'bash setup.sh' first, then start Clarity again."
    exit 1
fi

uv run --all-extras main.py

echo
echo "Clarity closed."
