"""NVIDIA detection for the desktop wizard (stdlib only).

Mirrors the shell's ``nvidia-smi -L`` probe so both sides pick the same torch
build: CUDA wheels for NVIDIA hardware, PyPI CPU wheels otherwise.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

CUDA_VARIANT = "cu126"
CPU_VARIANT = "cpu"

_GPU_LINE = re.compile(r"^GPU\s+\d+\s*:\s*(.+?)\s*$", re.IGNORECASE)

_NVIDIA_SMI_CANDIDATES: tuple[str, ...] = (
    "nvidia-smi",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)


@dataclass
class GpuInfo:
    has_nvidia: bool = False
    names: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def torch_variant(self) -> str:
        return CUDA_VARIANT if self.has_nvidia else CPU_VARIANT

    def to_dict(self) -> dict:
        return {
            "has_nvidia": self.has_nvidia,
            "names": list(self.names),
            "torch_variant": self.torch_variant,
            "error": self.error,
        }


def parse_nvidia_smi_output(output: str) -> List[str]:
    """Device names from ``nvidia-smi -L`` output, UUID suffixes stripped."""
    names: List[str] = []
    for line in output.splitlines():
        match = _GPU_LINE.match(line.strip())
        if not match:
            continue
        name = match.group(1)
        uuid_at = name.lower().find("(uuid:")
        if uuid_at != -1:
            name = name[:uuid_at].strip()
        # "GPU 0: NVIDIA ..." matches; a bare "CUDA error" line does not name a GPU.
        if name and not name.lower().startswith(("cuda init", "couldn't")):
            names.append(name)
    return names


def _windows_hide_flags() -> Tuple[object, int]:
    if os.name != "nt":
        return None, 0
    startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    return startupinfo, subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]


def _resolve(candidate: str) -> Optional[str]:
    if Path(candidate).is_absolute():
        return candidate if Path(candidate).is_file() else None
    return shutil.which(candidate)


def run_nvidia_smi() -> Tuple[Optional[str], Optional[str]]:
    """Return ``(stdout, error)`` for the first ``nvidia-smi`` that succeeds."""
    startupinfo, flags = _windows_hide_flags()
    for candidate in _NVIDIA_SMI_CANDIDATES:
        executable = _resolve(candidate)
        if not executable:
            continue
        try:
            completed = subprocess.run(
                [executable, "-L"],
                capture_output=True,
                text=True,
                timeout=15,
                startupinfo=startupinfo,
                creationflags=flags,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if completed.returncode == 0:
            return completed.stdout, None
    return None, "nvidia-smi not found or reported an error"


def detect_gpu() -> GpuInfo:
    output, error = run_nvidia_smi()
    if output is None:
        return GpuInfo(has_nvidia=False, names=[], error=error)
    names = parse_nvidia_smi_output(output)
    return GpuInfo(has_nvidia=bool(names), names=names, error=None if names else error)
