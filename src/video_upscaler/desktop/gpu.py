"""NVIDIA detection and hardware sync for Clarity (stdlib only).

Probes GPU model and NVIDIA compute capability to select the compatible
PyTorch build (cu130, cu126, or cpu), manages .hardware_stamp in virtual
environments, and verifies hardware state on application startup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

CUDA_130_VARIANT = "cu130"
CUDA_126_VARIANT = "cu126"
CUDA_VARIANT = "cu130"
CPU_VARIANT = "cpu"

PYTORCH_INDEX_URLS = {
    CUDA_130_VARIANT: "https://download.pytorch.org/whl/cu130",
    CUDA_126_VARIANT: "https://download.pytorch.org/whl/cu126",
    CPU_VARIANT: "https://download.pytorch.org/whl/cpu",
}

_GPU_LINE = re.compile(r"^GPU\s+\d+\s*:\s*(.+?)\s*$", re.IGNORECASE)

_LEGACY_PATTERNS = (
    re.compile(r"\bGTX\s+10\d{2}\b", re.IGNORECASE),
    re.compile(r"\bGTX\s+[789]\d{2}\b", re.IGNORECASE),
    re.compile(r"\bGT\s+[789]\d{2}\b", re.IGNORECASE),
    re.compile(r"\bTITAN\s+(X|Black|Z)\b", re.IGNORECASE),
    re.compile(r"\bQuadro\s+[KMP]\d+\b", re.IGNORECASE),
    re.compile(r"\bTesla\s+[KMP]\d+\b", re.IGNORECASE),
)

_NVIDIA_SMI_CANDIDATES: tuple[str, ...] = (
    "nvidia-smi",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)


def parse_compute_cap(cap_str: str) -> Optional[Tuple[int, int]]:
    """Parse 'major.minor' compute capability string (e.g. '12.0', '7.5', '6.1')."""
    if not cap_str:
        return None
    cleaned = cap_str.strip()
    if "." not in cleaned:
        return None
    parts = cleaned.split(".", 1)
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def is_legacy_nvidia_name(name: str) -> bool:
    """Check if GPU name is a legacy Maxwell/Pascal/Kepler architecture (< 7.5).

    GTX 16-series (Turing, CC 7.5) and RTX cards (CC >= 7.5) return False.
    """
    if not name:
        return False
    # Explicitly exclude Turing GTX 16-series (e.g. GTX 1650, 1660)
    if re.search(r"\bGTX\s+16\d{2}\b", name, re.IGNORECASE):
        return False
    return any(pattern.search(name) for pattern in _LEGACY_PATTERNS)


@dataclass
class GpuInfo:
    has_nvidia: bool = False
    names: List[str] = field(default_factory=list)
    compute_capabilities: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def torch_variant(self) -> str:
        if not self.has_nvidia:
            return CPU_VARIANT
        if self.compute_capabilities:
            parsed = parse_compute_cap(self.compute_capabilities[0])
            if parsed is not None:
                return CUDA_130_VARIANT if parsed >= (7, 5) else CUDA_126_VARIANT
        if self.names and is_legacy_nvidia_name(self.names[0]):
            return CUDA_126_VARIANT
        return CUDA_130_VARIANT

    def to_dict(self) -> dict:
        return {
            "has_nvidia": self.has_nvidia,
            "names": list(self.names),
            "compute_capabilities": list(self.compute_capabilities),
            "torch_variant": self.torch_variant,
            "error": self.error,
        }


def parse_nvidia_smi_query_output(output: str) -> Tuple[List[str], List[str]]:
    """Parse output from ``nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader``.

    Returns ``(names, compute_capabilities)``.
    """
    names: List[str] = []
    caps: List[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("cuda error", "couldn't", "n/a")):
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if not parts or not parts[0]:
            continue
        names.append(parts[0])
        caps.append(parts[1] if len(parts) > 1 else "")
    return names, caps


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


def _run_smi_cmd(cmd_args: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Execute nvidia-smi with given args, trying candidate locations."""
    startupinfo, flags = _windows_hide_flags()
    for candidate in _NVIDIA_SMI_CANDIDATES:
        executable = _resolve(candidate)
        if not executable:
            continue
        try:
            completed = subprocess.run(
                [executable, *cmd_args],
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


def run_nvidia_smi(args: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(stdout, error)`` for the first ``nvidia-smi`` that succeeds.

    If ``args`` is None, defaults to ``["-L"]`` for backwards compatibility.
    """
    return _run_smi_cmd(args if args is not None else ["-L"])


def detect_gpu() -> GpuInfo:
    """Detect GPU hardware and compute capability using nvidia-smi.

    Tries ``--query-gpu=name,compute_cap`` first, falling back to ``-L``.
    """
    query_out, query_err = _run_smi_cmd([
        "--query-gpu=name,compute_cap",
        "--format=csv,noheader",
    ])
    if query_out:
        names, caps = parse_nvidia_smi_query_output(query_out)
        if names:
            return GpuInfo(
                has_nvidia=True,
                names=names,
                compute_capabilities=caps,
                error=None,
            )

    # Fallback to -L for legacy drivers
    l_out, l_err = _run_smi_cmd(["-L"])
    if l_out:
        names = parse_nvidia_smi_output(l_out)
        if names:
            return GpuInfo(
                has_nvidia=True,
                names=names,
                compute_capabilities=[],
                error=None,
            )

    return GpuInfo(
        has_nvidia=False,
        names=[],
        compute_capabilities=[],
        error=query_err or l_err or "No NVIDIA GPU detected",
    )


def find_venv_python(venv_dir: Path) -> Optional[Path]:
    """Find the python binary inside a virtual environment directory."""
    if os.name == "nt":
        candidate = venv_dir / "Scripts" / "python.exe"
    else:
        candidate = venv_dir / "bin" / "python"
    return candidate if candidate.is_file() else None


def read_hardware_stamp(venv_dir: Path) -> Optional[dict]:
    """Read .hardware_stamp from a virtual environment directory."""
    stamp_path = venv_dir / ".hardware_stamp"
    if not stamp_path.is_file():
        return None
    try:
        return json.loads(stamp_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_hardware_stamp(venv_dir: Path, info: GpuInfo) -> None:
    """Write current GPU info to .hardware_stamp in the virtual environment."""
    stamp_path = venv_dir / ".hardware_stamp"
    stamp_path.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")


def find_uv() -> str:
    """Find the uv executable in PATH or standard user directories."""
    path = shutil.which("uv")
    if path:
        return path
    cargo_uv = Path.home() / ".cargo" / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    if cargo_uv.is_file():
        return str(cargo_uv)
    return "uv"


def check_venv(venv_dir: Path) -> int:
    """Verify that a virtual environment matches the active hardware and passes CUDA smoke test.

    Returns:
        0: Hardware matches and CUDA is usable (or CPU variant)
        1: Hardware stamp mismatch or stamp missing
        2: CUDA kernel unusable (sm_XX mismatch / cudaErrorNoKernelImageForDevice)
    """
    python_exe = find_venv_python(venv_dir)
    if not python_exe:
        print(f"[clarity] Virtualenv python not found in {venv_dir}")
        return 1

    stamp = read_hardware_stamp(venv_dir)
    if stamp is None:
        print("[clarity] No .hardware_stamp found in environment.")
        return 1

    info = detect_gpu()
    stamp_variant = stamp.get("torch_variant")
    if stamp_variant != info.torch_variant:
        print(
            f"[clarity] Hardware mismatch: stamp has {stamp_variant}, "
            f"detected {info.torch_variant}."
        )
        return 1

    stamp_names = stamp.get("names", [])
    if stamp_names != info.names:
        print(
            f"[clarity] GPU hardware changed: stamp has {stamp_names}, "
            f"detected {info.names}."
        )
        return 1

    if info.torch_variant != CPU_VARIANT:
        startupinfo, flags = _windows_hide_flags()
        try:
            res = subprocess.run(
                [
                    str(python_exe),
                    "-c",
                    "from video_upscaler.backend import cuda_usable; exit(0 if cuda_usable() else 2)",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                startupinfo=startupinfo,
                creationflags=flags,
                check=False,
            )
            if res.returncode != 0:
                print(
                    f"[clarity] CUDA kernel smoke test failed in {venv_dir} "
                    f"(exit code {res.returncode}): {res.stderr.strip() or res.stdout.strip()}"
                )
                return 2
        except Exception as exc:
            print(f"[clarity] Error testing CUDA usability in {venv_dir}: {exc}")
            return 2

    return 0


def sync_venv(venv_dir: Path) -> int:
    """Install the matching PyTorch wheel and Vulkan tools for the active hardware.

    Returns:
        0 on success
        1 on package installation failure
        2 on post-install CUDA kernel failure
    """
    python_exe = find_venv_python(venv_dir)
    if not python_exe:
        print(f"[clarity] Virtualenv python not found in {venv_dir}")
        return 1

    info = detect_gpu()
    variant = info.torch_variant
    index_url = PYTORCH_INDEX_URLS.get(variant, PYTORCH_INDEX_URLS[CPU_VARIANT])
    print(f"[clarity] Syncing environment for GPU: {info.names or 'None'} (variant: {variant})")

    uv_bin = find_uv()
    cmd = [
        uv_bin,
        "pip",
        "install",
        "--python",
        str(python_exe),
        "--reinstall-package",
        "torch",
        "--reinstall-package",
        "torchvision",
        "torch",
        "torchvision",
        "--index-url",
        index_url,
    ]

    try:
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            print(f"[clarity] Failed to install PyTorch wheel for {variant} (exit code {res.returncode})")
            return 1
    except Exception as exc:
        print(f"[clarity] Failed to invoke uv: {exc}")
        return 1

    # For non-NVIDIA (CPU variant), auto-download Real-CUGAN ncnn Vulkan binary
    if variant == CPU_VARIANT:
        print("[clarity] Non-NVIDIA hardware detected; staging realcugan-ncnn-vulkan...")
        try:
            subprocess.run(
                [
                    str(python_exe),
                    "-c",
                    (
                        "from video_upscaler import ncnn; "
                        "ncnn.download_ncnn('realcugan') if not ncnn.check_ncnn('realcugan') else None"
                    ),
                ],
                timeout=120,
                check=False,
            )
        except Exception as exc:
            print(f"[clarity] Warning: Failed to stage realcugan-ncnn-vulkan: {exc}")

    # For NVIDIA, verify CUDA kernel usability post-installation
    if variant != CPU_VARIANT:
        try:
            res = subprocess.run(
                [
                    str(python_exe),
                    "-c",
                    "from video_upscaler.backend import cuda_usable; exit(0 if cuda_usable(reset_cache=True) else 1)",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if res.returncode != 0:
                print("[clarity] Warning: Post-install CUDA smoke test failed. Device may lack kernel image.")
                return 2
        except Exception as exc:
            print(f"[clarity] Warning: Could not verify CUDA usability: {exc}")
            return 2

    write_hardware_stamp(venv_dir, info)
    print(f"[clarity] Environment synced successfully for {variant}.")
    return 0


def verify_state(data_dir: Path) -> int:
    """Verify that setup.json matches the active hardware and CUDA is usable.

    Used by the Tauri desktop boot sequence before skipping provisioning.
    Returns:
        0: Hardware matches and CUDA is usable (or CPU)
        1: Hardware changed or setup.json missing
        2: CUDA kernel unusable
    """
    setup_json = data_dir / "setup.json"
    if not setup_json.is_file():
        print(f"[clarity] setup.json not found in {data_dir}")
        return 1

    try:
        state = json.loads(setup_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[clarity] Failed to parse setup.json: {exc}")
        return 1

    info = detect_gpu()
    prov_variant = state.get("torch_variant")
    prov_names = state.get("gpu_names", [])
    prov_caps = state.get("compute_capabilities", [])

    if prov_variant != info.torch_variant:
        print(f"[clarity] Hardware variant mismatch: provisioned {prov_variant}, detected {info.torch_variant}")
        return 1

    if prov_names != info.names:
        print(f"[clarity] GPU name mismatch: provisioned {prov_names}, detected {info.names}")
        return 1

    if info.compute_capabilities and prov_caps != info.compute_capabilities:
        print(f"[clarity] GPU capability mismatch: provisioned {prov_caps}, detected {info.compute_capabilities}")
        return 1

    env_dir = data_dir / "env"
    python_exe = find_venv_python(env_dir)
    if not python_exe:
        print(f"[clarity] Desktop env python not found in {env_dir}")
        return 1

    if info.torch_variant != CPU_VARIANT:
        startupinfo, flags = _windows_hide_flags()
        try:
            res = subprocess.run(
                [
                    str(python_exe),
                    "-c",
                    "from video_upscaler.backend import cuda_usable; exit(0 if cuda_usable() else 2)",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                startupinfo=startupinfo,
                creationflags=flags,
                check=False,
            )
            if res.returncode != 0:
                print(f"[clarity] Desktop env CUDA kernel check failed (exit code {res.returncode})")
                return 2
        except Exception as exc:
            print(f"[clarity] Desktop env CUDA test error: {exc}")
            return 2

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Clarity GPU detection and hardware sync helper")
    parser.add_argument(
        "--check-venv",
        type=Path,
        metavar="VENV_DIR",
        help="Check if venv matches active hardware and passes CUDA smoke test",
    )
    parser.add_argument(
        "--sync-venv",
        type=Path,
        metavar="VENV_DIR",
        help="Sync venv PyTorch variant with active hardware and write .hardware_stamp",
    )
    parser.add_argument(
        "--verify-state",
        type=Path,
        metavar="DATA_DIR",
        help="Check if setup.json in DATA_DIR matches active hardware",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print detected GPU info as JSON",
    )

    args = parser.parse_args(argv)

    if args.check_venv:
        return check_venv(args.check_venv)

    if args.sync_venv:
        return sync_venv(args.sync_venv)

    if args.verify_state:
        return verify_state(args.verify_state)

    info = detect_gpu()
    if args.json:
        print(json.dumps(info.to_dict(), indent=2))
    else:
        print(f"NVIDIA: {info.has_nvidia}")
        print(f"GPUs: {', '.join(info.names) if info.names else 'None'}")
        print(f"Capabilities: {', '.join(info.compute_capabilities) if info.compute_capabilities else 'None'}")
        print(f"Variant: {info.torch_variant}")
        if info.error:
            print(f"Error: {info.error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
