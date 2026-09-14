"""Progress parsing and a thread-safe log buffer for the setup wizard.

``parse_uv_progress_line`` is a faithful port of the shell's original parser so
uv output renders identically to the previous implementation: a percentage in
0..100 unlocks a progress update, an optional transfer speed rides along.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

# Overall progress window allotted to each step (percent of the whole wizard).
STEP_RANGES: Dict[str, Tuple[float, float]] = {
    "gpu": (0.0, 5.0),
    "runtime": (5.0, 70.0),
    "models": (70.0, 92.0),
    "verify": (92.0, 98.0),
    "complete": (98.0, 100.0),
}

MAX_LOG_LINES = 400
SNAPSHOT_LOG_LINES = 200

_STRIP_CHARS = "[](){},"


def _as_float(text: str) -> Optional[float]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):  # reject NaN/inf
        return None
    return value


def parse_uv_progress_line(line: str) -> Optional[Tuple[float, str]]:
    """Return ``(percent, speed)`` for a uv progress line, or ``None``.

    The first token that is a bare percentage wins; the speed is either glued to
    its unit (``24.5MB/s``) or split across two tokens (``18.2 MB/s``).
    """
    trimmed = line.strip()
    if not trimmed:
        return None

    words = trimmed.split()
    percent: Optional[float] = None
    speed = ""

    for word in words:
        clean = word.strip(_STRIP_CHARS)
        if not clean.endswith("%"):
            continue
        value = _as_float(clean[:-1])
        if value is not None and 0.0 <= value <= 100.0:
            percent = value
            break

    if percent is None:
        return None

    for index, word in enumerate(words):
        clean = word.strip(_STRIP_CHARS)
        if not clean.lower().endswith("/s"):
            continue
        prefix = clean[:-2]
        unit = prefix.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if unit and _as_float(unit) is not None:
            speed = clean
            break
        if index > 0:
            previous = words[index - 1].strip(_STRIP_CHARS)
            if _as_float(previous) is not None:
                speed = f"{previous} {clean}"
                break

    return percent, speed


def _compact(line: str, limit: int = 160) -> str:
    text = " ".join(line.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ProgressTracker:
    """Monotonic overall progress plus a bounded log tail, safe across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: Deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._step = ""
        self._phase = ""
        self._message = ""
        self._speed = ""
        self._percent = 0.0
        self._floor = 0.0
        self._running = False
        self._error: Optional[str] = None
        self._started = 0.0
        self._total = 0

    def _emit(self, text: str) -> None:
        """Append one log line and count it, so clients can fetch only new lines."""
        self._lines.append(text)
        self._total += 1

    # -- lifecycle ---------------------------------------------------------
    def start(self, step: str, message: Optional[str] = None) -> None:
        with self._lock:
            low, _high = STEP_RANGES.get(step, (self._percent, 100.0))
            self._step = step
            self._phase = step
            self._floor = low
            self._percent = max(self._percent, low)
            self._running = True
            self._error = None
            self._speed = ""
            self._started = time.monotonic()
            self._message = message or ""
            if self._message:
                self._emit(_compact(self._message))

    def phase(self, name: str) -> None:
        """Label the current sub-phase (e.g. venv vs dependencies) for the UI."""
        with self._lock:
            self._phase = name or self._step

    def line(self, text: str) -> None:
        if not text:
            return
        parsed = parse_uv_progress_line(text)
        with self._lock:
            self._emit(_compact(text))
            if parsed is None:
                self._message = _compact(text)
                return
            percent, speed = parsed
            low, high = STEP_RANGES.get(self._step, (0.0, 100.0))
            target = low + (high - low) * (percent / 100.0)
            if target >= self._percent:
                self._percent = min(target, high)
            if speed:
                self._speed = speed
            self._message = _compact(text)

    def finish(self, error: Optional[str] = None) -> None:
        with self._lock:
            self._running = False
            self._speed = ""
            if error:
                self._error = error
                self._message = error
                self._emit(_compact(f"[error] {error}"))
                return
            _low, high = STEP_RANGES.get(self._step, (self._percent, 100.0))
            self._percent = max(self._percent, high)

    def rewind(self, step: str) -> None:
        """Drop the monotonic floor so a retried step can move the bar back."""
        with self._lock:
            low, _high = STEP_RANGES.get(step, (0.0, 100.0))
            self._percent = low
            self._floor = low
            self._error = None
            self._speed = ""

    # -- reads -------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "step": self._step,
                "phase": self._phase or self._step,
                "percent": round(self._percent, 1),
                "speed": self._speed,
                "message": self._message,
                "lines": list(self._lines)[-SNAPSHOT_LOG_LINES:],
                "total": self._total,
                "running": self._running,
                "error": self._error,
                "elapsed": round(time.monotonic() - self._started, 1)
                if self._started
                else 0.0,
            }

    @property
    def running(self) -> bool:
        return self._running
