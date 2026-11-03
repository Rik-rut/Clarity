"""Activity history store for recent render jobs.

Persists up to 20 recent jobs to `<DATA_DIR>/activity.json` with monotonic sequence
numbering so unread status is preserved across restarts and port changes.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from video_upscaler import config

logger = logging.getLogger("clarity.activity")

ACTIVITY_FILE_NAME = "activity.json"
STATE_VERSION = 1
MAX_ENTRIES = 20
MAX_ERROR_CHARS = 2000
ALLOWED_PARAM_KEYS = {
    "profile",
    "model",
    "factor",
    "npass",
    "outputs",
    "blend_edges",
    "scale",
}


def filter_summary(action: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract whitelisted parameters for history display, ignoring binary/mask payloads."""
    if not isinstance(params, dict):
        return {}
    summary: Dict[str, Any] = {}
    for key in ALLOWED_PARAM_KEYS:
        if key in params:
            val = params[key]
            if isinstance(val, (str, int, float, bool)):
                summary[key] = val
            elif isinstance(val, list) and all(
                isinstance(x, (str, int, float, bool)) for x in val
            ):
                summary[key] = val
    return summary


def truncate_error(msg: Optional[str]) -> Optional[str]:
    """Truncate long error messages to MAX_ERROR_CHARS with an ellipsis."""
    if not msg:
        return None
    msg_str = str(msg).strip()
    if len(msg_str) > MAX_ERROR_CHARS:
        return msg_str[:MAX_ERROR_CHARS] + "…"
    return msg_str


class ActivityStore:
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = data_dir
        self._lock = threading.Lock()

    @property
    def data_dir(self) -> Path:
        if self._data_dir is not None:
            return self._data_dir
        return config.DATA_DIR

    @property
    def file_path(self) -> Path:
        return self.data_dir / ACTIVITY_FILE_NAME

    def _default_state(self) -> Dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "next_seq": 1,
            "last_seen_seq": 0,
            "entries": [],
        }

    def _load_locked(self) -> Dict[str, Any]:
        path = self.file_path
        if not path.is_file():
            return self._default_state()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
                logger.warning(
                    "Activity file %s has invalid version or structure; resetting",
                    path,
                )
                return self._default_state()
            data.setdefault("next_seq", 1)
            data.setdefault("last_seen_seq", 0)
            if not isinstance(data.get("entries"), list):
                data["entries"] = []
            return data
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            logger.warning(
                "Failed to read activity file %s (%s); resetting", path, exc
            )
            return self._default_state()

    def _save_locked(self, data: Dict[str, Any]) -> None:
        target = self.file_path
        temp_name: Optional[str] = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", dir=target.parent, delete=False, encoding="utf-8"
            ) as tf:
                json.dump(data, tf, indent=2, ensure_ascii=False)
                tf.flush()
                os.fsync(tf.fileno())
                temp_name = tf.name
            os.replace(temp_name, target)
        except OSError as exc:
            logger.error("Failed to save activity to %s: %s", target, exc)
            if temp_name and os.path.exists(temp_name):
                try:
                    os.remove(temp_name)
                except OSError:
                    pass

    def get(self) -> Dict[str, Any]:
        with self._lock:
            data = self._load_locked()
            last_seen = data.get("last_seen_seq", 0)
            entries = data.get("entries", [])
            unread = sum(1 for e in entries if e.get("seq", 0) > last_seen)
            return {
                "entries": list(entries),
                "unread": unread,
            }

    def record(self, job: Any) -> Dict[str, Any]:
        with self._lock:
            data = self._load_locked()
            seq = data.get("next_seq", 1)
            data["next_seq"] = seq + 1

            finished_at = datetime.now(timezone.utc).isoformat()

            if hasattr(job, "to_dict"):
                job_id = job.job_id
                action = job.action
                status = job.status
                video_names = list(job.video_names)
                params = job.params
                output_dir = str(job.output_dir)
                output_files = list(job.output_files)
                error_message = job.error_message
                try:
                    elapsed_seconds = round(float(job.elapsed_seconds), 1)
                except (TypeError, ValueError):
                    elapsed_seconds = 0.0
            else:
                job_id = job.get("job_id", "")
                action = job.get("action", "")
                status = job.get("status", "completed")
                video_names = list(job.get("video_names", []))
                params = job.get("params", {})
                output_dir = str(job.get("output_dir", ""))
                output_files = list(job.get("output_files", []))
                error_message = job.get("error_message")
                try:
                    elapsed_seconds = round(
                        float(job.get("elapsed_seconds", 0.0)), 1
                    )
                except (TypeError, ValueError):
                    elapsed_seconds = 0.0

            entry = {
                "seq": seq,
                "job_id": job_id,
                "action": action,
                "status": status,
                "summary": filter_summary(action, params),
                "video_names": video_names,
                "output_dir": output_dir,
                "output_files": output_files,
                "error_message": truncate_error(error_message),
                "elapsed_seconds": elapsed_seconds,
                "finished_at": finished_at,
            }

            entries = data.get("entries", [])
            entries.insert(0, entry)
            data["entries"] = entries[:MAX_ENTRIES]
            self._save_locked(data)
            return entry

    def mark_seen(self) -> int:
        with self._lock:
            data = self._load_locked()
            entries = data.get("entries", [])
            max_seq = max((e.get("seq", 0) for e in entries), default=0)
            data["last_seen_seq"] = max(data.get("last_seen_seq", 0), max_seq)
            self._save_locked(data)
            return data["last_seen_seq"]

    def clear(self) -> None:
        with self._lock:
            data = self._load_locked()
            data["entries"] = []
            data["last_seen_seq"] = data.get("next_seq", 1) - 1
            self._save_locked(data)


_store = ActivityStore()


def get_store() -> ActivityStore:
    return _store


def record(job: Any) -> Dict[str, Any]:
    return _store.record(job)


def get_activity() -> Dict[str, Any]:
    return _store.get()


def mark_seen() -> int:
    return _store.mark_seen()


def clear() -> None:
    return _store.clear()
