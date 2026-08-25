"""Thread-safe background job execution and progress broadcasting."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import WebSocket

from video_upscaler import config
from video_upscaler.processor import (
    format_duration,
    process_videos,
    process_interpolate,
    process_dedup,
)
from video_upscaler.matanyone2.processor import process_matanyone2
from video_upscaler.models import (
    model_for_profile,
    description_for_profile,
    description_for_interp_model,
)

logger = logging.getLogger("clarity.jobs")

_MAX_REASON_CHARS = 400


def _format_failed(failed: list[tuple[str, str]]) -> str:
    """Human-readable per-file failure summary for the job UI.

    Includes the first line of each reason so users can act on the actual
    cause (missing module, bad mask, encoder error) instead of just names.
    Full reasons stay in the server console log.
    """
    parts = []
    for name, reason in failed:
        reason_line = (reason or "unknown error").strip().splitlines()[0]
        if len(reason_line) > _MAX_REASON_CHARS:
            reason_line = reason_line[:_MAX_REASON_CHARS] + "…"
        logger.error("Job file failed: %s — %s", name, reason)
        parts.append(f"{name}: {reason_line}")
    return "Failed: " + " | ".join(parts)


@dataclass
class JobInfo:
    job_id: str
    action: str
    video_names: List[str]
    params: Dict[str, Any]
    output_dir: str
    status: str = "pending"
    current_file_index: int = 0
    total_files: int = 0
    current_file_name: str = ""
    percent: int = 0
    stage: str = "Initializing"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None
    output_files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    cancel_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        elapsed = self.elapsed_seconds

        if self.status == "running" and self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
        return {
            "job_id": self.job_id,
            "action": self.action,
            "video_names": self.video_names,
            "params": self.params,
            "output_dir": self.output_dir,
            "status": self.status,
            "current_file_index": self.current_file_index,
            "total_files": self.total_files,
            "current_file_name": self.current_file_name,
            "percent": self.percent,
            "stage": self.stage,
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_formatted": format_duration(elapsed),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds is not None else None,
            "eta_formatted": format_duration(self.eta_seconds) if self.eta_seconds is not None else "--",
            "output_files": self.output_files,
            "error_message": self.error_message,
        }


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = threading.Lock()


    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        with self._lock:
            self.active_connections.add(websocket)


    def disconnect(self, websocket: WebSocket) -> None:
        with self._lock:
            self.active_connections.discard(websocket)


    async def broadcast_json(self, data: dict) -> None:
        with self._lock:
            connections = list(self.active_connections)
        for connection in connections:
            try:
                await connection.send_json(data)
            except Exception:
                with self._lock:
                    self.active_connections.discard(connection)


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, JobInfo] = {}
        self.active_job_id: Optional[str] = None
        self.connection_manager = ConnectionManager()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None


    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop


    def broadcast_sync(self, data: dict) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.connection_manager.broadcast_json(data), self._loop
            )


    def get_job(self, job_id: str) -> Optional[JobInfo]:
        with self._lock:
            return self.jobs.get(job_id)


    def get_active_job(self) -> Optional[JobInfo]:
        with self._lock:
            if self.active_job_id:
                return self.jobs.get(self.active_job_id)
            return None


    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            job = self.jobs.get(job_id)
            if not job or job.status not in ("pending", "running"):
                return False
            job.cancel_requested = True
            job.status = "cancelled"
            job.stage = "Cancelled by user"
            if self.active_job_id == job_id:
                self.active_job_id = None
        self.broadcast_sync({"type": "job_cancelled", "job": job.to_dict()})
        return True


    def submit_job(
        self,
        action: str,
        video_paths: List[Path],
        params: Dict[str, Any],
        output_dir: Path,
    ) -> JobInfo:
        with self._lock:
            if self.active_job_id:
                active = self.jobs.get(self.active_job_id)
                if active and active.status in ("pending", "running"):
                    raise RuntimeError("A render job is already running.")

            job_id = str(uuid.uuid4())[:8]
            job = JobInfo(
                job_id=job_id,
                action=action,
                video_names=[p.name for p in video_paths],
                params=params,
                output_dir=str(output_dir),
                total_files=len(video_paths),
                status="pending",
                stage="Queued",
            )
            self.jobs[job_id] = job
            self.active_job_id = job_id


        worker = threading.Thread(
            target=self._run_job_thread,
            args=(job_id, video_paths, params, output_dir),
            daemon=True,
        )
        worker.start()
        return job


    def _run_job_thread(
        self,
        job_id: str,
        video_paths: List[Path],
        params: Dict[str, Any],
        output_dir: Path,
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return


        job.status = "running"
        job.start_time = time.perf_counter()
        job.stage = f"Starting {job.action}..."
        self.broadcast_sync({"type": "job_started", "job": job.to_dict()})


        orig_output_dir = config.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_DIR = output_dir


        file_eta_start = time.perf_counter()
        last_file_idx = 0


        def progress_callback(file_idx: int, file_count: int, percent: int) -> None:
            nonlocal file_eta_start, last_file_idx
            if job.cancel_requested:
                raise RuntimeError("Job cancelled by user")

            job.current_file_index = file_idx
            job.total_files = file_count
            if 1 <= file_idx <= len(video_paths):
                job.current_file_name = video_paths[file_idx - 1].name
            job.percent = percent

            if file_idx != last_file_idx:
                file_eta_start = time.perf_counter()
                last_file_idx = file_idx
                job.stage = f"Processing [{file_idx}/{file_count}]: {job.current_file_name}"


            now = time.perf_counter()
            elapsed_for_file = now - file_eta_start
            if percent > 2 and percent < 100:
                rem_file = (elapsed_for_file / percent) * (100 - percent)
                rem_files_time = (
                    (file_count - file_idx)
                    * (elapsed_for_file / (percent / 100.0))
                    if percent > 10
                    else (file_count - file_idx) * elapsed_for_file
                )
                job.eta_seconds = rem_file + rem_files_time
            elif percent >= 100:
                job.eta_seconds = 0.0


            job.elapsed_seconds = now - (job.start_time or now)
            self.broadcast_sync({"type": "job_progress", "job": job.to_dict()})


        try:
            if job.action == "Upscale":
                profile = params.get("profile", "2x_Balanced")
                results = process_videos(video_paths, profile, progress_callback)
            elif job.action == "Slow-motion":
                model_key = params.get("model_key", "AMT-S")
                factor = int(params.get("factor", 2))
                results = process_interpolate(
                    video_paths, model_key, factor, progress_callback
                )
            elif job.action == "Interpolate":
                model = params.get("model", "gmfss")
                npass = int(params.get("npass", 0))
                factor = int(params.get("factor", 2))
                results = process_dedup(
                    video_paths, model, npass, factor, progress_callback
                )
            elif job.action == "MatAnyone2":
                if not params.get("mask_png"):
                    raise ValueError("MatAnyone2 requires a first-frame mask.")

                def stage_reporter(stage_text: str) -> None:
                    job.stage = stage_text

                results = process_matanyone2(
                    video_paths, params, progress_callback, stage_reporter
                )
            else:
                raise ValueError(f"Unknown action: {job.action}")

            job.end_time = time.perf_counter()
            job.elapsed_seconds = job.end_time - (job.start_time or job.end_time)
            job.output_files = [str(p) for p in results.get("success", [])]

            if results.get("failed"):
                job.error_message = _format_failed(results["failed"])
                if not results.get("success"):
                    job.status = "failed"
                else:
                    job.status = "completed"
            else:
                job.status = "completed"
                job.percent = 100
                job.stage = "Processing complete!"

        except Exception as exc:
            job.status = "failed" if not job.cancel_requested else "cancelled"
            job.error_message = str(exc)
            job.stage = f"Error: {str(exc).splitlines()[0] if str(exc).strip() else type(exc).__name__}"
            logger.exception("Job %s (%s) failed", job_id, job.action)
        finally:
            config.OUTPUT_DIR = orig_output_dir
            with self._lock:
                if self.active_job_id == job_id:
                    self.active_job_id = None
            try:
                from video_upscaler.memory import free_gpu_memory

                free_gpu_memory()
            except Exception as cleanup_exc:
                logger.debug("Post-job GPU memory cleanup error: %s", cleanup_exc)
            self.broadcast_sync({"type": "job_completed", "job": job.to_dict()})


job_manager = JobManager()
