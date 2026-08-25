"""REST API and WebSocket routes for Clarity Web UI."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import cv2
import numpy as np

from video_upscaler import config
from video_upscaler.backend import backend_label, detect_backend
from video_upscaler.cugan import detect_device
from video_upscaler.ffmpeg import decode_frames, probe
from video_upscaler.models import (
    INTERP_MODELS,
    PROFILES,
    description_for_interp_model,
    description_for_profile,
    scale_for_model,
)
from video_upscaler.processor import format_duration
from video_upscaler.scanner import scan_videos
from video_upscaler.web.jobs import job_manager
from video_upscaler.web.stream import stream_video_file

router = APIRouter(prefix="/api")

THUMBNAIL_CACHE_DIR = config.BASE_DIR / ".cache" / "thumbnails"
THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

class StartJobRequest(BaseModel):
    action: str
    video_names: List[str]
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    params: Dict[str, Any] = {}

class SegmentTargetRequest(BaseModel):
    path: str
    view_width: int
    view_height: int
    # Multi-click prompt (preview-image coordinates); falls back to x/y.
    points: Optional[List[List[float]]] = None
    labels: Optional[List[int]] = None
    x: Optional[float] = None
    y: Optional[float] = None

class ValidateDirRequest(BaseModel):
    path: str

class RenameVideoRequest(BaseModel):
    old_name: str
    new_name: str
    folder: Optional[str] = None

class DeleteVideoRequest(BaseModel):
    video_name: str
    folder: Optional[str] = None

class ClearVideosRequest(BaseModel):
    folder: Optional[str] = None

class BrowseDirRequest(BaseModel):
    initial_dir: Optional[str] = None

@router.get("/system/info")
def get_system_info() -> Dict[str, Any]:
    backend = detect_backend()
    device = detect_device()
    b_label = backend_label(backend)
    profiles = [
        {
            "name": name,
            "scale": scale_for_model(m_name),
            "model_file": m_name,
            "description": desc,
        }
        for name, (m_name, desc) in PROFILES.items()
    ]
    slow_mo_models = [
        {"key": k, "ckpt": ckpt, "description": desc}
        for k, (ckpt, desc) in INTERP_MODELS.items()
    ]
    dedup_models = [
        {"key": "gmfss", "name": "GMFSS (Fortuna)", "desc": "Best anime quality (default)"},
        {"key": "rife", "name": "RIFE (Practical-RIFE)", "desc": "Faster processing"},
    ]
    return {
        "backend": backend,
        "backend_label": b_label,
        "device": device,
        "input_dir": str(config.INPUT_DIR.resolve()),
        "output_dir": str(config.OUTPUT_DIR.resolve()),
        "profiles": profiles,
        "slow_mo_models": slow_mo_models,
        "dedup_models": dedup_models,
        "supported_extensions": sorted(list(config.SUPPORTED_EXTENSIONS)),
    }


@router.post("/system/reset")
def reset_system() -> Dict[str, Any]:
    """Cancel any active job, drop model singletons, and purge GPU VRAM."""
    from video_upscaler.memory import free_gpu_memory

    active = job_manager.get_active_job()
    if active:
        job_manager.cancel_job(active.job_id)

    res = free_gpu_memory()
    return {
        "success": True,
        "message": "System state reset and GPU memory cleared.",
        "vram": res.get("vram", {}),
    }


@router.get("/videos/scanned")
def get_scanned_videos(folder: Optional[str] = None) -> Dict[str, Any]:
    target_dir = Path(folder) if folder else config.INPUT_DIR
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    videos = scan_videos(target_dir)
    results = []
    for vid_path in videos:
        size_mb = float(vid_path.stat().st_size) / (1024 * 1024)
        info = {
            "name": vid_path.name,
            "path": str(vid_path.resolve()),
            "size_bytes": vid_path.stat().st_size,
            "size_formatted": f"{size_mb:.1f} MB",
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "duration": 0.0,
            "duration_formatted": "--:--",
        }
        try:
            meta = probe(vid_path)
            info["width"] = int(meta.get("width") or 0)
            info["height"] = int(meta.get("height") or 0)
            info["fps"] = round(float(meta.get("fps") or 0.0), 2)
            info["duration"] = round(float(meta.get("duration") or 0.0), 2)
            info["duration_formatted"] = format_duration(float(meta.get("duration") or 0.0))
        except Exception:
            pass
        results.append(info)
    return {"videos": results, "folder": str(target_dir.resolve())}


@router.post("/videos/upload")
async def upload_video(file: UploadFile = File(...), target_folder: Optional[str] = Form(None)) -> Dict[str, Any]:
    f_name = file.filename or "video.mp4"
    ext = Path(f_name).suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Supported: {', '.join(sorted(config.SUPPORTED_EXTENSIONS))}",
        )
    target_dir = Path(target_folder) if target_folder else config.INPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / f_name
    counter = 1
    stem = destination.stem
    while destination.exists():
        destination = target_dir / f"{stem}_{counter}{ext}"
        counter += 1
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    size_mb = float(destination.stat().st_size) / (1024 * 1024)
    info = {
        "name": destination.name,
        "path": str(destination.resolve()),
        "size_bytes": destination.stat().st_size,
        "size_formatted": f"{size_mb:.1f} MB",
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration": 0.0,
        "duration_formatted": "--:--",
    }
    try:
        meta = probe(destination)
        info["width"] = int(meta.get("width") or 0)
        info["height"] = int(meta.get("height") or 0)
        info["fps"] = round(float(meta.get("fps") or 0.0), 2)
        info["duration"] = round(float(meta.get("duration") or 0.0), 2)
        info["duration_formatted"] = format_duration(float(meta.get("duration") or 0.0))
    except Exception:
        pass
    return {"success": True, "video": info}


@router.post("/videos/rename")
def rename_video(req: RenameVideoRequest) -> Dict[str, Any]:
    target_dir = Path(req.folder) if req.folder else config.INPUT_DIR
    old_path = (target_dir / req.old_name).resolve()
    if not old_path.exists() or not old_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video '{req.old_name}' not found")

    new_name = req.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New video name cannot be empty")

    old_ext = old_path.suffix.lower()
    new_path = target_dir / new_name
    if not new_path.suffix:
        new_path = target_dir / f"{new_name}{old_ext}"

    if new_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{new_path.suffix}'. Supported: {', '.join(sorted(config.SUPPORTED_EXTENSIONS))}",
        )

    if new_path.exists() and new_path != old_path:
        raise HTTPException(status_code=409, detail=f"A file named '{new_path.name}' already exists")

    try:
        old_path.rename(new_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to rename file: {exc}")

    size_mb = float(new_path.stat().st_size) / (1024 * 1024)
    info = {
        "name": new_path.name,
        "path": str(new_path.resolve()),
        "size_bytes": new_path.stat().st_size,
        "size_formatted": f"{size_mb:.1f} MB",
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration": 0.0,
        "duration_formatted": "--:--",
    }
    try:
        meta = probe(new_path)
        info["width"] = int(meta.get("width") or 0)
        info["height"] = int(meta.get("height") or 0)
        info["fps"] = round(float(meta.get("fps") or 0.0), 2)
        info["duration"] = round(float(meta.get("duration") or 0.0), 2)
        info["duration_formatted"] = format_duration(float(meta.get("duration") or 0.0))
    except Exception:
        pass

    return {"success": True, "video": info}


@router.post("/videos/delete")
def delete_video(req: DeleteVideoRequest) -> Dict[str, Any]:
    target_dir = Path(req.folder) if req.folder else config.INPUT_DIR
    target_path = (target_dir / req.video_name).resolve()
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video '{req.video_name}' not found")
    try:
        target_path.unlink()
        return {"success": True, "deleted": req.video_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {exc}")


@router.post("/videos/clear")
def clear_videos(req: ClearVideosRequest = ClearVideosRequest()) -> Dict[str, Any]:
    target_dir = Path(req.folder) if req.folder else config.INPUT_DIR
    if not target_dir.exists():
        return {"success": True, "deleted_count": 0}
    deleted_count = 0
    for file in target_dir.iterdir():
        if file.is_file() and file.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            try:
                file.unlink()
                deleted_count += 1
            except Exception:
                pass
    return {"success": True, "deleted_count": deleted_count}


@router.post("/directories/browse")
async def browse_directory(req: BrowseDirRequest = BrowseDirRequest()) -> Dict[str, Any]:
    import asyncio

    def _run_native_picker() -> Optional[str]:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            init_dir = req.initial_dir or str(config.OUTPUT_DIR.resolve())
            path = filedialog.askdirectory(initialdir=init_dir, title="Select Output Destination")
            root.destroy()
            return path if path else None
        except Exception:
            return None

    picked_path = await asyncio.to_thread(_run_native_picker)
    if picked_path:
        return {"success": True, "path": picked_path, "cancelled": False}
    return {"success": False, "cancelled": True}


@router.get("/videos/thumbnail")
def get_thumbnail(path: str = Query(...)) -> FileResponse:
    vid_path = Path(path)
    if not vid_path.exists() or not vid_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    mtime = int(vid_path.stat().st_mtime)
    cache_file = THUMBNAIL_CACHE_DIR / f"{vid_path.stem}_{mtime}.jpg"
    if not cache_file.exists():
        import subprocess
        ffmpeg = config.ffmpeg_path()
        if not ffmpeg:
            raise HTTPException(status_code=500, detail="FFmpeg not found on system")
        cmd = [
            ffmpeg,
            "-y",
            "-ss", "00:00:01.000",
            "-i", str(vid_path.resolve()),
            "-vframes", "1",
            "-vf", "scale=320:-1",
            "-q:v", "3",
            str(cache_file.resolve()),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not cache_file.exists() or cache_file.stat().st_size == 0:
            cmd[3] = "00:00:00.000"
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return FileResponse(cache_file, media_type="image/jpeg")
    raise HTTPException(status_code=500, detail="Could not generate thumbnail")

FRAME_CACHE_DIR = config.BASE_DIR / ".cache" / "frames"


@router.get("/videos/frame")
def get_video_frame(
    path: str = Query(...),
    n: int = Query(0, ge=0),
    max: int = Query(1280, ge=64),
) -> FileResponse:
    """Full-resolution first frame for mask painting (cached by mtime)."""
    vid_path = Path(path)
    if not vid_path.exists() or not vid_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    if n != 0:
        raise HTTPException(status_code=400, detail="Only frame 0 is exposed")
    ffmpeg = config.ffmpeg_path()
    if not ffmpeg:
        raise HTTPException(status_code=500, detail="FFmpeg not found on system")
    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = FRAME_CACHE_DIR / (
        f"{vid_path.stem}_{int(vid_path.stat().st_mtime)}_{n}_{max}.jpg"
    )
    if not cache_file.exists() or cache_file.stat().st_size == 0:
        import subprocess

        cmd = [
            ffmpeg, "-y", "-i", str(vid_path.resolve()),
            "-map", "0:v:0", "-frames:v", "1",
            "-vf", f"scale='min({max},iw)':-2", "-q:v", "2",
            str(cache_file.resolve()),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return FileResponse(cache_file, media_type="image/jpeg")
    raise HTTPException(status_code=500, detail="Could not extract frame")

@router.post("/directories/validate")
def validate_directory(req: ValidateDirRequest) -> Dict[str, Any]:
    target = Path(req.path)
    try:
        target.mkdir(parents=True, exist_ok=True)
        test_file = target / ".clarity_write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        return {"valid": True, "path": str(target.resolve())}
    except Exception as exc:
        return {"valid": False, "error": str(exc), "path": req.path}

def _ensure_hub_group(group: str) -> None:
    """Install any missing manifest entries for a group (web first-use path).

    Raises modelhub.HubError when a download fails so callers can surface it.
    """
    from video_upscaler import modelhub

    missing = modelhub.missing_entries(modelhub.entries(group=group))
    for entry in missing:
        modelhub.install_entry(entry)


@router.post("/matanyone2/segment")
def segment_matanyone_target(req: SegmentTargetRequest) -> Dict[str, Any]:
    """Auto-detect the subject under one or more clicks on the first frame.

    Coordinates arrive in preview-image space; they are mapped back to
    native video geometry here so the returned mask matches the frame the
    matting job will process. SAM (same interaction model as the official
    demo) is used when its checkpoint is installed, with a GrabCut fallback
    for single clicks otherwise.
    """
    from video_upscaler.matanyone2.segment import (
        SubjectDetectionError,
        detect_subject_mask,
        mask_to_white_png_b64,
    )
    from video_upscaler.matanyone2 import sam_segment
    from video_upscaler import modelhub

    vid_path = Path(req.path)
    if not vid_path.exists() or not vid_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    if req.view_width <= 0 or req.view_height <= 0:
        raise HTTPException(status_code=400, detail="Invalid view dimensions")

    if req.points:
        raw_points = [(float(p[0]), float(p[1])) for p in req.points]
    elif req.x is not None and req.y is not None:
        raw_points = [(float(req.x), float(req.y))]
    else:
        raise HTTPException(status_code=400, detail="No click points provided")

    try:
        info = probe(vid_path)
        width, height = int(info["width"]), int(info["height"])
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read video dimensions")
        frames = decode_frames(vid_path)
        first = next(frames, None)
        if first is None:
            raise RuntimeError("Could not decode the first frame")
        frame_rgb = np.frombuffer(first, np.uint8).reshape(height, width, 3)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    scale_x = width / float(req.view_width)
    scale_y = height / float(req.view_height)
    native_points = [(px * scale_x, py * scale_y) for px, py in raw_points]

    try:
        try:
            mask = sam_segment.detect_subject_mask_sam(
                frame_rgb,
                native_points,
                labels=req.labels,
                cache_key=(str(vid_path), vid_path.stat().st_mtime),
            )
            engine = "sam"
        except sam_segment.SamModelMissing:
            # First-use recovery: fetch SAM from the model hub, then retry.
            # Single clicks fall back to GrabCut when the download fails.
            try:
                _ensure_hub_group("sam")
                mask = sam_segment.detect_subject_mask_sam(
                    frame_rgb,
                    native_points,
                    labels=req.labels,
                    cache_key=(str(vid_path), vid_path.stat().st_mtime),
                )
                engine = "sam"
            except modelhub.HubError as exc:
                if len(native_points) != 1:
                    # Multi-click refinement needs SAM; single clicks can fall back.
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "SAM model is not installed and the automatic "
                            f"download failed: {exc}"
                        ),
                    )
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                mask = detect_subject_mask(frame_bgr, native_points[0])
                engine = "grabcut"
    except SubjectDetectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "mask_png": mask_to_white_png_b64(mask),
        "width": width,
        "height": height,
        "engine": engine,
    }


@router.post("/jobs/start")
def start_job(req: StartJobRequest) -> Dict[str, Any]:
    input_dir = Path(req.input_dir) if req.input_dir else config.INPUT_DIR
    output_dir = Path(req.output_dir) if req.output_dir else config.OUTPUT_DIR
    if not req.video_names:
        raise HTTPException(status_code=400, detail="No videos specified for processing")
    video_paths: List[Path] = []
    for name in req.video_names:
        p = Path(name)
        if not p.is_absolute():
            p = input_dir / name
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Input video not found: {p.name}")
        video_paths.append(p)
    try:
        job = job_manager.submit_job(
            action=req.action,
            video_paths=video_paths,
            params=req.params,
            output_dir=output_dir,
        )
        return {"success": True, "job": job.to_dict()}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found or already finished")
    return {"success": True, "job_id": job_id}

@router.get("/jobs/status")
def get_job_status() -> Dict[str, Any]:
    active = job_manager.get_active_job()
    if active:
        return {"active": True, "job": active.to_dict()}
    return {"active": False, "job": None}

@router.get("/stream/video")
def stream_video(request: Request, path: str = Query(...)) -> Any:
    vid_path = Path(path)
    if not vid_path.is_absolute():
        vid_path = (config.BASE_DIR / path).resolve()
    return stream_video_file(vid_path, request)

@router.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket) -> None:
    await job_manager.connection_manager.connect(websocket)
    try:
        active = job_manager.get_active_job()
        if active:
            await websocket.send_json({"type": "job_progress", "job": active.to_dict()})
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        job_manager.connection_manager.disconnect(websocket)
    except Exception:
        job_manager.connection_manager.disconnect(websocket)
