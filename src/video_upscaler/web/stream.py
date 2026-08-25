"""HTTP 206 Partial Content video streaming response handler."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Generator

from fastapi import HTTPException, Request, status
from fastapi.responses import StreamingResponse

# Ensure common video mime types are recognized
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/x-matroska", ".mkv")
mimetypes.add_type("video/quicktime", ".mov")
mimetypes.add_type("video/x-msvideo", ".avi")
mimetypes.add_type("video/webm", ".webm")
mimetypes.add_type("video/mp4", ".m4v")
mimetypes.add_type("video/mp2t", ".ts")

CHUNK_SIZE = 1024 * 512  # 512 KB chunks for smooth scrubbing


def _iter_file(path: Path, start: int, length: int) -> Generator[bytes, None, None]:
    """Yield file chunks for range response."""
    with open(path, "rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            read_size = min(CHUNK_SIZE, remaining)
            data = f.read(read_size)
            if not data:
                break
            remaining -= len(data)
            yield data


def stream_video_file(path: Path, request: Request) -> StreamingResponse:
    """Return a StreamingResponse with HTTP 206 Partial Content support."""
    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found: {path.name}",
        )

    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "video/mp4"

    if range_header:
        # Expected format: "bytes=start-end" or "bytes=start-"
        try:
            h_range = range_header.strip().lower()
            if not h_range.startswith("bytes="):
                raise ValueError("Invalid range prefix")
            range_val = h_range[len("bytes=") :]
            parts = range_val.split("-")
            start_str = parts[0].strip()
            end_str = parts[1].strip() if len(parts) > 1 else ""

            if start_str and end_str:
                start = int(start_str)
                end = int(end_str)
            elif start_str:
                start = int(start_str)
                end = file_size - 1
            elif end_str:
                length = int(end_str)
                start = max(0, file_size - length)
                end = file_size - 1
            else:
                start = 0
                end = file_size - 1

            if start >= file_size or start < 0 or end >= file_size or start > end:
                return StreamingResponse(
                    iter([]),
                    status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                    headers={"Content-Range": f"bytes */{file_size}"},
                )

            content_length = end - start + 1
            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": mime_type,
                "Cache-Control": "no-cache",
            }
            return StreamingResponse(
                _iter_file(path, start, content_length),
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                headers=headers,
            )
        except Exception:
            pass

    # Full content delivery
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": mime_type,
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(
        _iter_file(path, 0, file_size),
        status_code=status.HTTP_200_OK,
        headers=headers,
    )
