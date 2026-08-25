"""FastAPI web server for Clarity Video Upscaler."""

from __future__ import annotations

import asyncio
import logging
import socket
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from video_upscaler import config
from video_upscaler.web.jobs import job_manager
from video_upscaler.web.routes.api import router as api_router

logger = logging.getLogger("clarity.web")

STATIC_DIR = Path(__file__).resolve().parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    job_manager.set_loop(loop)
    config.ensure_directories()
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    (STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
    (STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Clarity Video AI",
        version="0.1.0",
        description="Web-based graphical interface for Clarity video upscaling and interpolation.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def read_index():
        from fastapi.responses import HTMLResponse
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<!DOCTYPE html><html><head><title>Clarity Video AI</title></head><body><h1>Clarity Video AI</h1></body></html>")

    return app


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0



def find_free_port(start_port: int = 7860, host: str = "127.0.0.1") -> int:
    port = start_port
    while is_port_in_use(port, host):
        port += 1
        if port > start_port + 50:
            break
    return port


def open_browser_when_ready(url: str, delay: float = 1.0) -> None:
    def _open():
        time.sleep(delay)
        webbrowser.open(url)

    t = threading.Thread(target=_open, daemon=True)
    t.start()



def run_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = True,
) -> None:
    import uvicorn

    actual_port = find_free_port(port, host)
    url = f"http://{host}:{actual_port}"

    print("")
    print("======================================================")
    print(f"   Clarity Video UI is running at: {url}")
    print("======================================================")
    print("")

    if open_browser:
        open_browser_when_ready(url, 1.0)

    app = create_app()
    uvicorn.run(app, host=host, port=actual_port, log_level="info")


def main() -> None:
    port = 7860
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            try:
                port = int(sys.argv[idx + 1])
            except ValueError:
                pass

    no_browser = "--no-browser" in sys.argv
    run_server(port=port, open_browser=not no_browser)


if __name__ == "__main__":
    main()
