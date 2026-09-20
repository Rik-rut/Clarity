"""FastAPI web server for Clarity Video Upscaler."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse(
            "<!DOCTYPE html><html><head><title>Clarity Video AI</title></head>"
            "<body><h1>Clarity Video AI</h1></body></html>"
        )

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


def warm_heavy_imports(delay: float = 0.5) -> None:
    """Import torch and probe the backend off the request path.

    The first render (and ``/api/system/info``) needs torch, whose cold import
    costs roughly ten seconds. Doing it in the background lets the window open
    immediately instead of making the user wait at "Starting the studio server…".
    Best effort: any failure is logged and ignored.
    """
    if os.environ.get("CLARITY_DISABLE_WARMUP") == "1":
        return

    def _warm() -> None:
        time.sleep(delay)
        try:
            from video_upscaler.backend import prime_detection

            prime_detection()
            logger.debug("Heavy runtime warm-up complete")
        except Exception as exc:  # noqa: BLE001 - warm-up must never break boot
            logger.debug("Heavy runtime warm-up skipped: %s", exc)

    threading.Thread(target=_warm, name="clarity-warmup", daemon=True).start()



def run_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = True,
    strict_port: bool = False,
) -> None:
    import uvicorn

    if strict_port:
        # The desktop shell reserved this exact port and is polling it. Drifting
        # to port+1 would leave the shell polling a dead port forever, so fail
        # loudly instead; the shell surfaces the failure with the backend log.
        if is_port_in_use(port, host):
            raise SystemExit(f"Requested port {port} is already in use on {host}")
        actual_port = port
    else:
        actual_port = find_free_port(port, host)

    url = f"http://{host}:{actual_port}"

    print("")
    print("======================================================")
    print(f"   Clarity Video UI is running at: {url}")
    print("======================================================")
    print("")

    if os.environ.get("CLARITY_DESKTOP_MODE") == "1":
        open_browser = False

    if open_browser:
        open_browser_when_ready(url, 1.0)

    app = create_app()
    warm_heavy_imports()
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
    strict_port = "--strict-port" in sys.argv
    run_server(port=port, open_browser=not no_browser, strict_port=strict_port)


if __name__ == "__main__":
    main()
