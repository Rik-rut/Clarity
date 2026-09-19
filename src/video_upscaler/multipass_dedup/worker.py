"""Persistent MultiPassDedup inference worker.

The model is loaded once and then reused for every request. Running inference
in a long-lived process removes the per-video load cost that dominated dedup
renders: torch/cupy import, checkpoint loading, CUDA context creation, the
softsplat kernel compile and cuDNN autotuning used to happen again for every
file and every render.

Protocol: JSON objects, one per line.
  stdout events:
    {"event": "ready", "model": "gmfss"}
    {"event": "progress", "done": <int>, "total": <int>}
    {"event": "done", "output": "<path>"}
    {"event": "error", "message": "<text>"}
  stdin commands:
    {"cmd": "load", "model_type": "gmfss", "scale": 1.0, "weights": "<dir>"}
    {"cmd": "run", "video": "<path>", "output": "<path>", "model_type": "gmfss",
     "npass": 0, "times": 2, "scale": 1.0, "scdet": true,
     "threshold": 0.3, "hwaccel": false, "weights": "<dir>"}
    {"cmd": "exit"}

Anything that is not protocol output goes to stderr so stdout stays parseable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_PROGRESS_EVERY_FRAMES = 4


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _load_model(model_type: str, weights: str, scale: float):
    from models.vfi import VFI

    return VFI(model_type=model_type, weights=weights, scale=scale, device=DEVICE)


def run_video(
    model,
    video: str,
    output: str,
    n_pass: int,
    times: int,
    enable_scdet: bool,
    scdet_threshold: float,
    hwaccel: bool,
    on_progress,
) -> str:
    """One video through the upstream cache pipeline (port of infer.py main)."""
    from models.utils.tools import (
        TMapper,
        VideoFI_IO,
        check_scene,
        get_valid_net_inp_size,
        to_inp,
        to_out,
    )

    video_io = VideoFI_IO(video, output, dst_fps=60, times=times, hwaccel=hwaccel)
    src_fps = video_io.src_fps
    target_fps = video_io.dst_fps
    if target_fps <= src_fps:
        raise ValueError(
            "dst fps should be greater than src fps, but got "
            f"tar_fps={target_fps} and src_fps={src_fps}"
        )

    passes = int(n_pass)
    if passes == 0:
        # Auto-detect duplicate cadence (upstream formula).
        passes = math.ceil(src_fps / 24000 * 1001) * 2
    mapper = TMapper(src_fps, target_fps, times)

    first = video_io.read_frame()
    if first is None:
        raise ValueError("video doesn't contains any frames")
    size = get_valid_net_inp_size(first, model.scale, div=model.pad_size)
    src_size, dst_size = size["src_size"], size["dst_size"]
    first_in = to_inp(first, dst_size)

    cache = {index: [first_in] for index in range(passes)}
    state = {"head_end": False, "tail_end": False, "frame_idx": 0}
    total = int(video_io.total_frames_count or 0)

    def infer(cache_idx: int) -> None:
        head = cache_idx == 0
        tail = cache_idx == len(cache) - 1

        if head and len(cache[cache_idx]) != 2:
            frame = video_io.read_frame()
            if frame is None:
                state["head_end"] = True
                cache[cache_idx].append(cache[cache_idx][0])
            else:
                cache[cache_idx].append(to_inp(frame, dst_size))

        if len(cache[cache_idx]) == 2:
            inp0 = cache[cache_idx][0]
            inp1 = cache[cache_idx][1]
            timestamps = [0.5]
            if tail:
                timestamps = mapper.get_range_timestamps(
                    state["frame_idx"],
                    state["frame_idx"] + 1,
                    lclose=True,
                    rclose=state["head_end"],
                    normalize=True,
                )
            if enable_scdet and check_scene(inp0, inp1, scdet_threshold):
                timestamps = [0 for _ in timestamps]

            if not tail:
                cache[cache_idx + 1].append(
                    model.gen_ts_frame(inp0, inp1, timestamps)[0]
                )
            else:
                for out in model.gen_ts_frame(inp0, inp1, timestamps):
                    video_io.write_frame(to_out(out, src_size))
                state["frame_idx"] += 1
                if state["head_end"]:
                    state["tail_end"] = True

            cache[cache_idx].pop(0)
        elif len(cache[cache_idx]) == 1:
            if state["head_end"]:
                cache[cache_idx].append(cache[cache_idx][0])
                infer(cache_idx)
            else:
                infer(cache_idx - 1)
        else:
            raise ValueError(
                f"cache[{cache_idx}] should have 1 or 2 elements, "
                f"but got {len(cache[cache_idx])}"
            )

    while not state["tail_end"]:
        for index in range(passes):
            infer(index)
        if on_progress is not None:
            on_progress(state["frame_idx"], total)

    while not video_io.finish_writing():
        time.sleep(0.1)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="MultiPassDedup worker")
    parser.add_argument("-w", "--weights", default="weights", help="weights folder")
    args = parser.parse_args()

    loaded: tuple[str, float, str] | None = None
    model = None

    def ensure_model(request: dict):
        nonlocal model, loaded
        model_type = str(request.get("model_type") or "gmfss")
        scale = float(request.get("scale") or 1.0)
        weights = str(request.get("weights") or args.weights)
        key = (model_type, scale, weights)
        if model is None or loaded != key:
            model = _load_model(model_type, weights, scale)
            loaded = key
        return model

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"event": "error", "message": f"invalid request: {exc}"})
            continue

        command = request.get("cmd")
        if command == "exit":
            break

        try:
            if command == "load":
                ensure_model(request)
                _emit({"event": "ready", "model": request.get("model_type")})
            elif command == "run":
                current = ensure_model(request)

                def on_progress(done: int, total: int) -> None:
                    if done % _PROGRESS_EVERY_FRAMES == 0 or (total and done >= total):
                        _emit({"event": "progress", "done": done, "total": total})

                output = run_video(
                    current,
                    video=str(request["video"]),
                    output=str(request["output"]),
                    n_pass=int(request.get("npass") or 0),
                    times=int(request.get("times") or 2),
                    enable_scdet=bool(request.get("scdet", True)),
                    scdet_threshold=float(request.get("threshold") or 0.3),
                    hwaccel=bool(request.get("hwaccel", False)),
                    on_progress=on_progress,
                )
                _emit({"event": "done", "output": output})
            else:
                _emit({"event": "error", "message": f"unknown cmd {command!r}"})
        except Exception as exc:  # noqa: BLE001 - report and keep serving
            _emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
