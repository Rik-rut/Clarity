# Clarity Windows Desktop App — Design

Date: 2026-09-14
Status: Approved (pending implementation plan)

Note: intended final home is `docs/superpowers/specs/2026-09-14-tauri-windows-desktop-design.md`
(staging here is required by current workspace write permissions).

## Goal

Package Clarity as a Windows desktop app: a Tauri v2 shell with a per-user NSIS
installer (Clarity logo), where a first-run setup wizard handles all heavy
dependencies (Python runtime, torch, models). Windows-only for now.

## Decisions

- **Framework**: Tauri v2. The existing FastAPI backend runs as a sidecar
  process; the webview loads the existing vanilla HTML/JS/CSS UI unchanged.
- **Dependency strategy**: Setup wizard downloads. Installer is small
  (~150 MB: Tauri shell, uv, FFmpeg). torch, models, and the Python runtime
  are downloaded on first run. Internet is required during setup only.
- **GPU handling**: The wizard detects NVIDIA via `nvidia-smi` and installs
  the matching torch build (cu126 for NVIDIA, CPU otherwise). TensorRT is an
  optional wizard extra on NVIDIA machines. The ncnn Vulkan fallback remains
  available for AMD/Intel/no-GPU users.
- **Frontend**: Reused as-is. Desktop chrome (title bar, icon, installer,
  window lifecycle) is the only new surface.
- **Install model**: Per-user, no admin/UAC. App in
  `%LOCALAPPDATA%\Programs\Clarity\`; writable data in `%LOCALAPPDATA%\Clarity\`.
- **Dev flow preserved**: `run.bat` / CLI keep working unchanged; packaging is
  purely additive.

## Architecture

### Install layout

```
%LOCALAPPDATA%\Programs\Clarity\
├── Clarity.exe                       Tauri shell
├── sidecar\
│   ├── uv.exe                        bundled
│   ├── bootstrap.py                  tiny: setup-state check + launch
│   ├── app\                          Python project (source, not frozen)
│   │   ├── video_upscaler\...
│   │   ├── pyproject.toml
│   │   └── uv.lock
│   └── ffmpeg\bin\ffmpeg.exe, ffprobe.exe   bundled
└── resources\                        icons, licenses

%LOCALAPPDATA%\Clarity\               created at first run
├── runtime\                          uv-managed Python 3.11 + .venv
├── models\                           modelhub downloads
├── tools\ncnn\                       ncnn vulkan runtime (existing flow)
├── input\, output\                   user media
├── logs\
└── setup.json                        wizard state machine
```

### Process model

1. `Clarity.exe` starts, checks `setup.json`.
2. Spawn sidecar: `uv run --project sidecar\app clarity-web --port <free>`,
   with `CLARITY_*` env vars pointed at `%LOCALAPPDATA%\Clarity\*` and
   `CLARITY_FFMPEG` at the bundled ffmpeg.
3. Webview navigates to `127.0.0.1:<port>`. If setup is incomplete, the server
   serves the wizard UI; otherwise the existing main UI.
4. On app exit the shell kills the sidecar. On sidecar crash the shell shows
   an error window with captured stderr and a Relaunch button.

Tauri is responsible only for: window/chrome/icon, sidecar lifecycle,
single-instance, error surface. All product logic stays in the FastAPI app.

## First-run wizard

Implemented as a new FastAPI router (`video_upscaler/web/routes/setup.py`) plus
a static page (`static/setup.html` + `js/setup.js`), so the wizard is just
another page in the same app — no separate tooling.

Steps (each writes state to `setup.json` and is individually retriable):

1. **Welcome** — logo, download-size summary (~2.8 GB with GPU torch),
   disk-space check on `%LOCALAPPDATA%`.
2. **GPU detection** — run `nvidia-smi`; select torch wheel index
   (cu126 vs CPU); offer TensorRT checkbox when NVIDIA.
3. **Install runtime** — `uv sync` / `uv pip install` into
   `runtime\.venv`; stdout streamed to the UI for progress; retriable.
4. **Models** — reuse `modelhub.py` + `data/manifest.json` exactly;
   essential set by default, optional "download everything"; sha256
   verification already built in.
5. **Verify + finish** — probe ffmpeg, import torch, small CUDA smoke test
   when applicable; write final `setup.json`; hand off to main UI.

On later launches the shell checks `setup.json` (plus a cheap torch-import
sanity probe) and goes straight to the main UI.

## Error handling

- **No internet / mid-download failure**: wizard steps retriable; uv and
  modelhub already write atomically, so partial downloads never corrupt state.
- **Backend crash at launch**: shell captures stderr, shows error window with
  log path (`%LOCALAPPDATA%\Clarity\logs\`) and Relaunch.
- **Port conflicts**: `find_free_port()` already handles this.
- **SmartScreen/AV**: unsigned binaries warn on first run. Code signing is a
  future cost, not a blocker for v1.

## Build, testing, verification

- **Build**: GitHub Actions Windows runner → `tauri build` → NSIS installer
  + portable zip artifacts.
- **Automated tests**: pytest for `routes/setup.py` and the bootstrap/setup
  state machine (mocked subprocesses); existing suite must stay green.
- **Manual verification**: clean Windows VM smoke checklist — install, wizard
  with and without internet drop, NVIDIA and CPU paths, launch, upscale one
  clip, uninstall leaves `%LOCALAPPDATA%\Clarity\` (documented behavior).

## Out of scope (v1)

- macOS/Linux packaging.
- Code signing certificates.
- Auto-update (Tauri updater) — possible follow-up.
- File associations / deep links.
- Any UI redesign beyond the wizard.

---

## Addendum (2026-09-15): one directory, installer-provisioned

There is exactly one Clarity directory: the folder chosen on the installer's
directory page. It holds the program, the uv-managed interpreter, the venv, the
model weights, the caches, the logs and the default input/output media.

`AppState::default_app_data_dir` resolves, in order:

1. `CLARITY_DATA_DIR` — development and tests only.
2. The folder containing `clarity-desktop.exe`, when a packaged build can write
   to it (or already provisioned it).
3. `%LOCALAPPDATA%\Clarity` — only for a read-only install such as Program Files.

`hooks.nsh` computes the same location from `$INSTDIR` with the same write probe,
so the installer and the app cannot disagree. There is no remembered-location
file: two sources of truth is how the models came to be downloaded twice.

Provisioning has one implementation, `video_upscaler/desktop/provision.py`. The
installer runs it during setup; the boot shell runs it only when
`.setup_complete` is missing, and it resumes from `setup.json`, adopts an
environment that already imports the stack, and never re-downloads what is on
disk. `.setup_complete` is written by that script alone, after every step
succeeded.

Uninstalling removes the regenerable directories and asks before deleting
`input\` and `output\`. An update (`$UpdateMode = 1`) removes nothing.
