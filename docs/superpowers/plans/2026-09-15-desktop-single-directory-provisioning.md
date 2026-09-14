# Desktop Single-Directory Provisioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the folder the user picks in the installer the one and only place Clarity lives — program, interpreter, venv, models, media — with the installer doing the provisioning and no first-run wizard.

**Architecture:** One data-root rule (`CLARITY_DATA_DIR` → the folder containing `clarity-desktop.exe` → `%LOCALAPPDATA%\Clarity` for read-only installs) computed identically by NSIS and Rust, with no pointer file to disagree. One headless provisioner (`video_upscaler/desktop/provision.py`) that both the installer and the boot shell call, so the two can never diverge. The wizard HTTP server, its page and its JS are deleted; the Tauri boot shell renders repair progress instead.

**Tech Stack:** Tauri v2 (Rust), NSIS installer hooks, Python 3.11 stdlib-only provisioner, `uv` for interpreter/venv/deps, existing `video_upscaler` package.

**Spec:** `docs/superpowers/specs/2026-09-14-tauri-windows-desktop-design.md` — note that its "Addendum (2026-09-15): where desktop data actually lives" section describes the pointer-file design this plan **replaces**; Task 10 rewrites it.

## Global Constraints

- Python: uv-managed **3.11**, installed into `<data>\python`. Never use a host or roaming interpreter (`--python-preference only-managed`).
- Torch index for NVIDIA: `https://download.pytorch.org/whl/cu126` (`PYTORCH_CU126_INDEX` in `runtime.py`). CPU otherwise.
- Model tier at install time: `essential` (`models.DEFAULT_TIER`).
- `.setup_complete` is written **only** by `provision.py`, and only after every step succeeded. NSIS must never write it.
- Data root layout is **flat** inside the install folder: `python\ env\ models\ tools\ .cache\ logs\ input\ output\ setup.json .setup_complete`.
- Uninstalling removes regenerable data and **asks** before deleting `input\`/`output\`; an update (`$UpdateMode = 1`) must delete nothing.
- No new dependencies (Rust crates, npm packages, or Python packages) beyond the existing `tensorrt` extra in `pyproject.toml`.
- Windows-only code paths must be `#[cfg(windows)]`/`os.name == "nt"` guarded so the suite still runs elsewhere.
- Commit style follows history: `feat(desktop): …`, `fix(desktop): …`, `build(desktop): …`.
- Every task ends green: `pytest -q --ignore=tests/e2e`, `node --test tests/<file>.js`, `cargo test`, `cargo clippy --all-targets`.
- Progress wire format (both callers parse it): `PROGRESS <percent>|<phase>|<message>`, one per line, flushed.

## File Structure

**Create**
- `src/video_upscaler/desktop/provision.py` — the single headless provisioner (CLI + progress emitter + lock).
- `tests/test_desktop_provision.py` — its tests (keepers moved out of `tests/test_desktop_wizard.py`).
- `tools/verify_install_layout.ps1` — asserts layout after a sandbox install/uninstall (Tasks 8–9).
- `docs/superpowers/plans/2026-09-15-desktop-single-directory-provisioning.md` — this plan.

**Modify**
- `src-tauri/src/state.rs` — data-root rule, writability probe, legacy merge; delete pointer-file code.
- `src-tauri/src/setup.rs` — `parse_provision_line`, `run_provisioner`, parser parameter on the stream helpers.
- `src-tauri/src/process.rs` — `build_bootstrap_command` → `build_provision_command`.
- `src-tauri/src/boot.rs` — remove the wizard; drive the provisioner; emit `boot-progress`.
- `src-tauri/shell/index.html`, `src-tauri/shell/shell.js` — progress bar, percent, phase, log tail.
- `src-tauri/hooks.nsh` — data-dir probe, managed interpreter, TensorRT question, provisioner call, error gating, uninstall cleanup.
- `src/video_upscaler/desktop/runtime.py` — `tensorrt` extra + backend reporting in `verify_runtime`.
- `src/video_upscaler/desktop/state.py` — drop the pointer read; add `tensorrt`/`backend` fields.
- `src/video_upscaler/desktop/context.py` — nothing structural; `build_context` already accepts `data_dir`/`resources_dir`.
- `tests/test_frontend_assets.py`, `tests/test_setup_wizard_js.js` (→ `tests/test_desktop_bridge.js`).
- `docs/superpowers/specs/2026-09-14-tauri-windows-desktop-design.md` — replace the addendum.

**Delete**
- `src/video_upscaler/desktop/server.py`, `bootstrap.py`, `location.py`
- `src/video_upscaler/web/static/setup.html`, `src/video_upscaler/web/static/js/setup.js`
- `tests/test_desktop_wizard.py` (after Task 3 moves the keepers)

---

### Task 1: One data root — the install folder

**Files:**
- Modify: `src-tauri/src/state.rs` (replace `default_app_data_dir`, `resolve_data_dir`; delete `data_location_pointer`, `read_pointer`, `write_pointer`, `install_drive_dir`; keep `drive_root`, `packaged_install_dir`)
- Modify: `src/video_upscaler/desktop/state.py:44-66` (`default_data_dir`)
- Delete: `src/video_upscaler/desktop/location.py`
- Test: `src-tauri/src/state.rs` (`mod tests`)

**Interfaces:**
- Produces: `AppState::default_app_data_dir() -> PathBuf`, `AppState::resolve_data_dir(override_dir: Option<&Path>, install_dir: Option<&Path>, local_dir: &Path) -> PathBuf`, `AppState::is_writable(dir: &Path) -> bool`, `drive_root(path: &Path) -> Option<String>` (kept for Task 2).
- Consumes: `crate::setup::is_setup_complete(&Path) -> bool`, `packaged_install_dir() -> Option<PathBuf>`.

- [ ] **Step 1: Write the failing Rust tests**

Replace the pointer-file tests in `src-tauri/src/state.rs` `mod tests` with:

```rust
    #[test]
    fn test_data_root_is_the_install_folder() {
        let root = std::env::temp_dir().join("clarity_root_install");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let dir = AppState::resolve_data_dir(
            None,
            Some(root.as_path()),
            Path::new("C:\Users\someone\AppData\Local"),
        );
        assert_eq!(dir, root, "the folder the user installed into is the data root");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_data_root_falls_back_when_the_install_folder_is_not_writable() {
        // A file where the directory should be makes create_dir_all fail, which
        // is how a read-only install (Program Files) presents itself.
        let root = std::env::temp_dir().join("clarity_root_blocked");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let blocker = root.join("not-a-dir");
        std::fs::write(&blocker, b"x").unwrap();

        let dir = AppState::resolve_data_dir(
            None,
            Some(blocker.as_path()),
            root.as_path(),
        );
        assert_eq!(dir, root.join("Clarity"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_data_root_keeps_a_complete_install_even_if_the_probe_would_fail() {
        let root = std::env::temp_dir().join("clarity_root_complete");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("env").join("Scripts")).unwrap();
        std::fs::write(root.join(".setup_complete"), b"1").unwrap();
        std::fs::write(root.join("env").join("Scripts").join("python.exe"), b"").unwrap();

        assert!(crate::setup::is_setup_complete(&root));
        let dir = AppState::resolve_data_dir(None, Some(root.as_path()), Path::new("C:\nowhere"));
        assert_eq!(dir, root);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_explicit_override_still_wins() {
        let dir = AppState::resolve_data_dir(
            Some(Path::new("/tmp/hand-picked")),
            Some(Path::new("/tmp/install")),
            Path::new("/tmp/local"),
        );
        assert_eq!(dir, PathBuf::from("/tmp/hand-picked"));
    }
```

Delete `test_resolve_data_dir_reads_the_remembered_choice`, `test_resolve_data_dir_ignores_a_relative_pointer`, `test_install_drive_rule_follows_the_app_not_the_profile` and `test_install_drive_dir_never_nests_inside_the_install_folder` — the pointer file and the `Clarity-data` rule are gone. Keep `test_resolve_data_dir_prefers_explicit_override` (rewritten above), `test_drive_root_recognises_windows_prefixes`, `test_default_app_data_dir` (a debug build has no packaged install dir, so it still resolves to `<local>\Clarity`).

- [ ] **Step 2: Run them to verify they fail**

Run: `cd src-tauri && cargo test state::`
Expected: FAIL — `resolve_data_dir` takes four arguments today and `install_drive_dir` still exists.

- [ ] **Step 3: Implement the rule**

In `src-tauri/src/state.rs` replace `default_app_data_dir` and `resolve_data_dir`, and delete `data_location_pointer`, `read_pointer`, `write_pointer`, `install_drive_dir`:

```rust
    /// Resolves where Clarity keeps everything: program, interpreter, venv,
    /// models and media.
    ///
    /// There is exactly one directory — the one the user chose in the
    /// installer. Anything else produced the bug this replaces: the installer
    /// provisioned into `%LOCALAPPDATA%\Clarity` while the app looked in
    /// `<drive>\Clarity-data`, found nothing, and downloaded the models again.
    pub fn default_app_data_dir() -> PathBuf {
        let override_dir = std::env::var_os("CLARITY_DATA_DIR").map(PathBuf::from);
        let install_dir = packaged_install_dir();
        let local_dir = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from("."));
        Self::resolve_data_dir(override_dir.as_deref(), install_dir.as_deref(), &local_dir)
    }

    /// Pure precedence rules, split out so they can be tested without env state.
    fn resolve_data_dir(
        override_dir: Option<&Path>,
        install_dir: Option<&Path>,
        local_dir: &Path,
    ) -> PathBuf {
        if let Some(dir) = override_dir.filter(|dir| !dir.as_os_str().is_empty()) {
            return dir.to_path_buf();
        }
        if let Some(dir) = install_dir {
            // An already-provisioned install wins even when the write probe
            // would fail, so a read-only mount never strands existing data.
            if crate::setup::is_setup_complete(dir) || Self::is_writable(dir) {
                return dir.to_path_buf();
            }
        }
        local_dir.join("Clarity")
    }

    /// True when Clarity can create files in `dir`. A Program Files install
    /// cannot, and must fall back rather than fail during a 5 GB download.
    pub fn is_writable(dir: &Path) -> bool {
        if std::fs::create_dir_all(dir).is_err() {
            return false;
        }
        let probe = dir.join(".clarity-write-test");
        match std::fs::write(&probe, b"ok") {
            Ok(()) => {
                let _ = std::fs::remove_file(&probe);
                true
            }
            Err(_) => false,
        }
    }
```

Keep `packaged_install_dir()` (release build with `uv.exe` or `resources\` beside the exe) — that gate is what stops `cargo test` from resolving to a stray folder and writing state.

- [ ] **Step 4: Run the Rust tests**

Run: `cd src-tauri && cargo test state::`
Expected: PASS.

- [ ] **Step 5: Simplify the Python side and delete `location.py`**

In `src/video_upscaler/desktop/state.py` replace `default_data_dir` (lines 44-66) with:

```python
def default_data_dir() -> Path:
    r"""Fallback data root: ``%LOCALAPPDATA%\Clarity`` on Windows.

    The desktop shell always passes ``CLARITY_DATA_DIR`` — the folder it was
    installed into — so this only matters for the CLI, tests and a read-only
    install. There is deliberately no remembered-location file: one rule,
    computed the same way by the installer and the app, cannot disagree with
    itself.
    """
    override = os.environ.get("CLARITY_DATA_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "Clarity"
        return Path.home() / "AppData" / "Local" / "Clarity"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "clarity"
    return Path.home() / ".local" / "share" / "clarity"
```

Drop the `read_pointer` import. Then `git rm src/video_upscaler/desktop/location.py`.

- [ ] **Step 6: Remove the now-dead location surface**

- `src/video_upscaler/desktop/server.py`: delete `relocate()`, `_root_has_provisioned_data()`, `PROVISIONING_STEPS`, the `location` import, the `location` key in `status()`, and the `/api/setup/data-dir` route plus `_data_dir()`.
- `src/video_upscaler/web/static/setup.html`: delete the `/* Runtime & media storage picker */` CSS block, the `.storage-row[hidden]` rule and the `<div class="storage-row" id="setup-storage-row">…</div>` markup.
- `src/video_upscaler/web/static/js/setup.js`: delete `renderLocation`, `showStorageError`, `tauriInvoke`, `pickFolder`, `chooseStorageLocation`, the `storage*`/`btnChangeStorage` dom entries, the `renderLocation(status.location)` call and the `btn-change-storage` listener.
- `tests/test_desktop_wizard.py`: delete the `isolated_pointer` autouse fixture, the `location`/`LocationError` imports, and these tests — `test_default_data_dir_follows_the_remembered_pointer`, the five `test_validate_new_root_*`, `test_status_reports_the_data_root_and_its_free_space`, `test_data_root_moves_before_provisioning_starts`, `test_data_root_refuses_to_move_once_provisioning_has_written_data`, `test_a_finished_gpu_scan_alone_does_not_lock_the_location`, `test_data_root_refuses_the_program_folder_over_http`.
- `tests/test_frontend_assets.py`: delete `test_wizard_lets_the_user_choose_where_the_data_lives`.

Task 7 deletes the wizard outright; this step only stops it referencing a deleted module.

- [ ] **Step 7: Run everything**

Run: `cd src-tauri && cargo test && cargo clippy --all-targets`
Run: `.venv/Scripts/python.exe -m pytest -q --ignore=tests/e2e`
Run: `node --test tests/test_setup_wizard_js.js`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add -A src-tauri/src/state.rs src/video_upscaler/desktop/ src/video_upscaler/web/static/ tests/
git commit -m "fix(desktop): make the install folder the single data root"
```

---

### Task 2: Adopt a legacy data folder instead of re-downloading

**Files:**
- Create: `src-tauri/src/migrate.rs`
- Modify: `src-tauri/src/main.rs` (module list), `src-tauri/src/boot.rs:98-118`, `src-tauri/src/state.rs` (`drive_root` → `pub(crate)`)
- Test: `src-tauri/src/migrate.rs` (`mod tests`)

**Interfaces:**
- Produces: `migrate::legacy_roots(install_dir: &Path, local_dir: &Path) -> Vec<PathBuf>`, `migrate::same_volume(a: &Path, b: &Path) -> bool`, `migrate::merge_into(legacy: &Path, target: &Path) -> Result<Vec<String>, String>`, `migrate::adopt_legacy_data(install_dir: &Path, local_dir: &Path) -> Vec<String>`.
- Consumes: `state::drive_root(&Path) -> Option<String>`, `setup::is_setup_complete(&Path) -> bool`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_moves_entries_without_overwriting_the_program() {
        let root = std::env::temp_dir().join("clarity_merge");
        let _ = std::fs::remove_dir_all(&root);
        let legacy = root.join("Clarity-data");
        let target = root.join("Clarity");
        std::fs::create_dir_all(legacy.join("models")).unwrap();
        std::fs::write(legacy.join("models").join("amt-s.pth"), b"weights").unwrap();
        std::fs::write(legacy.join("setup.json"), b"{}").unwrap();
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("clarity-desktop.exe"), b"program").unwrap();
        std::fs::write(legacy.join("clarity-desktop.exe"), b"stale").unwrap();

        let moved = merge_into(&legacy, &target).unwrap();

        assert!(moved.iter().any(|name| name == "models"));
        assert!(target.join("models").join("amt-s.pth").is_file());
        assert!(target.join("setup.json").is_file());
        assert_eq!(
            std::fs::read(target.join("clarity-desktop.exe")).unwrap(),
            b"program",
            "the installed program must survive a merge"
        );
        assert!(!legacy.exists(), "an emptied legacy folder is removed");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_same_volume_compares_drive_roots() {
        assert!(same_volume(Path::new("D:\Clarity-data"), Path::new("D:\Clarity")));
        assert!(!same_volume(Path::new("C:\Users\x"), Path::new("D:\Clarity")));
        assert!(!same_volume(Path::new("relative"), Path::new("D:\Clarity")));
    }

    #[test]
    fn test_legacy_roots_are_the_two_historical_locations() {
        let roots = legacy_roots(
            Path::new("D:\Clarity"),
            Path::new("C:\Users\x\AppData\Local"),
        );
        assert!(roots.contains(&PathBuf::from("D:\Clarity-data")));
        assert!(roots.contains(&PathBuf::from("C:\Users\x\AppData\Local\Clarity")));
    }

    #[test]
    fn test_adoption_skips_a_legacy_folder_that_was_never_finished() {
        // `local_dir.join("Clarity")` is the second legacy candidate, so this
        // exercises adopt_legacy_data end to end without touching a drive root.
        let root = std::env::temp_dir().join("clarity_merge_partial");
        let _ = std::fs::remove_dir_all(&root);
        let legacy = root.join("Clarity");
        std::fs::create_dir_all(legacy.join("env").join("Scripts")).unwrap();
        std::fs::write(legacy.join("env").join("Scripts").join("python.exe"), b"").unwrap();
        // No .setup_complete: the install that made this folder never finished.
        let install = root.join("install");
        std::fs::create_dir_all(&install).unwrap();

        let log = adopt_legacy_data(&install, &root);

        assert!(log.is_empty());
        assert!(legacy.join("env").is_dir(), "a half-finished folder is left alone");
        assert!(!install.join("env").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd src-tauri && cargo test migrate::`
Expected: FAIL — `file not found for module migrate`.

- [ ] **Step 3: Implement `src-tauri/src/migrate.rs`**

```rust
//! One-time adoption of data folders created by earlier builds.
//!
//! Two historical locations exist: `<install drive>\Clarity-data` (a rule that
//! put data on the install drive but outside the install folder) and
//! `%LOCALAPPDATA%\Clarity` (what the installer hook used). Both can hold an
//! interpreter, a venv and gigabytes of weights, so the fix is to move them into
//! the install folder rather than download them again — but only when a rename
//! can do it, because copying 5 GB across volumes during startup is worse than
//! the duplication it avoids.

use std::path::{Path, PathBuf};

use crate::state::drive_root;

/// Folders an earlier build may have provisioned into.
pub fn legacy_roots(install_dir: &Path, local_dir: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(root) = drive_root(install_dir) {
        roots.push(Path::new(&root).join("Clarity-data"));
    }
    roots.push(local_dir.join("Clarity"));
    roots
}

/// True when both paths sit on the same Windows volume, so `rename` is instant.
pub fn same_volume(a: &Path, b: &Path) -> bool {
    match (drive_root(a), drive_root(b)) {
        (Some(left), Some(right)) => left.eq_ignore_ascii_case(&right),
        _ => false,
    }
}

/// Move every entry of `legacy` into `target` that is not already there, then
/// remove `legacy` if it ended up empty. Returns the moved entry names.
pub fn merge_into(legacy: &Path, target: &Path) -> Result<Vec<String>, String> {
    let entries = std::fs::read_dir(legacy)
        .map_err(|e| format!("cannot read {}: {e}", legacy.display()))?;
    let mut moved = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read {}: {e}", legacy.display()))?;
        let from = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        let to = target.join(&name);
        // Never replace program files, and never mix two envs or model sets.
        if to.exists() {
            continue;
        }
        std::fs::rename(&from, &to)
            .map_err(|e| format!("cannot move {} to {}: {e}", from.display(), to.display()))?;
        moved.push(name);
    }

    let _ = std::fs::remove_dir(legacy);
    Ok(moved)
}

/// Adopt a legacy folder when it holds a finished runtime and the install folder
/// does not. Same volume only; everything else is left for the user to delete.
pub fn adopt_legacy_data(install_dir: &Path, local_dir: &Path) -> Vec<String> {
    if crate::setup::is_setup_complete(install_dir) {
        return Vec::new();
    }
    let mut log = Vec::new();
    for legacy in legacy_roots(install_dir, local_dir) {
        if !legacy.is_dir() || legacy == install_dir || !same_volume(&legacy, install_dir) {
            continue;
        }
        if !crate::setup::is_setup_complete(&legacy) {
            continue;
        }
        match merge_into(&legacy, install_dir) {
            Ok(moved) if !moved.is_empty() => {
                log.push(format!("adopted {} from {}", moved.join(", "), legacy.display()))
            }
            Ok(_) => {}
            Err(err) => log.push(format!("could not adopt {}: {err}", legacy.display())),
        }
    }
    log
}
```

In `src-tauri/src/state.rs` change `fn drive_root(` to `pub(crate) fn drive_root(`. In `src-tauri/src/main.rs` add `mod migrate;` beside the other module declarations.

- [ ] **Step 4: Run the tests**

Run: `cd src-tauri && cargo test migrate::`
Expected: PASS (4 tests).

- [ ] **Step 5: Call it from boot, before the completeness check**

In `src-tauri/src/boot.rs`, at the top of `boot_sequence`:

```rust
    report(app, "Starting Clarity…");

    // Adopt data an earlier build left elsewhere, so nobody re-downloads 5 GB.
    {
        let state = app.state::<AppState>();
        let local_dir = dirs::data_local_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        for line in crate::migrate::adopt_legacy_data(state.app_data_dir(), &local_dir) {
            eprintln!("[clarity] {line}");
            report(app, line);
        }
    }
```

The completeness check that follows must re-read the marker, because the merge may have just delivered it — it already does (`setup::is_setup_complete(state.app_data_dir())` runs after this block).

- [ ] **Step 6: Verify and commit**

Run: `cd src-tauri && cargo test && cargo clippy --all-targets`
Expected: green.

```bash
git add src-tauri/src/migrate.rs src-tauri/src/main.rs src-tauri/src/boot.rs src-tauri/src/state.rs
git commit -m "feat(desktop): adopt legacy data folders instead of re-downloading"
```

---

### Task 3: One headless provisioner

**Files:**
- Create: `src/video_upscaler/desktop/provision.py`
- Create: `tests/test_desktop_provision.py`
- Modify: `src/video_upscaler/desktop/state.py` (add `tensorrt` and `backend` fields)
- Move: the still-relevant tests out of `tests/test_desktop_wizard.py` (Task 7 deletes that file)

**Interfaces:**
- Produces: `provision.parse_args(argv) -> argparse.Namespace` (flags `--data-dir`, `--resources-dir`, `--tier`, `--tensorrt`, `--force`), `provision.main(argv) -> int` (`EXIT_OK = 0`, `EXIT_FAILED = 1`), `provision.acquire_lock(data_dir) -> Optional[int]`, `provision.release_lock(data_dir) -> None`, `provision.LinePrinter` (a `ProgressTracker` subclass), `provision.run_steps(context, state, tracker, args) -> None`, `provision.STEPS`.
- Consumes: `context.build_context(data_dir=…, resources_dir=…) -> SetupContext`, `gpu.detect_gpu() -> GpuInfo`, `runtime.create_venv(context, tracker)`, `runtime.install_dependencies(context, tracker, torch_variant)`, `runtime.verify_runtime(context, tracker) -> dict`, `models.install_models(context, tracker, tier)`, `state.load_state/save_state/write_marker/is_setup_complete`, `progress.ProgressTracker`.

- [ ] **Step 1: Add the two state fields**

In `src/video_upscaler/desktop/state.py`, extend `SetupState` and its (de)serialisation:

```python
    torch_variant: str = ""
    gpu_names: List[str] = field(default_factory=list)
    models_tier: str = ""
    tensorrt: bool = False
    backend: str = ""
```

In `to_dict()` add `"tensorrt": self.tensorrt, "backend": self.backend,` and in `from_dict()` add:

```python
            tensorrt=bool(payload.get("tensorrt", False)),
            backend=str(payload.get("backend", "") or ""),
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_desktop_provision.py`:

```python
"""Contract tests for the headless provisioner.

Both callers — the NSIS installer and the Tauri boot shell — parse the same
`PROGRESS <percent>|<phase>|<message>` lines and rely on the same exit codes, so
those two things are what these tests pin down.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from video_upscaler.desktop import provision
from video_upscaler.desktop import state as setup_state
from video_upscaler.desktop.context import build_context
from video_upscaler.desktop.gpu import GpuInfo
from video_upscaler.desktop.progress import ProgressTracker
from video_upscaler.desktop.runtime import SetupError


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "Clarity"
    root.mkdir()
    monkeypatch.setenv("CLARITY_DATA_DIR", str(root))
    return root


def write_fake_env(env_dir: Path, home: Path) -> None:
    scripts = env_dir / ("Scripts" if os.name == "nt" else "bin")
    binary = "python.exe" if os.name == "nt" else "python"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / binary).write_bytes(b"")
    (env_dir / "pyvenv.cfg").write_text(f"home = {home}\nversion_info = 3.11\n", encoding="utf-8")


def test_parse_args_defaults(data_dir: Path) -> None:
    args = provision.parse_args(["--data-dir", str(data_dir)])
    assert args.data_dir == data_dir
    assert args.tier == "essential"
    assert args.tensorrt == "auto"
    assert args.force is False


def test_progress_lines_are_machine_readable(capsys) -> None:
    tracker = provision.LinePrinter()
    tracker.start("runtime", "Installing")
    tracker.phase("dependencies")
    tracker.line("Downloading torch")

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("PROGRESS ")]
    assert lines, "every update must reach the caller"
    percent, phase, message = lines[-1].split("|")
    assert percent.replace(".", "").isdigit()
    assert phase == "dependencies"
    assert message == "Downloading torch"


def test_lock_blocks_a_second_provisioner(data_dir: Path) -> None:
    first = provision.acquire_lock(data_dir)
    assert first is None, "the first caller takes the lock"
    holder = provision.acquire_lock(data_dir)
    assert holder == os.getpid(), "a live holder is reported, not overwritten"
    provision.release_lock(data_dir)
    assert provision.acquire_lock(data_dir) is None


def test_a_stale_lock_is_taken_over(data_dir: Path) -> None:
    (data_dir / provision.LOCK_NAME).write_text("999999999", encoding="utf-8")
    assert provision.acquire_lock(data_dir) is None


def test_marker_is_written_only_on_success(data_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(provision, "detect_gpu", lambda: GpuInfo())
    calls: list[str] = []

    def boom(*_args, **_kwargs):
        calls.append("runtime")
        raise SetupError("venv failed (exit 2): Access is denied. (os error 5)")

    monkeypatch.setattr(provision.runtime, "create_venv", boom)
    args = provision.parse_args(["--data-dir", str(data_dir)])

    assert provision.main(["--data-dir", str(data_dir)]) == provision.EXIT_FAILED
    assert not setup_state.is_setup_complete(data_dir), "a failed run must not look complete"
    state = setup_state.load_state(data_dir)
    assert state.status("runtime") == "failed"
    assert "Access is denied" in state.errors["runtime"]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_desktop_provision.py -q`
Expected: FAIL — `No module named video_upscaler.desktop.provision`.

- [ ] **Step 4: Implement `src/video_upscaler/desktop/provision.py`**

```python
"""Headless provisioning driver — the only implementation of first-run setup.

Two callers, one contract:

* the NSIS installer (``hooks.nsh``) runs it at install time;
* the Tauri boot shell runs it when an installation is incomplete (repair).

It prints one machine-readable line per update::

    PROGRESS <percent>|<phase>|<message>

and writes ``.setup_complete`` only after every step succeeded, so both callers
can trust the marker. Exits 0 on success, 1 on failure after an ``ERROR <msg>``
line.

Runs on the uv-managed interpreter with nothing but the standard library and
this package — no venv is needed to start, because creating it is the job.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

if __package__ in (None, ""):  # invoked as a script by NSIS or the shell
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_upscaler.desktop import models as model_steps
from video_upscaler.desktop import runtime
from video_upscaler.desktop import state as setup_state
from video_upscaler.desktop.context import SetupContext, build_context
from video_upscaler.desktop.gpu import detect_gpu
from video_upscaler.desktop.progress import ProgressTracker

EXIT_OK = 0
EXIT_FAILED = 1
LOCK_NAME = ".provisioning.lock"
STEPS: tuple[str, ...] = ("gpu", "runtime", "models", "verify")


class StepFailure(RuntimeError):
    """A step failed; the message is safe to show the user."""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="clarity-provision")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--resources-dir", type=Path, default=None)
    parser.add_argument("--tier", choices=model_steps.TIERS, default=model_steps.DEFAULT_TIER)
    parser.add_argument("--tensorrt", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore setup.json and redo every step",
    )
    return parser.parse_args(argv)


class LinePrinter(ProgressTracker):
    """ProgressTracker that also emits the wire format both callers parse."""

    def _report(self, message: str = "") -> None:
        snapshot = self.snapshot()
        print(
            f"PROGRESS {snapshot['percent']}|{snapshot['phase']}|{message or snapshot.get('message', '')}",
            flush=True,
        )

    def start(self, step: str, message: Optional[str] = None) -> None:
        super().start(step, message)
        self._report(message or "")

    def phase(self, name: str) -> None:
        super().phase(name)
        self._report()

    def line(self, text: str) -> None:
        super().line(text)
        if text:
            self._report(text)

    def finish(self, error: Optional[str] = None) -> None:
        super().finish(error)
        self._report(error or "")


def _pid_alive(pid: int) -> bool:
    """Liveness check. ``os.kill(pid, 0)`` is NOT safe on Windows: for any
    signal other than CTRL_C_EVENT/CTRL_BREAK_EVENT Python calls
    TerminateProcess, which would kill the installer."""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return True  # assume alive: refusing to start beats double-provisioning
    return str(pid) in (completed.stdout or "")


def acquire_lock(data_dir: Path) -> Optional[int]:
    """Take the provisioning lock; returns a live holder's PID, else None."""
    data_dir.mkdir(parents=True, exist_ok=True)
    lock = data_dir / LOCK_NAME
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = -1
        if _pid_alive(pid):
            return pid
    try:
        lock.write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        print(f"ERROR Could not take the provisioning lock: {exc}", flush=True)
        return os.getpid()
    return None


def release_lock(data_dir: Path) -> None:
    try:
        (data_dir / LOCK_NAME).unlink(missing_ok=True)
    except OSError:
        pass
```

```python
def _step(
    context: SetupContext,
    state: setup_state.SetupState,
    tracker: ProgressTracker,
    name: str,
    work: Callable[[], None],
) -> None:
    """Run one step, persisting its status before and after.

    The state file is written first so a killed run resumes instead of starting
    over — a partially downloaded model set must not be fetched twice.
    """
    state.mark(name, setup_state.RUNNING)
    setup_state.save_state(context.data_dir, state)
    tracker.start(name)
    try:
        work()
    except Exception as exc:  # noqa: BLE001 - reported to the caller, then failed
        message = str(exc) or exc.__class__.__name__
        state.mark(name, setup_state.FAILED, message)
        setup_state.save_state(context.data_dir, state)
        tracker.finish(message)
        raise StepFailure(f"{name} failed: {message}") from exc
    state.mark(name, setup_state.DONE)
    setup_state.save_state(context.data_dir, state)


def run_steps(
    context: SetupContext,
    state: setup_state.SetupState,
    tracker: ProgressTracker,
    args: argparse.Namespace,
) -> None:
    if args.force:
        state.reset_from(STEPS[0])

    if not state.is_done("gpu"):
        _step(context, state, tracker, "gpu", lambda: _step_gpu(state, args))
    if not state.is_done("runtime"):
        _step(
            context, state, tracker, "runtime",
            lambda: runtime.install_runtime(context, tracker, state),
        )
    if not state.is_done("models"):
        _step(
            context, state, tracker, "models",
            lambda: model_steps.install_models(context, tracker, args.tier),
        )
    if not state.is_done("verify"):
        _step(context, state, tracker, "verify", lambda: _step_verify(context, state, tracker))

    state.models_tier = args.tier
    setup_state.save_state(context.data_dir, state)


def _step_gpu(state: setup_state.SetupState, args: argparse.Namespace) -> None:
    info = detect_gpu()
    state.torch_variant = info.torch_variant
    state.gpu_names = list(info.names)
    state.tensorrt = args.tensorrt == "yes" or (args.tensorrt == "auto" and info.has_nvidia)
    if info.error:
        print(f"PROGRESS 0|gpu|{info.error}", flush=True)


def _step_verify(
    context: SetupContext,
    state: setup_state.SetupState,
    tracker: ProgressTracker,
) -> None:
    report = runtime.verify_runtime(context, tracker)
    state.backend = str(report.get("backend", ""))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    context = build_context(data_dir=args.data_dir, resources_dir=args.resources_dir)
    context.ensure_directories()
    state = setup_state.load_state(context.data_dir)
    tracker = LinePrinter()

    holder = acquire_lock(context.data_dir)
    if holder is not None:
        print(
            f"ERROR Another Clarity process (PID {holder}) is already provisioning "
            f"{context.data_dir}. Close it and try again.",
            flush=True,
        )
        return EXIT_FAILED

    try:
        run_steps(context, state, tracker, args)
    except StepFailure as exc:
        print(f"ERROR {exc}", flush=True)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR {exc.__class__.__name__}: {exc}", flush=True)
        return EXIT_FAILED
    finally:
        release_lock(context.data_dir)

    # The marker is the only thing either caller trusts, so it is written last.
    state.mark("complete", setup_state.DONE)
    state.completed_at = setup_state.timestamp()
    setup_state.save_state(context.data_dir, state)
    setup_state.write_marker(context.data_dir)
    tracker.start("complete", "Clarity is ready")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

`runtime.install_runtime(context, tracker, state)` is a thin wrapper added in Task 4 that calls `create_venv` then `install_dependencies` with the TensorRT decision; until then, define it at the bottom of `runtime.py` as:

```python
def install_runtime(context: SetupContext, tracker: ProgressTracker, state) -> None:
    """Venv plus dependencies, honouring the TensorRT decision in ``state``."""
    create_venv(context, tracker)
    install_dependencies(context, tracker, state.torch_variant or "cpu")
```

- [ ] **Step 5: Add the success, resume and TensorRT-flag tests**

Append to `tests/test_desktop_provision.py`:

```python
def _fake_steps(monkeypatch, backend: str = "torch-cuda") -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(provision, "detect_gpu", lambda: GpuInfo(has_nvidia=True, names=["RTX 3050"]))
    monkeypatch.setattr(
        provision.runtime, "install_runtime",
        lambda *a, **k: calls.append("runtime"),
    )
    monkeypatch.setattr(
        provision.model_steps, "install_models",
        lambda *a, **k: calls.append("models"),
    )
    monkeypatch.setattr(
        provision.runtime, "verify_runtime",
        lambda *a, **k: (calls.append("verify"), {"backend": backend})[1],
    )
    return calls


def test_a_successful_run_writes_the_marker_last(data_dir: Path, monkeypatch) -> None:
    _fake_steps(monkeypatch, backend="tensorrt")
    write_fake_env(data_dir / "env", data_dir / "python")

    assert provision.main(["--data-dir", str(data_dir)]) == provision.EXIT_OK

    assert setup_state.is_setup_complete(data_dir)
    state = setup_state.load_state(data_dir)
    assert state.status("verify") == "done"
    assert state.status("complete") == "done"
    assert state.torch_variant == "cu126"
    assert state.tensorrt is True, "auto means yes when NVIDIA is present"
    assert state.backend == "tensorrt"
    assert not (data_dir / provision.LOCK_NAME).exists(), "the lock must not survive"


def test_tensorrt_no_is_respected_on_an_nvidia_machine(data_dir: Path, monkeypatch) -> None:
    _fake_steps(monkeypatch)
    write_fake_env(data_dir / "env", data_dir / "python")

    provision.main(["--data-dir", str(data_dir), "--tensorrt", "no"])

    assert setup_state.load_state(data_dir).tensorrt is False


def test_finished_steps_are_not_repeated(data_dir: Path, monkeypatch) -> None:
    """A killed download must resume, not start over."""
    calls = _fake_steps(monkeypatch)
    write_fake_env(data_dir / "env", data_dir / "python")
    state = setup_state.load_state(data_dir)
    state.mark("gpu", setup_state.DONE)
    state.mark("runtime", setup_state.DONE)
    setup_state.save_state(data_dir, state)

    assert provision.main(["--data-dir", str(data_dir)]) == provision.EXIT_OK
    assert calls == ["models", "verify"]


def test_force_redoes_every_step(data_dir: Path, monkeypatch) -> None:
    calls = _fake_steps(monkeypatch)
    write_fake_env(data_dir / "env", data_dir / "python")
    state = setup_state.load_state(data_dir)
    for name in ("gpu", "runtime", "models", "verify"):
        state.mark(name, setup_state.DONE)
    setup_state.save_state(data_dir, state)

    provision.main(["--data-dir", str(data_dir), "--force"])

    assert calls == ["runtime", "models", "verify"], "gpu is re-detected but not in `calls`"
```

- [ ] **Step 6: Move the surviving runtime/context tests**

Move these from `tests/test_desktop_wizard.py` into `tests/test_desktop_provision.py` unchanged (they cover `runtime.py`/`context.py`, which survive): `test_an_environment_that_still_works_is_adopted_rather_than_deleted` (drop its HTTP wrapper — call `runtime.create_venv(base, ProgressTracker(), probe=lambda env_dir: True)` directly), `test_an_unusable_env_that_cannot_be_deleted_names_the_running_process`, `test_the_probe_asks_the_interpreter_itself`, `test_the_probe_is_false_when_there_is_nothing_to_ask`, `test_dependency_install_refuses_an_env_without_an_interpreter`, `test_windows_access_denied_is_retried_once_and_explains_itself`, `test_managed_interpreter_lookup_skips_uv_staging_dirs`, and every `canonical_executable`/`venv_is_ours`/`venv_home` test. Add the `base` fixture they need:

```python
@pytest.fixture()
def base(data_dir: Path, tmp_path: Path) -> SetupContext:
    resources = tmp_path / "app"
    resources.mkdir()
    return build_context(data_dir=data_dir, resources_dir=resources)
```

- [ ] **Step 7: Run everything**

Run: `.venv/Scripts/python.exe -m pytest tests/test_desktop_provision.py -q` then `-m pytest -q --ignore=tests/e2e`
Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/video_upscaler/desktop/provision.py src/video_upscaler/desktop/state.py src/video_upscaler/desktop/runtime.py tests/test_desktop_provision.py tests/test_desktop_wizard.py
git commit -m "feat(desktop): add the headless provisioner both installer and shell call"
```

---

### Task 4: TensorRT is installed, and the result is reported

**Files:**
- Modify: `src/video_upscaler/desktop/runtime.py` (`install_dependencies`, `install_runtime`, `verify_runtime`, new `_first_field`)
- Test: `tests/test_desktop_provision.py`

**Interfaces:**
- Produces: `runtime.install_dependencies(context, tracker, torch_variant, tensorrt=False) -> None`, `runtime.install_runtime(context, tracker, state) -> None`, `runtime.verify_runtime(context, tracker) -> dict` with a new `"backend"` key.
- Consumes: `state.tensorrt` (Task 3), `video_upscaler.backend.detect_backend` (already in the app; it returns `tensorrt` only when `import tensorrt` succeeds).

**Why this is a bug, not a feature:** `backend.py:detect_backend()` and `interp.py:select_amt_backend()` already choose TensorRT automatically when the package imports. Neither the installer hook nor the wizard ever installed the `tensorrt` extra from `pyproject.toml`, so every desktop install silently ran `torch-cuda` — on a machine whose GPU was detected correctly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_desktop_provision.py` (add `from video_upscaler.desktop import runtime` to its imports):

```python
def _record_commands(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def streamed(command, tracker, **kwargs):
        calls.append([str(part) for part in command])
        return "ok"

    monkeypatch.setattr(runtime, "run_streamed", streamed)
    return calls


def test_the_tensorrt_extra_is_requested_when_chosen(base, monkeypatch) -> None:
    write_fake_env(base.env_dir, base.python_install_dir)
    calls = _record_commands(monkeypatch)

    runtime.install_dependencies(base, ProgressTracker(), "cu126", tensorrt=True)

    assert any(part.endswith("[tensorrt]") for part in calls[-1])
    assert "--extra-index-url" in calls[-1]


def test_tensorrt_is_not_requested_on_a_machine_without_nvidia(base, monkeypatch) -> None:
    write_fake_env(base.env_dir, base.python_install_dir)
    calls = _record_commands(monkeypatch)

    runtime.install_dependencies(base, ProgressTracker(), "cpu", tensorrt=False)

    assert not any("tensorrt" in part for part in calls[-1])


def test_a_failed_tensorrt_install_falls_back_to_cuda(base, monkeypatch) -> None:
    """TensorRT is an optimisation: a missing wheel must not brick the install."""
    write_fake_env(base.env_dir, base.python_install_dir)
    attempts: list[list[str]] = []

    def streamed(command, tracker, **kwargs):
        argv = [str(part) for part in command]
        attempts.append(argv)
        if any(part.endswith("[tensorrt]") for part in argv):
            raise SetupError("dependency install failed (exit 1): no solution for tensorrt")
        return "ok"

    monkeypatch.setattr(runtime, "run_streamed", streamed)
    runtime.install_dependencies(base, ProgressTracker(), "cu126", tensorrt=True)

    assert len(attempts) == 2
    assert not any(part.endswith("[tensorrt]") for part in attempts[-1])


def test_verify_runtime_reports_the_active_backend(base, monkeypatch) -> None:
    write_fake_env(base.env_dir, base.python_install_dir)
    monkeypatch.setattr(
        runtime, "run_streamed",
        lambda *a, **k: "imports ok 2.7.0 cuda True\nbackend tensorrt\n",
    )

    report = runtime.verify_runtime(base, ProgressTracker())

    assert report["backend"] == "tensorrt"
    assert report["cuda"] is True
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_desktop_provision.py -q -k tensorrt`
Expected: FAIL — `install_dependencies() got an unexpected keyword argument 'tensorrt'`.

- [ ] **Step 3: Implement**

In `src/video_upscaler/desktop/runtime.py` replace the tail of `install_dependencies`:

```python
def install_dependencies(
    context: SetupContext,
    tracker: ProgressTracker,
    torch_variant: str,
    tensorrt: bool = False,
) -> None:
    """Install the project and its dependencies into the provisioned venv."""
    # uv will create an environment for itself if the interpreter is missing;
    # that is how a failed venv step used to end up with a roaming interpreter.
    _require_venv_interpreter(context)

    base = [
        str(context.uv_exe),
        "pip",
        "install",
        "--python",
        str(context.venv_python),
    ]
    if torch_variant == "cu126":
        base += ["--extra-index-url", PYTORCH_CU126_INDEX]

    tracker.phase("dependencies")
    if tensorrt:
        tracker.line(
            f"Installing Clarity, PyTorch ({torch_variant}) and TensorRT — the long step…"
        )
        try:
            run_streamed(
                base + ["-e", f"{context.app_dir}[tensorrt]"],
                tracker,
                context=context,
                stage="dependency install",
            )
            return
        except SetupError as exc:
            tracker.line(f"TensorRT could not be installed ({exc}). Continuing with CUDA.")
    else:
        tracker.line(f"Installing Clarity and PyTorch ({torch_variant}) — this is the long step…")

    run_streamed(
        base + ["-e", str(context.app_dir)],
        tracker,
        context=context,
        stage="dependency install",
    )


def install_runtime(context: SetupContext, tracker: ProgressTracker, state) -> None:
    """Venv plus dependencies, honouring the TensorRT decision in ``state``."""
    create_venv(context, tracker)
    install_dependencies(
        context, tracker, state.torch_variant or "cpu", tensorrt=bool(state.tensorrt)
    )
```

In `verify_runtime`, extend the probe and the result:

```python
    code = (
        f"import {VERIFY_IMPORTS}\n"
        "import torch\n"
        "from video_upscaler.backend import detect_backend\n"
        "print('imports ok', torch.__version__, 'cuda', torch.cuda.is_available())\n"
        "print('backend', detect_backend())\n"
    )
```

```python
    return {
        "ok": True,
        "output": output.strip(),
        "cuda": "cuda True" in output,
        "torch_version": _first_version(output),
        "backend": _first_field(output, "backend"),
    }


def _first_field(output: str, key: str) -> str:
    """Value printed as ``key value`` by the verification probe, else ''."""
    for line in output.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0] == key:
            return parts[1]
    return ""
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_desktop_provision.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_upscaler/desktop/runtime.py tests/test_desktop_provision.py
git commit -m "feat(desktop): install the TensorRT extra on NVIDIA and report the backend"
```

---

### Task 5: The shell drives the provisioner

**Files:**
- Modify: `src-tauri/src/process.rs` (`build_bootstrap_command` → `build_provision_command`)
- Modify: `src-tauri/src/setup.rs` (add `parse_provision_line`, `run_provisioner`)
- Modify: `src-tauri/src/boot.rs` (rewrite `provision`; delete `wait_for_wizard`, `WizardStatus`, the `WIZARD_*` constants)
- Test: `src-tauri/src/setup.rs` and `src-tauri/src/process.rs` (`mod tests`)

**Interfaces:**
- Produces: `process::build_provision_command(python_bin: &Path, app_data_dir: &Path, resources_dir: &Path, tier: &str, tensorrt: &str) -> Result<tokio::process::Command, String>`, `setup::parse_provision_line(line: &str) -> Option<SetupProgressEvent>`, `setup::run_provisioner(cmd: tokio::process::Command, log_path: &Path, on_event: impl Fn(SetupProgressEvent) + Send + Sync + 'static) -> Result<(), String>`.
- Consumes: `setup::ensure_python_runtime(...) -> Result<PathBuf, String>` (unchanged), `SetupProgressEvent { stage, percent, speed, message }`, `process::ChildJob::attach`.

- [ ] **Step 1: Write the failing tests**

In `src-tauri/src/setup.rs` `mod tests`:

```rust
    #[test]
    fn test_parse_provision_line() {
        let event = parse_provision_line("PROGRESS 42.5|dependencies|Downloading torch").unwrap();
        assert_eq!(event.percent, 42.5);
        assert_eq!(event.stage, "dependencies");
        assert_eq!(event.message, "Downloading torch");
        assert!(event.speed.is_empty());
    }

    #[test]
    fn test_parse_provision_line_ignores_noise() {
        // uv's own output is interleaved with the wire format; only the wire
        // format carries progress, and everything else must not be mistaken for it.
        assert!(parse_provision_line("Resolved 41 packages").is_none());
        assert!(parse_provision_line("PROGRESS not-a-number|x|y").is_none());
        assert!(parse_provision_line("").is_none());
    }

    #[test]
    fn test_parse_provision_line_tolerates_a_missing_message() {
        let event = parse_provision_line("PROGRESS 7|gpu|").unwrap();
        assert_eq!(event.percent, 7.0);
        assert_eq!(event.stage, "gpu");
        assert_eq!(event.message, "");
    }
```

In `src-tauri/src/process.rs` `mod tests`:

```rust
    #[test]
    fn test_build_provision_command_targets_the_script_and_pins_the_data_dir() {
        let root = std::env::temp_dir().join("clarity_provision_cmd");
        let _ = std::fs::remove_dir_all(&root);
        let resources = root.join("resources");
        std::fs::create_dir_all(resources.join("src").join("video_upscaler").join("desktop")).unwrap();
        std::fs::write(
            resources.join("src").join("video_upscaler").join("desktop").join("provision.py"),
            b"",
        ).unwrap();
        let data = root.join("data");

        let cmd = build_provision_command(
            Path::new("python.exe"), &data, &resources, "essential", "auto",
        ).expect("command");
        let argv: Vec<String> = cmd.as_std().get_args().map(|a| a.to_string_lossy().to_string()).collect();

        assert_eq!(argv[0], "-u");
        assert!(argv[1].ends_with("provision.py"));
        assert!(argv.contains(&"--data-dir".to_string()));
        assert!(argv.contains(&data.to_string_lossy().to_string()));
        assert!(argv.contains(&"--tensorrt".to_string()));
        assert!(argv.contains(&"auto".to_string()));
        assert_eq!(
            cmd.as_std().get_envs().find(|(k, _)| *k == "CLARITY_DATA_DIR").unwrap().1.unwrap(),
            std::ffi::OsStr::new(data.as_os_str())
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_build_provision_command_refuses_a_missing_script() {
        let root = std::env::temp_dir().join("clarity_provision_cmd_missing");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let err = build_provision_command(
            Path::new("python.exe"), &root, &root, "essential", "auto",
        ).unwrap_err();

        assert!(err.contains("provision.py"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd src-tauri && cargo test parse_provision` and `cargo test build_provision_command`
Expected: FAIL — both functions are undefined.

- [ ] **Step 3: Implement `build_provision_command`**

In `src-tauri/src/process.rs`, replace `build_bootstrap_command` with:

```rust
/// Builds the command that provisions the AI engine.
///
/// The installer runs the exact same script with the exact same arguments, so
/// there is one implementation of "install Clarity" and no way for the two to
/// disagree about what a complete install contains.
pub fn build_provision_command(
    python_bin: &Path,
    app_data_dir: &Path,
    resources_dir: &Path,
    tier: &str,
    tensorrt: &str,
) -> Result<tokio::process::Command, String> {
    let script = resources_dir
        .join("src")
        .join("video_upscaler")
        .join("desktop")
        .join("provision.py");
    if !script.is_file() {
        return Err(format!(
            "The provisioning script is missing: {}. The installation is incomplete — reinstall Clarity.",
            script.display()
        ));
    }

    let mut cmd = tokio::process::Command::new(python_bin);
    cmd.args([
        "-u",
        &script.to_string_lossy(),
        "--data-dir",
        &app_data_dir.to_string_lossy(),
        "--resources-dir",
        &resources_dir.to_string_lossy(),
        "--tier",
        tier,
        "--tensorrt",
        tensorrt,
    ]);

    // PYTHONPATH so `import video_upscaler` resolves from the staged sources.
    cmd.env("PYTHONPATH", resources_dir.join("src"));
    cmd.env("PYTHONUNBUFFERED", "1");
    cmd.env("CLARITY_DATA_DIR", app_data_dir);
    cmd.env("CLARITY_DESKTOP_MODE", "1");
    Ok(cmd)
}
```

Unlike the old bootstrap command this does **not** redirect stdout to a file: `run_provisioner` reads the stream for progress and writes the log itself.

- [ ] **Step 4: Implement `parse_provision_line` and `run_provisioner`**

In `src-tauri/src/setup.rs`:

```rust
/// Parses the provisioner's wire format: `PROGRESS <percent>|<phase>|<message>`.
pub fn parse_provision_line(line: &str) -> Option<SetupProgressEvent> {
    let rest = line.trim().strip_prefix("PROGRESS ")?;
    let mut parts = rest.splitn(3, '|');
    let percent = parts.next()?.trim().parse::<f32>().ok()?;
    let stage = parts.next().unwrap_or("").trim().to_string();
    let message = parts.next().unwrap_or("").trim().to_string();
    Some(SetupProgressEvent { stage, percent, speed: String::new(), message })
}

/// Runs the headless provisioner, mirroring its output to a log and forwarding
/// progress to the shell.
///
/// Kept apart from `run_command_with_progress` on purpose: that one scrapes
/// percentages out of uv's human-readable output, this one reads a wire format
/// we control. Sharing a parser would couple the interpreter install to the
/// engine install for no benefit.
pub async fn run_provisioner(
    mut cmd: tokio::process::Command,
    log_path: &Path,
    on_event: impl Fn(SetupProgressEvent) + Send + Sync + 'static,
) -> Result<(), String> {
    use tokio::io::AsyncBufReadExt;

    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| format!("Cannot write {}: {e}", log_path.display()))?;
    let log = Arc::new(Mutex::new(log));
    let tail: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::with_capacity(16)));

    #[cfg(windows)]
    {
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start provisioning: {e}"))?;

    let on_event = Arc::new(on_event);
    let mut readers = Vec::new();
    for pipe in [child.stdout.take(), child.stderr.take()].into_iter().flatten() {
        let cb = on_event.clone();
        let log = log.clone();
        let tail = tail.clone();
        readers.push(tokio::spawn(async move {
            use std::io::Write;
            let mut lines = tokio::io::BufReader::new(pipe).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(mut file) = log.lock() {
                    let _ = writeln!(file, "{line}");
                }
                if let Ok(mut tail) = log_tail_lock(&tail) {
                    if tail.len() == 16 {
                        tail.pop_front();
                    }
                    tail.push_back(line.clone());
                }
                match parse_provision_line(&line) {
                    Some(event) => cb(event),
                    None => eprintln!("[clarity:provision] {line}"),
                }
            }
        }));
    }

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Provisioning process error: {e}"))?;
    for reader in readers {
        let _ = reader.await;
    }
    if status.success() {
        return Ok(());
    }

    // Prefer the provisioner's own ERROR line: it is written for a user to read.
    let collected: Vec<String> = log_tail_lock(&tail)
        .map(|t| t.iter().cloned().collect())
        .unwrap_or_default();
    let detail = collected
        .iter()
        .rev()
        .find(|line| line.starts_with("ERROR "))
        .map(|line| line.trim_start_matches("ERROR ").trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| collected.iter().rev().take(6).rev().cloned().collect::<Vec<_>>().join(" | "));
    Err(if detail.trim().is_empty() {
        format!("Provisioning stopped with {status}")
    } else {
        detail
    })
}

/// Poisoned-mutex-tolerant lock, so a panic in a reader cannot hang startup.
fn log_tail_lock(tail: &Arc<Mutex<VecDeque<String>>>) -> Result<std::sync::MutexGuard<'_, VecDeque<String>>, ()> {
    tail.lock().map_err(|_| ())
}
```

Add `use std::os::windows::process::CommandExt;` inside the existing `#[cfg(windows)]` import block if `creation_flags` is not already in scope (it is, for `run_command_with_progress`).

- [ ] **Step 5: Run the Rust tests**

Run: `cd src-tauri && cargo test`
Expected: PASS.

- [ ] **Step 6: Rewrite `boot.rs::provision` and delete the wizard plumbing**

Delete from `src-tauri/src/boot.rs`: `WIZARD_PORT_START`, `WIZARD_READY_TIMEOUT_SECS`, `WIZARD_POLL_INTERVAL`, the `WizardStatus` struct, `wait_for_wizard`, and the whole wizard half of `provision`. Replace `provision` with:

```rust
const BOOT_PROGRESS_EVENT: &str = "boot-progress";

/// Progress for the shell's bar. Deliberately a separate event from
/// `boot-status`, which carries one-line human text.
#[derive(Clone, serde::Serialize)]
struct ProgressPayload {
    percent: f32,
    phase: String,
    message: String,
}

/// Install the AI engine, or repair an installation that never finished.
///
/// The installer already ran this same script; reaching here means it did not
/// complete (offline install, antivirus, killed download). Running it again
/// resumes from `setup.json` and adopts an environment that already works, so a
/// repair never re-downloads what is on disk.
async fn provision(app: &AppHandle) -> Result<(), String> {
    let (data_dir, resources_dir) = {
        let state = app.state::<AppState>();
        (
            state.app_data_dir().to_path_buf(),
            state.resources_dir().to_path_buf(),
        )
    };

    report(app, "Installing the Python runtime…");
    let handle = app.clone();
    let python = setup::ensure_python_runtime(&data_dir, &resources_dir, move |event| {
        let message = if event.speed.trim().is_empty() {
            event.message.clone()
        } else {
            format!("{} — {}", event.message, event.speed)
        };
        report(&handle, message);
    })
    .await
    .map_err(|err| format!("Could not install the Python runtime: {err}"))?;

    report(app, "Installing the AI engine…");
    let cmd = process::build_provision_command(&python, &data_dir, &resources_dir, "essential", "auto")?;
    let log_path = data_dir.join("logs").join("provision.log");
    let handle = app.clone();

    setup::run_provisioner(cmd, &log_path, move |event| {
        let _ = handle.emit(
            BOOT_PROGRESS_EVENT,
            ProgressPayload {
                percent: event.percent,
                phase: event.stage.clone(),
                message: event.message.clone(),
            },
        );
        if !event.message.trim().is_empty() {
            report(&handle, event.message.clone());
        }
    })
    .await
    .map_err(|err| format!("{err}\n\nLog: {}", log_path.display()))?;

    if !setup::is_setup_complete(&data_dir) {
        return Err(format!(
            "Provisioning finished but the runtime is incomplete. See {}",
            log_path.display()
        ));
    }
    Ok(())
}
```

`boot_sequence` keeps its shape (adoption from Task 2 → completeness check → `provision` → `launch_studio`), and the window now navigates exactly once, to the studio.

- [ ] **Step 7: Verify and commit**

Run: `cd src-tauri && cargo test && cargo clippy --all-targets`
Expected: green; clippy must not report the removed `reqwest` status-polling types as unused (delete `WizardStatus` and any now-unused `use` lines it needed).

```bash
git add src-tauri/src/boot.rs src-tauri/src/process.rs src-tauri/src/setup.rs
git commit -m "feat(desktop): drive the provisioner from the boot shell instead of a wizard server"
```

---

### Task 6: The boot shell shows repair progress

**Files:**
- Modify: `src-tauri/shell/index.html`, `src-tauri/shell/shell.js`
- Test: `tests/test_frontend_assets.py`

**Interfaces:**
- Consumes: `boot-progress` event with `{ percent, phase, message }` (Task 5), existing `boot-status` / `boot-error` events, `retry_boot` command.
- Produces: DOM ids `progress`, `progress-bar`, `progress-percent`, `progress-phase`, `progress-log`, and `window.__ClarityShell.showProgress(percent, phase, message)`.

- [ ] **Step 1: Write the failing asset test**

In `tests/test_frontend_assets.py`, extend the shell test:

```python
def test_desktop_shell_renders_provisioning_progress():
    html = _read(TAURI_DIR / "shell" / "index.html")
    js = _read(TAURI_DIR / "shell" / "shell.js")

    for element in ('id="progress"', 'id="progress-bar"', 'id="progress-percent"',
                    'id="progress-phase"', 'id="progress-log"'):
        assert element in html, f"shell must render {element}"

    assert "'boot-progress'" in js, "the shell must listen for progress events"
    assert "showProgress" in js
    # The shell still owns no product logic: no API calls, no fetch.
    assert "/api/" not in js
    assert "fetch(" not in js
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_frontend_assets.py -q -k progress`
Expected: FAIL — the ids do not exist yet.

- [ ] **Step 3: Add the markup**

In `src-tauri/shell/index.html`, after the `.status` paragraph and before the error section:

```html
    <section class="progress" id="progress" hidden>
      <div class="bar"><div class="fill" id="progress-bar"></div></div>
      <p class="row">
        <span id="progress-percent">0%</span>
        <span id="progress-phase"></span>
      </p>
      <pre id="progress-log" hidden></pre>
    </section>
```

And in the `<style>` block:

```css
    .progress { margin-top: 1.4rem; text-align: left; }
    .progress .bar {
      height: 6px; border-radius: 999px; overflow: hidden;
      background: var(--bg-surface); border: 1px solid var(--border);
    }
    .progress .fill {
      height: 100%; width: 0%;
      background: linear-gradient(90deg, var(--accent), #84b6cf);
      transition: width 0.35s ease;
    }
    .progress .row {
      display: flex; justify-content: space-between; gap: 0.75rem;
      margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-muted);
    }
    .progress pre {
      margin-top: 0.75rem; max-height: 8rem; overflow: auto;
      padding: 0.6rem 0.7rem; font-size: 0.68rem; line-height: 1.45;
      background: var(--bg-surface); border: 1px solid var(--border);
      border-radius: 0.5rem; color: var(--text-muted); white-space: pre-wrap;
      user-select: text;
    }
```

- [ ] **Step 4: Wire it in `shell.js`**

Add to the `dom` map:

```js
    progress: document.getElementById('progress'),
    progressBar: document.getElementById('progress-bar'),
    progressPercent: document.getElementById('progress-percent'),
    progressPhase: document.getElementById('progress-phase'),
    progressLog: document.getElementById('progress-log')
```

Add the handler (bounded, so a chatty uv cannot grow the DOM without limit):

```js
  const LOG_LIMIT = 12;
  const logLines = [];

  /**
   * Provisioning progress from Rust, which is parsing the provisioner's
   * `PROGRESS <percent>|<phase>|<message>` lines. The shell renders it; it never
   * decides what to install.
   */
  function showProgress(percent, phase, message) {
    const value = Number(percent);
    dom.progress.hidden = false;
    dom.progressBar.style.width = `${Number.isFinite(value) ? Math.min(100, Math.max(0, value)) : 0}%`;
    dom.progressPercent.textContent = `${Number.isFinite(value) ? value.toFixed(1) : '0'}%`;
    dom.progressPhase.textContent = phase || '';
    if (message) {
      logLines.push(message);
      while (logLines.length > LOG_LIMIT) logLines.shift();
      dom.progressLog.hidden = false;
      dom.progressLog.textContent = logLines.join('\n');
      dom.progressLog.scrollTop = dom.progressLog.scrollHeight;
    }
  }
```

Register it in `init()` beside the other listeners, hide the bar when an error or the studio takes over, and export it:

```js
    await listen('boot-progress', (payload) =>
      showProgress(payload && payload.percent, payload && payload.phase, payload && payload.message)
    );
```

```js
  // inside showError(), above the existing lines:
  dom.progress.hidden = true;
```

```js
  window.__ClarityShell = { setStatus, showError, showProgress, retry, hasIpc };
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_frontend_assets.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src-tauri/shell/index.html src-tauri/shell/shell.js tests/test_frontend_assets.py
git commit -m "feat(desktop): show provisioning progress in the boot shell"
```

---

### Task 7: Delete the wizard

**Files:**
- Delete: `src/video_upscaler/desktop/server.py`, `src/video_upscaler/desktop/bootstrap.py`, `src/video_upscaler/web/static/setup.html`, `src/video_upscaler/web/static/js/setup.js`, `tests/test_desktop_wizard.py`
- Rename: `tests/test_setup_wizard_js.js` → `tests/test_desktop_bridge.js` (keep only the app.js notification/dialog suites)
- Modify: `tests/test_frontend_assets.py`, `package.json` (test script name if it lists the file)

**Interfaces:**
- Consumes: nothing new — this task only removes.
- Produces: the guarantee that no `/api/setup/*` endpoint, wizard page or wizard IPC remains in the tree.

- [ ] **Step 1: Write the failing absence tests**

In `tests/test_frontend_assets.py`, replace the wizard tests (`test_setup_html_loads_the_wizard_script`, `test_setup_js_drives_the_http_wizard`) with:

```python
def test_the_desktop_wizard_is_gone():
    """Provisioning belongs to the installer and the boot shell.

    The wizard was a second implementation of first-run setup, served from a
    second origin on a second port. When it disagreed with the installer about
    where data lived, the models were downloaded twice.
    """
    assert not (STATIC_DIR / "setup.html").exists()
    assert not (STATIC_DIR / "js" / "setup.js").exists()
    assert not (DESKTOP_DIR / "server.py").exists()
    assert not (DESKTOP_DIR / "bootstrap.py").exists()
    assert not (DESKTOP_DIR / "location.py").exists()


def test_no_setup_api_remains_anywhere():
    for path in list(STATIC_DIR.rglob("*.js")) + list(STATIC_DIR.rglob("*.html")):
        content = _read(path)
        assert "/api/setup/" not in content, f"{path.name} still calls the wizard API"
    for path in (TAURI_DIR / "src").rglob("*.rs"):
        assert b"api/setup/" not in path.read_bytes(), f"{path.name} still polls the wizard"
```

Add `DESKTOP_DIR = REPO_ROOT / "src" / "video_upscaler" / "desktop"` next to the existing path constants.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_frontend_assets.py -q -k "wizard_is_gone or setup_api"`
Expected: FAIL — the files still exist.

- [ ] **Step 3: Delete**

```bash
git rm src/video_upscaler/desktop/server.py src/video_upscaler/desktop/bootstrap.py \
       src/video_upscaler/web/static/setup.html src/video_upscaler/web/static/js/setup.js \
       tests/test_desktop_wizard.py
git mv tests/test_setup_wizard_js.js tests/test_desktop_bridge.js
```

In `tests/test_desktop_bridge.js` delete the whole `describe('Setup Wizard (setup.js) Unit Tests', …)` block and its helpers (`setupJsPathForTests`, `runSetupJs`, `makeFetchStub` if now unused, the `setupMockEnvironment` element ids that only the wizard needed), keeping the `Desktop IPC & Notification Bridge (app.js)` suite. Update `package.json` if a script names the old file.

Check for stragglers:

```bash
grep -rn "desktop.bootstrap\|desktop.server\|setup.html\|js/setup.js\|api/setup" \
  src src-tauri/src src-tauri/shell tests tools package.json
```

Expected: no hits outside this plan and the docs.

- [ ] **Step 4: Run everything**

Run: `.venv/Scripts/python.exe -m pytest -q --ignore=tests/e2e`
Run: `node --test tests/test_desktop_bridge.js`
Run: `cd src-tauri && cargo test && cargo clippy --all-targets`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add -A src/video_upscaler src-tauri tests package.json
git commit -m "refactor(desktop)!: remove the first-run wizard; the installer and boot shell own setup"
```

---

### Task 8: The installer provisions into the folder it installed to

**Files:**
- Create: `tools/verify_install_layout.ps1`
- Rewrite: `src-tauri/hooks.nsh`

**Interfaces:**
- Consumes: `provision.py` (Task 3/4) at `$INSTDIR\resources\src\video_upscaler\desktop\provision.py`, `resources\uv.exe`.
- Produces: `$DataDir` (NSIS var) equal to what `AppState::default_app_data_dir()` computes at runtime.

- [ ] **Step 1: Write the layout checker first**

Create `tools/verify_install_layout.ps1`:

```powershell
<#
.SYNOPSIS
    Asserts the single-directory contract after a sandbox install or uninstall.
.EXAMPLE
    tools\verify_install_layout.ps1 -InstallDir D:\clarity-sbx -ExpectTensorrt
    tools\verify_install_layout.ps1 -InstallDir D:\clarity-sbx -Uninstalled
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$InstallDir,
    [switch]$Uninstalled,
    [switch]$ExpectCleanProfile,
    [switch]$ExpectTensorrt
)

$ErrorActionPreference = 'Stop'
$script:failures = @()

function Test-Condition([bool]$Ok, [string]$What) {
    if ($Ok) { Write-Host "  ok    $What" }
    else { Write-Host "  FAIL  $What"; $script:failures += $What }
}

$required = @('python', 'env', 'models', 'logs', 'input', 'output')
$marker   = Join-Path $InstallDir '.setup_complete'
$venvPy   = Join-Path $InstallDir 'env\Scripts\python.exe'
$drive    = (Split-Path -Qualifier $InstallDir)

if ($Uninstalled) {
    foreach ($dir in $required) {
        Test-Condition (-not (Test-Path (Join-Path $InstallDir $dir))) "removed $dir\"
    }
    Test-Condition (-not (Test-Path $marker)) 'removed .setup_complete'
    Test-Condition (-not (Test-Path (Join-Path $InstallDir 'clarity-desktop.exe'))) 'removed the program'
} else {
    foreach ($dir in $required) {
        Test-Condition (Test-Path (Join-Path $InstallDir $dir)) "exists $dir\"
    }
    Test-Condition (Test-Path $marker) 'marker written'
    Test-Condition (Test-Path $venvPy) 'venv interpreter present'
    Test-Condition (@(Get-ChildItem (Join-Path $InstallDir 'models') -File).Count -gt 0) 'models downloaded'

    # The whole point: nothing provisioned anywhere else.
    Test-Condition (-not (Test-Path (Join-Path $drive 'Clarity-data'))) 'no <drive>\Clarity-data sibling'
    if ($ExpectCleanProfile) {
        Test-Condition (-not (Test-Path "$env:LOCALAPPDATA\Clarity")) 'nothing in %LOCALAPPDATA%\Clarity'
    }

    # The venv must be based on the interpreter inside the install folder, not
    # on uv's roaming copy in %APPDATA%\uv.
    $home_line = (Get-Content (Join-Path $InstallDir 'env\pyvenv.cfg') | Where-Object { $_ -match '^home' }) -join ''
    Test-Condition ($home_line -like "*$InstallDir\python*") "pyvenv.cfg home is inside the install folder ($home_line)"

    if ($ExpectTensorrt) {
        Test-Condition (Test-Path (Join-Path $InstallDir 'env\Lib\site-packages\tensorrt')) 'tensorrt installed'
    }
}

if ($script:failures.Count -gt 0) {
    Write-Host "`n$($script:failures.Count) check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "`nAll layout checks passed" -ForegroundColor Green
exit 0
```

- [ ] **Step 2: Rewrite `src-tauri/hooks.nsh`**

```nsis
; Clarity Desktop NSIS Installer Hooks
;
; The installer owns first-run provisioning, and everything it creates lands in
; the folder the user just chose. The app computes the same location from its own
; executable path (AppState::default_app_data_dir), so the two cannot disagree —
; which is exactly what happened when the hook used $LOCALAPPDATA\Clarity while
; the app looked in <drive>\Clarity-data and downloaded the models a second time.

Var DataDir
Var TensorrtChoice

!macro NSIS_HOOK_PREINSTALL
  DetailPrint "Preparing Clarity installation..."
!macroend
```
```nsis
!macro NSIS_HOOK_POSTINSTALL
  ; ---- where everything lives ----------------------------------------------
  ; $INSTDIR is the folder the user chose on the directory page. It is writable
  ; in the normal case; a Program Files install is not, and must fall back
  ; rather than fail during a multi-gigabyte download.
  StrCpy $DataDir "$INSTDIR"
  FileOpen $0 "$INSTDIR\.clarity-write-test" w
  IfErrors 0 clarity_writable
    StrCpy $DataDir "$LOCALAPPDATA\Clarity"
    CreateDirectory "$DataDir"
    DetailPrint "[WARN] $INSTDIR is not writable; provisioning into $DataDir"
  clarity_writable:
  FileClose $0
  Delete "$INSTDIR\.clarity-write-test"

  IfFileExists "$INSTDIR\resources\uv.exe" clarity_have_uv 0
    DetailPrint "[WARN] resources\uv.exe is missing; cannot provision."
    Goto clarity_hook_done
  clarity_have_uv:

  ; ---- 1/4 Python runtime ---------------------------------------------------
  DetailPrint "[1/4] Installing the Python 3.11 runtime into $DataDir\python ..."
  System::Call 'kernel32::SetEnvironmentVariable(t "UV_PYTHON_INSTALL_DIR", t "$DataDir\python")'
  nsExec::ExecToLog '"$INSTDIR\resources\uv.exe" python install 3.11 --no-bin --install-dir "$DataDir\python"'
  Pop $0
  IntCmp $0 0 clarity_python_ok clarity_failed clarity_failed
  clarity_python_ok:

  ; ---- 2/4 TensorRT ---------------------------------------------------------
  ; The app picks TensorRT by itself whenever `import tensorrt` works, so this
  ; is only about whether the package gets installed at all.
  StrCpy $TensorrtChoice "no"
  nsExec::ExecToStack 'nvidia-smi -L'
  Pop $0
  Pop $1
  IntCmp $0 0 clarity_nvidia clarity_no_nvidia clarity_no_nvidia
  clarity_nvidia:
    DetailPrint "  NVIDIA GPU detected: $1"
    IfSilent clarity_trt_silent 0
      MessageBox MB_YESNO|MB_ICONQUESTION "Install TensorRT acceleration for your NVIDIA GPU?$\n$\nInterpolation runs considerably faster. Adds roughly 1-2 GB to this download." IDNO clarity_trt_no
      StrCpy $TensorrtChoice "yes"
      Goto clarity_no_nvidia
    clarity_trt_silent:
      ; A silent install is an unattended deployment: give it the acceleration.
      StrCpy $TensorrtChoice "yes"
    clarity_trt_no:
  clarity_no_nvidia:

  ; ---- 3/4 engine, dependencies, models -------------------------------------
  ; One implementation, shared with the app's repair path. `uv run --no-project`
  ; executes the script with the managed interpreter without creating an
  ; environment of its own; provision.py finds the package via its own path.
  DetailPrint "[2/4] Creating the environment and installing PyTorch (TensorRT: $TensorrtChoice) ..."
  DetailPrint "[3/4] Downloading the essential model weights ..."
  System::Call 'kernel32::SetEnvironmentVariable(t "PYTHONUNBUFFERED", t "1")'
  nsExec::ExecToLog '"$INSTDIR\resources\uv.exe" run --no-project --python 3.11 "$INSTDIR\resources\src\video_upscaler\desktop\provision.py" --data-dir "$DataDir" --resources-dir "$INSTDIR\resources" --tier essential --tensorrt $TensorrtChoice'
  Pop $0
  IntCmp $0 0 clarity_provisioned clarity_failed clarity_failed

  clarity_provisioned:
    ; 4/4 is the marker, and provision.py writes it — never this script. A hook
    ; that writes it unconditionally is how a failed install used to look
    ; complete, and how the app then skipped provisioning forever.
    DetailPrint "[4/4] Clarity AI engine ready in $DataDir"
    Goto clarity_hook_done

  clarity_failed:
    DetailPrint "[ERROR] The AI engine could not be installed (code $0)."
    DetailPrint "        Log: $DataDir\logs\provision.log"
    MessageBox MB_OK|MB_ICONSTOP "Clarity was installed, but its AI engine could not be set up (code $0).$\n$\nClarity will finish the setup when you next start it.$\nLog: $DataDir\logs\provision.log"
  clarity_hook_done:
!macroend
```

- [ ] **Step 3: Build and run a sandbox install**

```powershell
npm run desktop:prepare
npx tauri build
# /D must be the LAST argument and unquoted (NSIS rule).
Start-Process -Wait "src-tauri\target\release\bundle\nsis\Clarity_0.1.0_x64-setup.exe" -ArgumentList '/D=D:\clarity-sbx'
tools\verify_install_layout.ps1 -InstallDir D:\clarity-sbx -ExpectCleanProfile -ExpectTensorrt
```

Expected: every check `ok`. This downloads several gigabytes once; it is the only way to prove the installer path, and it is the bug this plan exists to fix.

Also confirm the negative case — no stray sibling folder and no second copy:

```powershell
Get-ChildItem D:\ | Where-Object Name -like 'Clarity*' | Select-Object Name
Get-ChildItem "$env:LOCALAPPDATA" | Where-Object Name -like 'Clarity*' | Select-Object Name
```

Expected: only `clarity-sbx`.

- [ ] **Step 4: Launch it, then commit**

Run `D:\clarity-sbx\clarity-desktop.exe`. Expected: the boot shell appears briefly and goes straight to the studio — no progress bar, no downloads, because `.setup_complete` is already there.

```bash
git add src-tauri/hooks.nsh tools/verify_install_layout.ps1
git commit -m "build(desktop): provision into the install folder from the installer"
```

---

### Task 9: Uninstall removes the data it created — and updates do not

**Files:**
- Modify: `src-tauri/hooks.nsh` (add `NSIS_HOOK_PREUNINSTALL`, `NSIS_HOOK_POSTUNINSTALL`)
- Test: `tools/verify_install_layout.ps1 -Uninstalled`

**Interfaces:**
- Consumes: template vars `$UpdateMode`, `$DeleteAppDataCheckboxState`, `$INSTDIR` (all defined by Tauri's `installer.nsi`; the template's own uninstall section uses them).

**Why this is required:** the template ends with `RMDir "$INSTDIR"` — non-recursive. With a flat data layout, uninstalling would leave `env\`, `python\` and `models\` behind: gigabytes of orphans, which is the complaint this plan exists to answer.

- [ ] **Step 1: Add the uninstall hooks**

```nsis
!macro NSIS_HOOK_PREUNINSTALL
  DetailPrint "Removing Clarity..."
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ; An update runs the uninstaller before installing the new files. Deleting the
  ; engine there would turn every update into a multi-gigabyte re-download.
  ${If} $UpdateMode = 1
    DetailPrint "Update mode: keeping the AI engine and models."
    Goto clarity_uninstall_done
  ${EndIf}

  ; Regenerable data. An installer can fetch all of this again, and leaving it
  ; behind is precisely the bloat the single-directory layout is meant to avoid.
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\env"
  RMDir /r "$INSTDIR\models"
  RMDir /r "$INSTDIR\tools"
  RMDir /r "$INSTDIR\.cache"
  RMDir /r "$INSTDIR\logs"
  Delete "$INSTDIR\setup.json"
  Delete "$INSTDIR\.setup_complete"
  Delete "$INSTDIR\.provisioning.lock"
  Delete "$INSTDIR\.clarity-write-test"

  ; User media is not ours to delete silently.
  ${If} $DeleteAppDataCheckboxState = 1
    RMDir /r "$INSTDIR\input"
    RMDir /r "$INSTDIR\output"
    RMDir /r "$LOCALAPPDATA\Clarity"
  ${Else}
    MessageBox MB_YESNO|MB_ICONQUESTION "Delete your Clarity videos as well?$\n$\n$INSTDIR\input$INSTDIR\output" IDNO clarity_keep_media
      RMDir /r "$INSTDIR\input"
      RMDir /r "$INSTDIR\output"
    clarity_keep_media:
  ${EndIf}

  RMDir "$INSTDIR"
  DetailPrint "Clarity data removed."
  clarity_uninstall_done:
!macroend
```

Note the two paths that must stay in sync with the layout in `context.py:ensure_directories()` — if a new subdirectory is ever added there, add its `RMDir /r` here too.

- [ ] **Step 2: Verify uninstall, and verify that an update keeps the data**

```powershell
# Put a sentinel in the media folder so we can see which answer the prompt got.
New-Item -ItemType Directory -Force D:\clarity-sbx\input | Out-Null
'mine' | Set-Content D:\clarity-sbx\input\keep.txt

Start-Process -Wait "D:\clarity-sbx\uninstall.exe"      # answer NO to the media question
tools\verify_install_layout.ps1 -InstallDir D:\clarity-sbx -Uninstalled
Test-Path D:\clarity-sbx\input\keep.txt                 # expected: True
```

Then reinstall over the top and prove an update does not re-download:

```powershell
$before = (Get-Item D:\clarity-sbx\models\amt-s.pth).LastWriteTime
Start-Process -Wait "src-tauri\target\release\bundle\nsis\Clarity_0.1.0_x64-setup.exe" -ArgumentList '/S','/D=D:\clarity-sbx'
$after = (Get-Item D:\clarity-sbx\models\amt-s.pth).LastWriteTime
"$before -> $after"    # expected: unchanged, and the install finished in seconds
```

Expected: `env\`, `python\`, `models\` gone after a real uninstall; media kept when the user says no; a silent reinstall over an existing install does not re-fetch the weights.

- [ ] **Step 3: Commit**

```bash
git add src-tauri/hooks.nsh
git commit -m "build(desktop): remove provisioned data on uninstall, keep it on update"
```

---

### Task 10: Documentation and full verification

**Files:**
- Modify: `docs/superpowers/specs/2026-09-14-tauri-windows-desktop-design.md` (replace the 2026-09-15 addendum)
- Modify: `README.md` if it documents the data location

- [ ] **Step 1: Replace the addendum**

Delete the section titled "Addendum (2026-09-15): where desktop data actually lives" — it describes the pointer-file and `Clarity-data` design that Tasks 1 and 2 remove — and replace it with:

```markdown
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
```

- [ ] **Step 2: Full verification**

```bash
.venv/Scripts/python.exe -m pytest -q --ignore=tests/e2e
node --test tests/test_desktop_bridge.js
cd src-tauri && cargo test && cargo clippy --all-targets
cd .. && npm run desktop:prepare && npx tauri build
tools/verify_install_layout.ps1 -InstallDir D:\clarity-sbx -ExpectCleanProfile -ExpectTensorrt
```

Expected: all suites green, clippy silent, installer built, layout checks `ok`.

- [ ] **Step 3: Migrate the machine this was reported on**

The install at `D:\Clarity` predates this change; `D:\Clarity-data` holds a
complete engine and `%LOCALAPPDATA%\Clarity` holds 4.5 GB of a half-finished one.

1. Install the new build over `D:\Clarity`. On first launch Task 2's adoption
   renames `D:\Clarity-data\*` into `D:\Clarity\` — same volume, instant, no
   re-download — and the marker makes the shell go straight to the studio.
2. Confirm: `tools\verify_install_layout.ps1 -InstallDir D:\Clarity -ExpectTensorrt`.
3. Ask before deleting the leftovers, then remove `C:\Users\Biboy\AppData\Local\Clarity`
   (env only, no models, no marker) and the emptied `D:\Clarity-data`.

- [ ] **Step 4: Commit**

```bash
git add docs/ README.md
git commit -m "docs(desktop): document the single-directory, installer-provisioned layout"
```

---

## Self-Review

**Requirement coverage** (each item from the report that produced this plan):

| Requirement | Task |
|---|---|
| One directory only — install to `E:\Clarity`, everything lives there | 1, 8 |
| Location decided by the installer, not by the app | 8 (`$INSTDIR` from `MUI_PAGE_DIRECTORY`) |
| No double download / no double file size | 1 (one rule), 2 (adopt instead of re-fetch), 8 (same rule in NSIS) |
| Remove the desktop setup process | 7 (delete wizard), 5–6 (shell-driven repair instead) |
| Works on any Windows machine, any install drive | 1 (rule + read-only fallback), 8 (same probe) |
| TensorRT on the detected NVIDIA GPU | 4 (extra + backend report), 8 (ask, default yes, silent = yes) |
| Uninstall must not leave gigabytes behind | 9 |
| Updates must not re-download | 9 (`$UpdateMode` guard, verified) |

**Placeholder scan:** none. Every code step contains the code to write; every test step contains the assertions; every run step names the command and the expected result.

**Type/name consistency:** `resolve_data_dir(override, install, local)` is used with three arguments everywhere; `build_provision_command(python, data, resources, tier, tensorrt)` matches its call in `boot.rs`; `run_provisioner(cmd, log_path, on_event)` matches; `runtime.install_runtime(context, tracker, state)` matches `provision.run_steps`; `state.tensorrt` / `state.backend` are added in Task 3 Step 1 and consumed in Task 4; `provision.LOCK_NAME`, `EXIT_OK`, `EXIT_FAILED`, `LinePrinter` are defined in Task 3 and referenced by its tests; `parse_provision_line -> Option<SetupProgressEvent>` matches `run_provisioner`'s use.

**Fixes to apply while executing** (found in review, so they are not forgotten):

1. `tests/test_desktop_provision.py` needs `from video_upscaler.desktop.context import SetupContext, build_context` for the `base` fixture added in Task 3 Step 6.
2. `src-tauri/src/process.rs` has an existing test for `build_bootstrap_command`; delete it in Task 5 (the function no longer exists) and make sure `use std::path::Path;` is in scope inside `mod tests`.
3. Task 5's `run_provisioner` uses `Arc`, `Mutex`, `VecDeque` — all already imported at the top of `setup.rs`; add `use std::io::Write;` only inside the reader task, as written.
4. `tests/test_frontend_assets.py` still contains `test_setup_html_loads_the_wizard_script`; Task 7 Step 1 replaces it — do not leave both.
5. `package.json` may reference `tests/test_setup_wizard_js.js` in a script; update it when Task 7 renames the file.

## Risks

- **Flat layout makes uninstall correctness load-bearing.** The template's `RMDir "$INSTDIR"` is non-recursive, so Task 9 is not optional; it is verified with `verify_install_layout.ps1 -Uninstalled`.
- **Update path.** If `$UpdateMode` is not set the way the template intends, an update would delete the engine. Task 9 Step 2 verifies a silent reinstall keeps the weights.
- **TensorRT size and availability.** The extra adds roughly 1–2 GB and its wheels can lag a driver release. A failed TensorRT install must degrade to CUDA with a warning, never fail the install (Task 4 Step 3).
- **Read-only installs** (Program Files, perMachine) fall back to `%LOCALAPPDATA%\Clarity`; both sides must probe identically or the app will look where the installer did not write.
- **Removing the wizard** deletes roughly 1.5k lines and their tests. Nothing else references them — the studio backend never served `setup.html` — but Task 7 Step 3 greps to prove it before committing.
- **`uv run --no-project`** must not create an environment inside the install folder. Task 8 Step 3's layout check catches it (`env\pyvenv.cfg` home must point inside `<install>\python`).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-15-desktop-single-directory-provisioning.md`. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session with checkpoints for review.

Tasks 1–7 are safe to run without touching a real install. Tasks 8–9 need a sandbox install on a spare directory (`D:\clarity-sbx`) and several gigabytes of download. Task 10 Step 3 touches the machine this was reported on and includes a deletion that must be confirmed first.
