//! Desktop boot orchestration.
//!
//! The shell has exactly three jobs: get an interpreter on disk, hand over to
//! the Python provisioning wizard, then launch the studio backend and point the
//! window at it. Every failure ends up as readable text on the boot shell —
//! previously it went to `stderr`, which a windowed Windows process has no way
//! to show, so a failed start looked like a frozen application.

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::commands::{self, navigate_window};
use crate::process;
use crate::setup;
use crate::state::AppState;

/// Event carrying a human-readable boot progress message.
pub const BOOT_STATUS_EVENT: &str = "boot-status";
/// Event carrying a fatal boot error.
pub const BOOT_ERROR_EVENT: &str = "boot-error";

const BACKEND_READY_TIMEOUT_SECS: u64 = 180;

#[derive(Clone, Serialize)]
struct StatusPayload {
    message: String,
}

#[derive(Clone, Serialize)]
struct ErrorPayload {
    message: String,
    details: String,
    log_dir: String,
}

fn report(app: &AppHandle, message: impl Into<String>) {
    let message = message.into();
    // Reaches the boot shell; also visible from `tauri dev`.
    println!("[clarity] {message}");
    let _ = app.emit(
        BOOT_STATUS_EVENT,
        StatusPayload {
            message,
        },
    );
}

/// Surfaces a fatal startup failure on the boot shell.
pub fn fail(app: &AppHandle, message: impl Into<String>) {
    let message = message.into();
    let log_dir = {
        let state = app.state::<AppState>();
        state.app_data_dir().join("logs").to_string_lossy().to_string()
    };
    eprintln!("[clarity] startup failed: {message}");

    let _ = app.emit(
        BOOT_ERROR_EVENT,
        ErrorPayload {
            details: message.clone(),
            log_dir,
            message,
        },
    );

    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Picks the launch path. Failures are reported on the shell, never swallowed.
pub async fn start(app: AppHandle) {
    if !app.state::<AppState>().try_begin_boot() {
        report(&app, "Startup is already running — please wait…");
        return;
    }

    let outcome = boot_sequence(&app).await;
    app.state::<AppState>().finish_boot();

    if let Err(err) = outcome {
        fail(&app, err);
    }
}

async fn boot_sequence(app: &AppHandle) -> Result<(), String> {
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

    let (provisioned, dev_flow) = {
        let state = app.state::<AppState>();
        (
            setup::is_setup_complete(state.app_data_dir()),
            setup::has_dev_environment(),
        )
    };

    // A repository checkout already has a working venv; provisioning there would
    // be wrong and slow, so the development flow launches directly.
    if provisioned || dev_flow {
        return launch_studio(app).await;
    }

    provision(app).await?;
    launch_studio(app).await
}

/// Re-run the whole boot sequence. Safe to call repeatedly: every step is
/// idempotent and the wizard resumes from `setup.json`.
pub async fn retry(app: AppHandle) {
    start(app).await;
}

async fn launch_studio(app: &AppHandle) -> Result<(), String> {
    report(app, "Starting the studio server…");

    let port = {
        let state = app.state::<AppState>();
        let port = commands::launch_backend_internal(&state, BACKEND_READY_TIMEOUT_SECS).await?;
        state.set_setup_complete(true);
        port
    };

    let url = format!("http://127.0.0.1:{port}/");
    report(app, "Opening Clarity Studio");
    navigate_window(app, &url)
        .map_err(|err| format!("The studio server is running, but the window could not open it: {err}"))
}

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
