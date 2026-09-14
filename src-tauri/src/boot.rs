//! Desktop boot orchestration.
//!
//! The shell has exactly three jobs: get an interpreter on disk, hand over to
//! the Python provisioning wizard, then launch the studio backend and point the
//! window at it. Every failure ends up as readable text on the boot shell —
//! previously it went to `stderr`, which a windowed Windows process has no way
//! to show, so a failed start looked like a frozen application.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

use crate::commands::{self, navigate_window};
use crate::process::{self, ChildJob};
use crate::setup;
use crate::state::AppState;

/// Event carrying a human-readable boot progress message.
pub const BOOT_STATUS_EVENT: &str = "boot-status";
/// Event carrying a fatal boot error.
pub const BOOT_ERROR_EVENT: &str = "boot-error";

const WIZARD_PORT_START: u16 = 8610;
const WIZARD_READY_TIMEOUT_SECS: u64 = 60;
const BACKEND_READY_TIMEOUT_SECS: u64 = 180;
const WIZARD_POLL_INTERVAL: Duration = Duration::from_millis(700);

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

/// Only the completion flag matters to the shell; the wizard owns the rest.
#[derive(Deserialize)]
struct WizardStatus {
    complete: bool,
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

/// Install an interpreter, run the provisioning wizard, wait for it to finish.
async fn provision(app: &AppHandle) -> Result<(), String> {
    let (data_dir, resources_dir) = {
        let state = app.state::<AppState>();
        (
            state.app_data_dir().to_path_buf(),
            state.resources_dir().to_path_buf(),
        )
    };
    let handle = app.clone();

    report(app, "Installing the Python runtime…");
    let python = setup::ensure_python_runtime(&data_dir, &resources_dir, move |event| {
        let message = if event.speed.trim().is_empty() {
            event.message
        } else {
            format!("{} — {}", event.message, event.speed)
        };
        report(&handle, message);
    })
    .await
    .map_err(|err| format!("Could not install the Python runtime: {err}"))?;

    report(app, "Preparing first-run setup…");
    let port = process::find_available_port(WIZARD_PORT_START, 50)
        .ok_or("No free port was available for the setup wizard")?;

    let mut cmd = process::build_bootstrap_command(&python, port, &data_dir, &resources_dir)?;
    let mut child = cmd
        .spawn()
        .map_err(|err| format!("Could not start the setup wizard: {err}"))?;
    // Bound for the whole function: guarantees the wizard dies with the shell.
    let _job = ChildJob::attach(&child)?;

    let base_url = format!("http://127.0.0.1:{port}/");
    wait_for_wizard(app, &base_url, port, &mut child).await?;

    // The wizard is the UI now: it reports its own progress over HTTP, so the
    // remote page needs no desktop IPC permission to work.
    navigate_window(app, &base_url)?;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|err| err.to_string())?;
    let status_url = format!("{base_url}api/setup/status");

    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "The setup wizard stopped unexpectedly ({status}). See {}",
                data_dir.join("logs").join("bootstrap.log").display()
            ));
        }

        match client.get(&status_url).send().await {
            Ok(response) if response.status().is_success() => match response
                .json::<WizardStatus>()
                .await
            {
                Ok(snapshot) if snapshot.complete => break,
                Ok(_) => {}
                Err(err) => eprintln!("[clarity] unreadable wizard status: {err}"),
            },
            Ok(response) => eprintln!("[clarity] wizard status returned {}", response.status()),
            Err(err) => eprintln!("[clarity] wizard status unavailable: {err}"),
        }

        tokio::time::sleep(WIZARD_POLL_INTERVAL).await;
    }

    let _ = child.start_kill();
    Ok(())
}

/// Wait for the wizard to answer, watching for a process that died at startup
/// (missing staged sources, unusable interpreter) instead of timing out in
/// silence and leaving the window on a blank page.
async fn wait_for_wizard(
    app: &AppHandle,
    base_url: &str,
    port: u16,
    child: &mut tokio::process::Child,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|err| err.to_string())?;
    let status_url = format!("{base_url}api/setup/status");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(WIZARD_READY_TIMEOUT_SECS);
    report(app, "Preparing the setup wizard…");

    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|err| format!("Failed to poll the setup wizard: {err}"))?
        {
            return Err(format!(
                "The setup wizard exited ({status}) before it started. Check the logs folder in your Clarity data directory."
            ));
        }

        if let Ok(response) = client.get(&status_url).send().await {
            if response.status().is_success() {
                return Ok(());
            }
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(format!(
                "The setup wizard did not answer on port {port} within {WIZARD_READY_TIMEOUT_SECS} seconds."
            ));
        }

        tokio::time::sleep(Duration::from_millis(400)).await;
    }
}
