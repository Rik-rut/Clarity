use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};

use crate::process::{self, BackendProcessManager};
use crate::setup;
use crate::state::AppState;

/// Configuration payload returned by `get_app_config`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfigResponse {
    pub app_data_dir: String,
    pub resources_dir: String,
    pub log_dir: String,
    pub port: u16,
    pub setup_complete: bool,
    pub backend_running: bool,
}

/// Navigates the main webview window to the specified URL or route.
pub fn navigate_window(app: &AppHandle, target: &str) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .or_else(|| app.webview_windows().into_values().next())
        .ok_or_else(|| "No webview window found".to_string())?;

    let res = if let Ok(url) = target.parse::<tauri::Url>() {
        window
            .navigate(url)
            .map_err(|e| format!("Navigation failed: {e}"))
    } else {
        let escaped = serde_json::to_string(target).unwrap_or_else(|_| format!("\"{}\"", target));
        let js = format!("window.location.href = {};", escaped);
        window
            .eval(&js)
            .map_err(|e| format!("Failed to evaluate navigation script: {e}"))
    };

    let _ = window.show();
    res
}

/// Pure implementation of app configuration query.
pub fn get_app_config_impl(state: &AppState) -> Result<AppConfigResponse, String> {
    // The wizard can move the data root before anything is provisioned, so read
    // the current answer instead of the value captured at startup.
    let data_dir = AppState::default_app_data_dir();
    let setup_complete = state.is_setup_complete() || setup::is_setup_complete(&data_dir);
    Ok(AppConfigResponse {
        app_data_dir: data_dir.to_string_lossy().to_string(),
        resources_dir: state.resources_dir().to_string_lossy().to_string(),
        log_dir: data_dir.join("logs").to_string_lossy().to_string(),
        port: state.port(),
        setup_complete,
        backend_running: state.is_backend_running(),
    })
}

/// Internal helper to launch the Python FastAPI backend process.
///
/// `ready_timeout_secs` is supplied by the caller: torch plus opencv import in
/// seconds on a cold install with an antivirus scanning them, and the old
/// fixed 30s budget reported a perfectly healthy server as a failure.
pub async fn launch_backend_internal(
    state: &AppState,
    ready_timeout_secs: u64,
) -> Result<u16, String> {
    if state.is_backend_running() {
        return Ok(state.port());
    }

    // Re-resolved: first-run provisioning may have moved the data root to the
    // drive the user picked, and the backend must see that, not the startup one.
    let data_dir = AppState::default_app_data_dir();

    // Stale-backend recovery: a previous shell may have died leaving its
    // backend behind, or the user may run their own server on the configured
    // port. Adopt a healthy backend, fail loudly on a foreign holder, else
    // fall through to the normal strict-port spawn path. Never kill the
    // holder: it may not be ours.
    let configured = state.port();
    if process::is_port_occupied(configured) {
        let health = process::probe_backend_health(configured).await;
        match process::classify_port_holder(true, || health) {
            process::PortHolderDecision::Adopt => {
                crate::boot::append_boot_log(
                    &data_dir,
                    &format!("adopted healthy backend on port {configured}"),
                );
                state.set_port(configured);
                return Ok(configured);
            }
            process::PortHolderDecision::OccupiedByForeign => {
                let msg = format!(
                    "Port {configured} is already in use by another application. \
                     Close it (or point Clarity at a free port) and retry."
                );
                crate::boot::append_boot_log(&data_dir, &msg);
                return Err(msg);
            }
            process::PortHolderDecision::Spawn => {
                crate::boot::append_boot_log(
                    &data_dir,
                    &format!(
                        "port {configured} occupied but holder unhealthy; spawning via strict-port path"
                    ),
                );
            }
        }
    }

    let port = process::find_available_port(7860, 50)
        .ok_or_else(|| "Failed to find available TCP port between 7860 and 7910".to_string())?;

    let mut manager = BackendProcessManager::spawn(port, &data_dir, state.resources_dir()).await?;
    crate::boot::append_boot_log(
        &data_dir,
        &format!("backend spawned pid={:?} port={port}", manager.child_id()),
    );

    // The error already names the log file; wrapping it again only hides it.
    // Logged verbatim: the message itself says ready, early-exit, or timeout.
    match manager.wait_until_ready(ready_timeout_secs).await {
        Ok(()) => {
            crate::boot::append_boot_log(&data_dir, &format!("backend ready on port {port}"));
        }
        Err(err) => {
            crate::boot::append_boot_log(&data_dir, &format!("backend not ready: {err}"));
            return Err(err);
        }
    }

    state.set_backend_manager(manager);
    Ok(port)
}

/// Tauri command: Query desktop configuration (rendered on the boot shell).
#[tauri::command]
pub fn get_app_config(state: State<'_, AppState>) -> Result<AppConfigResponse, String> {
    get_app_config_impl(&state)
}

/// Tauri command: Re-run the boot sequence after a reported failure.
///
/// Returns immediately — provisioning can take many minutes and must not hold
/// an IPC thread or leave the shell button looking dead.
#[tauri::command]
pub async fn retry_boot(app: AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn(async move {
        crate::boot::retry(app).await;
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_app_config_response_serialization_roundtrip() {
        let res = AppConfigResponse {
            app_data_dir: "C:\\Users\\User\\AppData\\Local\\Clarity".to_string(),
            resources_dir: "C:\\Program Files\\Clarity\\resources".to_string(),
            log_dir: "C:\\Users\\User\\AppData\\Local\\Clarity\\logs".to_string(),
            port: 7860,
            setup_complete: false,
            backend_running: false,
        };

        let json = serde_json::to_string(&res).expect("serialize AppConfigResponse");
        let deserialized: AppConfigResponse =
            serde_json::from_str(&json).expect("deserialize AppConfigResponse");
        assert_eq!(res, deserialized);
        assert!(json.contains("\"setup_complete\":false"));
        assert!(json.contains("\"backend_running\":false"));
    }

    #[test]
    fn test_get_app_config_impl() {
        let state = AppState::new(PathBuf::from("my_data_dir"), PathBuf::from("my_res_dir"));
        state.set_port(7869);

        let config = get_app_config_impl(&state).expect("get_app_config_impl");
        // The data root is resolved live: first-run provisioning can move it to
        // the drive the user chose, and the shell must be told the truth.
        assert_eq!(
            config.app_data_dir,
            AppState::default_app_data_dir().to_string_lossy()
        );
        assert_eq!(config.resources_dir, "my_res_dir");
        assert_eq!(config.port, 7869);
        assert!(!config.setup_complete);
        assert!(!config.backend_running);
        assert!(
            config.log_dir.ends_with("logs"),
            "the shell shows this path, so it must point at the logs: {}",
            config.log_dir
        );
    }

    #[test]
    fn test_navigate_target_url_vs_relative_path() {
        // Standard absolute HTTP/HTTPS URLs parse as tauri::Url
        let valid_url = "http://127.0.0.1:7860/api";
        assert!(valid_url.parse::<tauri::Url>().is_ok());

        let valid_tauri_url = "tauri://localhost/setup.html";
        assert!(valid_tauri_url.parse::<tauri::Url>().is_ok());

        // Relative routes do not parse as tauri::Url and must use escaped JS eval
        let rel_path = "setup.html";
        assert!(rel_path.parse::<tauri::Url>().is_err());

        let rel_slash_path = "/setup.html?retry=1";
        assert!(rel_slash_path.parse::<tauri::Url>().is_err());
    }

    #[test]
    fn test_escape_relative_path_for_js() {
        let tricky_target = "setup.html?msg=hello\"world'&foo=bar";
        let escaped = serde_json::to_string(tricky_target).unwrap();
        let js = format!("window.location.href = {};", escaped);
        // Ensure proper quotes and escaped quotes in JS string
        assert!(js.starts_with("window.location.href = \""));
        assert!(js.contains(r#"\"world"#));
        assert!(js.ends_with("\";"));
    }
}
