use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::hardware::{self, GpuTarget};
use crate::process::{self, BackendProcessManager};
use crate::setup;
use crate::state::AppState;

/// Status payload returned by `get_setup_status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetupStatusResponse {
    pub complete: bool,
    pub running: bool,
    pub port: u16,
    pub gpu_target: GpuTarget,
}

/// Configuration payload returned by `get_app_config`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfigResponse {
    pub app_data_dir: String,
    pub resources_dir: String,
    pub port: u16,
    pub gpu_target: GpuTarget,
    pub setup_complete: bool,
    pub backend_running: bool,
}

/// Navigates the main webview window to the specified URL or route.
pub fn navigate_window(app: &AppHandle, target: &str) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .or_else(|| app.webview_windows().into_values().next())
        .ok_or_else(|| "No webview window found".to_string())?;

    if let Ok(url) = target.parse::<tauri::Url>() {
        window
            .navigate(url)
            .map_err(|e| format!("Navigation failed: {e}"))
    } else {
        let escaped = serde_json::to_string(target).unwrap_or_else(|_| format!("\"{}\"", target));
        let js = format!("window.location.href = {};", escaped);
        window
            .eval(&js)
            .map_err(|e| format!("Failed to evaluate navigation script: {e}"))
    }
}

/// Pure implementation of setup status query.
pub fn get_setup_status_impl(state: &AppState) -> Result<SetupStatusResponse, String> {
    let complete = state.is_setup_complete() || setup::is_setup_complete(state.app_data_dir());
    let running = state.is_setup_running();
    let port = state.port();
    let gpu_target = hardware::detect_gpu_target();

    Ok(SetupStatusResponse {
        complete,
        running,
        port,
        gpu_target,
    })
}

/// Pure implementation of app configuration query.
pub fn get_app_config_impl(state: &AppState) -> Result<AppConfigResponse, String> {
    let setup_complete = state.is_setup_complete() || setup::is_setup_complete(state.app_data_dir());
    Ok(AppConfigResponse {
        app_data_dir: state.app_data_dir().to_string_lossy().to_string(),
        resources_dir: state.resources_dir().to_string_lossy().to_string(),
        port: state.port(),
        gpu_target: hardware::detect_gpu_target(),
        setup_complete,
        backend_running: state.is_backend_running(),
    })
}

/// Pure validation and state transition for starting setup.
pub fn start_setup_check_and_lock(state: &AppState) -> Result<bool, String> {
    if state.is_setup_complete() || setup::is_setup_complete(state.app_data_dir()) {
        state.set_setup_complete(true);
        return Ok(true); // Already complete
    }

    if !state.try_start_setup() {
        return Err("Setup is already in progress".to_string());
    }

    Ok(false) // Needs to run
}

/// Internal helper to launch the Python FastAPI backend process.
pub async fn launch_backend_internal(state: &AppState) -> Result<u16, String> {
    if state.is_backend_running() {
        return Ok(state.port());
    }

    let port = process::find_available_port(7860, 50)
        .ok_or_else(|| "Failed to find available TCP port between 7860 and 7910".to_string())?;

    let manager = BackendProcessManager::spawn(port, state.app_data_dir(), state.resources_dir()).await?;

    if let Err(e) = manager.wait_until_ready(30).await {
        return Err(format!("Backend process failed healthcheck readiness: {}", e));
    }

    state.set_backend_manager(manager);
    Ok(port)
}

/// Tauri command: Query setup status and current backend port.
#[tauri::command]
pub fn get_setup_status(state: State<'_, AppState>) -> Result<SetupStatusResponse, String> {
    get_setup_status_impl(&state)
}

/// Tauri command: Query desktop environment configuration.
#[tauri::command]
pub fn get_app_config(state: State<'_, AppState>) -> Result<AppConfigResponse, String> {
    get_app_config_impl(&state)
}

/// Tauri command: Begin first-run setup and dependency installation.
#[tauri::command]
pub async fn start_setup(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let already_complete = start_setup_check_and_lock(&state)?;
    if already_complete {
        let _ = app.emit(
            "setup-complete",
            serde_json::json!({
                "complete": true,
                "port": state.port(),
                "url": format!("http://127.0.0.1:{}", state.port()),
            }),
        );
        return Ok(());
    }

    let app_handle = app.clone();
    let app_data_dir = state.app_data_dir().to_path_buf();
    let resources_dir = state.resources_dir().to_path_buf();
    let gpu_target = hardware::detect_gpu_target();

    tauri::async_runtime::spawn(async move {
        let emit_handle = app_handle.clone();
        let on_progress = move |event: setup::SetupProgressEvent| {
            let _ = emit_handle.emit("setup-progress", &event);
        };

        let res = setup::run_setup(&app_data_dir, &resources_dir, gpu_target, on_progress).await;

        if let Some(state) = app_handle.try_state::<AppState>() {
            state.set_setup_running(false);
            match res {
                Ok(()) => match launch_backend_internal(&state).await {
                    Ok(port) => {
                        state.set_setup_complete(true);
                        let _ = app_handle.emit(
                            "setup-complete",
                            serde_json::json!({
                                "complete": true,
                                "port": port,
                                "url": format!("http://127.0.0.1:{}", port),
                            }),
                        );
                    }
                    Err(err) => {
                        eprintln!("Backend launch after setup failed: {}", err);
                        let err_msg = format!("Setup completed, but failed to launch backend server: {}", err);
                        let _ = app_handle.emit(
                            "setup-error",
                            serde_json::json!({
                                "error": err_msg,
                                "message": err_msg,
                            }),
                        );
                    }
                },
                Err(err) => {
                    let _ = app_handle.emit(
                        "setup-error",
                        serde_json::json!({
                            "error": err,
                            "message": err,
                        }),
                    );
                }
            }
        }
    });

    Ok(())
}

/// Tauri command: Retry setup after failure.
#[tauri::command]
pub async fn retry_setup(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    state.set_setup_running(false);
    start_setup(app, state).await
}

/// Tauri command: Launch the backend server and navigate the window to it.
#[tauri::command]
pub async fn launch_backend(app: AppHandle, state: State<'_, AppState>) -> Result<u16, String> {
    let port = launch_backend_internal(&state).await?;
    let target_url = format!("http://127.0.0.1:{}", port);
    let _ = navigate_window(&app, &target_url);
    Ok(port)
}

/// Tauri command: Navigate the webview window to the main application or target URL.
#[tauri::command]
pub async fn launch_main_app(
    app: AppHandle,
    state: State<'_, AppState>,
    url: Option<String>,
) -> Result<(), String> {
    let port = if state.is_backend_running() {
        state.port()
    } else {
        launch_backend_internal(&state).await?
    };
    let target_url = url.unwrap_or_else(|| format!("http://127.0.0.1:{}", port));
    navigate_window(&app, &target_url)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_setup_status_response_serialization_roundtrip() {
        let res = SetupStatusResponse {
            complete: true,
            running: false,
            port: 7860,
            gpu_target: GpuTarget::NvidiaCuda {
                device_name: "NVIDIA RTX 3080".to_string(),
            },
        };

        let json = serde_json::to_string(&res).expect("serialize SetupStatusResponse");
        let deserialized: SetupStatusResponse =
            serde_json::from_str(&json).expect("deserialize SetupStatusResponse");
        assert_eq!(res, deserialized);
        assert!(json.contains("\"complete\":true"));
        assert!(json.contains("\"port\":7860"));
    }

    #[test]
    fn test_app_config_response_serialization_roundtrip() {
        let res = AppConfigResponse {
            app_data_dir: "C:\\Users\\User\\AppData\\Local\\Clarity".to_string(),
            resources_dir: "C:\\Program Files\\Clarity\\resources".to_string(),
            port: 7860,
            gpu_target: GpuTarget::CpuFallback,
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
    fn test_get_setup_status_impl_initial() {
        let temp_dir = std::env::temp_dir().join(format!("clarity_cmd_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let state = AppState::new(temp_dir.clone(), temp_dir.clone());
        let status = get_setup_status_impl(&state).expect("get_setup_status_impl");
        assert!(!status.complete);
        assert!(!status.running);
        assert_eq!(status.port, 7860);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_get_setup_status_impl_when_complete() {
        let state = AppState::new(PathBuf::from("data"), PathBuf::from("res"));
        state.set_setup_complete(true);
        state.set_port(7862);

        let status = get_setup_status_impl(&state).expect("get_setup_status_impl");
        assert!(status.complete);
        assert!(!status.running);
        assert_eq!(status.port, 7862);
    }

    #[test]
    fn test_get_app_config_impl() {
        let state = AppState::new(PathBuf::from("my_data_dir"), PathBuf::from("my_res_dir"));
        state.set_port(7869);

        let config = get_app_config_impl(&state).expect("get_app_config_impl");
        assert_eq!(config.app_data_dir, "my_data_dir");
        assert_eq!(config.resources_dir, "my_res_dir");
        assert_eq!(config.port, 7869);
        assert!(!config.setup_complete);
        assert!(!config.backend_running);
    }

    #[test]
    fn test_start_setup_check_and_lock_prevents_duplicate_runs() {
        let temp_dir = std::env::temp_dir().join(format!("clarity_lock_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let state = AppState::new(temp_dir.clone(), temp_dir.clone());

        // First attempt succeeds and flags setup as running
        let res1 = start_setup_check_and_lock(&state);
        assert_eq!(res1, Ok(false)); // Needs to run
        assert!(state.is_setup_running());

        // Second attempt while running should return Err
        let res2 = start_setup_check_and_lock(&state);
        assert!(res2.is_err());
        assert_eq!(res2.unwrap_err(), "Setup is already in progress");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_start_setup_check_and_lock_when_already_complete() {
        let temp_dir = std::env::temp_dir().join(format!("clarity_complete_lock_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let state = AppState::new(temp_dir.clone(), temp_dir.clone());
        state.set_setup_complete(true);

        let res = start_setup_check_and_lock(&state);
        assert_eq!(res, Ok(true)); // Already complete
        assert!(!state.is_setup_running());

        let _ = std::fs::remove_dir_all(&temp_dir);
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
