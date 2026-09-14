use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::Mutex;
use crate::process::BackendProcessManager;

/// Central desktop application state managed by Tauri and shared across IPC command handlers.
pub struct AppState {
    pub backend_manager: Mutex<Option<BackendProcessManager>>,
    pub app_data_dir: PathBuf,
    pub resources_dir: PathBuf,
    pub setup_complete: AtomicBool,
    pub setup_running: AtomicBool,
    pub port: AtomicU16,
}

impl AppState {
    /// Creates a new `AppState` instance, detecting whether setup is already completed.
    pub fn new(app_data_dir: PathBuf, resources_dir: PathBuf) -> Self {
        let is_complete = crate::setup::is_setup_complete(&app_data_dir);
        Self {
            backend_manager: Mutex::new(None),
            app_data_dir,
            resources_dir,
            setup_complete: AtomicBool::new(is_complete),
            setup_running: AtomicBool::new(false),
            port: AtomicU16::new(7860),
        }
    }

    /// Resolves the default `%LOCALAPPDATA%\Clarity` storage location on Windows.
    pub fn default_app_data_dir() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("Clarity")
    }

    /// Resolves the bundled application resources directory.
    pub fn resolve_resources_dir(app: Option<&tauri::AppHandle>) -> PathBuf {
        if let Some(app) = app {
            use tauri::Manager;
            if let Ok(res_path) = app.path().resource_dir() {
                if res_path.exists() {
                    return res_path;
                }
            }
        }

        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let adjacent_res = exe_dir.join("resources");
                if adjacent_res.exists() {
                    return adjacent_res;
                }
                if exe_dir.join("uv.exe").is_file() {
                    return exe_dir.to_path_buf();
                }
            }
        }

        let candidates = [
            PathBuf::from("resources"),
            PathBuf::from("src-tauri").join("resources"),
            PathBuf::from("..").join("src-tauri").join("resources"),
        ];

        for c in &candidates {
            if c.exists() {
                return c.clone();
            }
        }

        candidates[1].clone()
    }

    pub fn app_data_dir(&self) -> &Path {
        &self.app_data_dir
    }

    pub fn resources_dir(&self) -> &Path {
        &self.resources_dir
    }

    pub fn port(&self) -> u16 {
        self.port.load(Ordering::SeqCst)
    }

    pub fn set_port(&self, port: u16) {
        self.port.store(port, Ordering::SeqCst);
    }

    pub fn is_setup_complete(&self) -> bool {
        self.setup_complete.load(Ordering::SeqCst)
    }

    pub fn set_setup_complete(&self, complete: bool) {
        self.setup_complete.store(complete, Ordering::SeqCst);
    }

    pub fn is_setup_running(&self) -> bool {
        self.setup_running.load(Ordering::SeqCst)
    }

    /// Atomically transitions `setup_running` from `false` to `true`.
    /// Returns `true` if this caller succeeded in initiating setup, or `false` if setup was already running.
    pub fn try_start_setup(&self) -> bool {
        self.setup_running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    pub fn set_setup_running(&self, running: bool) {
        self.setup_running.store(running, Ordering::SeqCst);
    }

    pub fn is_backend_running(&self) -> bool {
        let guard = self.backend_manager.lock().unwrap();
        guard.is_some()
    }

    pub fn set_backend_manager(&self, manager: BackendProcessManager) {
        self.set_port(manager.port);
        let mut guard = self.backend_manager.lock().unwrap();
        if let Some(mut old) = guard.take() {
            old.terminate();
        }
        *guard = Some(manager);
    }

    pub fn terminate_backend(&self) {
        let mut guard = self.backend_manager.lock().unwrap();
        if let Some(mut manager) = guard.take() {
            manager.terminate();
        }
    }
}

impl Drop for AppState {
    fn drop(&mut self) {
        self.terminate_backend();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_app_state_is_send_and_sync() {
        assert_send_sync::<AppState>();
    }

    #[test]
    fn test_app_state_initialization_defaults() {
        let temp_dir = std::env::temp_dir().join(format!("clarity_test_state_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);

        let state = AppState::new(temp_dir.clone(), temp_dir.clone());
        assert_eq!(state.port(), 7860);
        assert!(!state.is_setup_complete());
        assert!(!state.is_setup_running());
        assert!(!state.is_backend_running());
        assert_eq!(state.app_data_dir(), temp_dir.as_path());
        assert_eq!(state.resources_dir(), temp_dir.as_path());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_app_state_port_mutation() {
        let state = AppState::new(PathBuf::from("data"), PathBuf::from("res"));
        assert_eq!(state.port(), 7860);
        state.set_port(7865);
        assert_eq!(state.port(), 7865);
    }

    #[test]
    fn test_app_state_setup_flags_concurrency() {
        let state = AppState::new(PathBuf::from("data"), PathBuf::from("res"));
        assert!(!state.is_setup_complete());
        state.set_setup_complete(true);
        assert!(state.is_setup_complete());

        assert!(!state.is_setup_running());
        assert!(state.try_start_setup());
        assert!(state.is_setup_running());
        // Second attempt must fail
        assert!(!state.try_start_setup());

        state.set_setup_running(false);
        assert!(!state.is_setup_running());
        assert!(state.try_start_setup());
    }

    #[test]
    fn test_default_app_data_dir() {
        let data_dir = AppState::default_app_data_dir();
        assert!(data_dir.ends_with("Clarity"));
    }

    #[test]
    fn test_resolve_resources_dir_fallback() {
        let res_dir = AppState::resolve_resources_dir(None);
        assert!(!res_dir.as_os_str().is_empty());
    }

    #[test]
    fn test_terminate_backend_safe_when_none() {
        let state = AppState::new(PathBuf::from("data"), PathBuf::from("res"));
        assert!(!state.is_backend_running());
        state.terminate_backend();
        assert!(!state.is_backend_running());
    }
}
