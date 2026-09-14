use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::Mutex;
use crate::process::BackendProcessManager;

/// The `C:\` style root of a path, when it has a Windows drive prefix.
fn drive_root(path: &Path) -> Option<String> {
    match path.components().next()? {
        std::path::Component::Prefix(prefix) => {
            let text = prefix.as_os_str().to_string_lossy().to_string();
            if text.is_empty() {
                return None;
            }
            Some(if text.ends_with('\\') {
                text
            } else {
                format!("{text}\\")
            })
        }
        _ => None,
    }
}

/// The directory of a *real installation*, if this process is one.
///
/// A repository checkout or `cargo test` must not be treated as an install:
/// following the exe's drive there would scatter a `Clarity-data` folder across
/// every development volume and remember it as the user's choice.
fn packaged_install_dir() -> Option<PathBuf> {
    if cfg!(debug_assertions) {
        return None;
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?.to_path_buf();
    if dir.join("uv.exe").is_file() || dir.join("resources").is_dir() {
        Some(dir)
    } else {
        None
    }
}

/// Central desktop application state managed by Tauri and shared across IPC command handlers.
pub struct AppState {
    pub backend_manager: Mutex<Option<BackendProcessManager>>,
    pub app_data_dir: PathBuf,
    pub resources_dir: PathBuf,
    pub setup_complete: AtomicBool,
    pub boot_running: AtomicBool,
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
            boot_running: AtomicBool::new(false),
            port: AtomicU16::new(7860),
        }
    }

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

    pub fn is_boot_running(&self) -> bool {
        self.boot_running.load(Ordering::SeqCst)
    }

    /// Atomically transitions the boot guard from `false` to `true`.
    ///
    /// Returns `true` when this caller owns the boot sequence. That is what
    /// keeps a double-clicked Retry (or a second launch) from provisioning the
    /// environment twice.
    pub fn try_begin_boot(&self) -> bool {
        self.boot_running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Releases the boot guard so a later retry can run.
    pub fn finish_boot(&self) {
        self.boot_running.store(false, Ordering::SeqCst);
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
        assert!(!state.is_boot_running());
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
    fn test_app_state_boot_guard_concurrency() {
        let state = AppState::new(PathBuf::from("data"), PathBuf::from("res"));
        assert!(!state.is_setup_complete());
        state.set_setup_complete(true);
        assert!(state.is_setup_complete());

        assert!(!state.is_boot_running());
        assert!(state.try_begin_boot());
        assert!(state.is_boot_running());
        // Second attempt must fail
        assert!(!state.try_begin_boot());

        state.finish_boot();
        assert!(!state.is_boot_running());
        assert!(state.try_begin_boot());
    }

    #[test]
    fn test_default_app_data_dir() {
        let data_dir = AppState::default_app_data_dir();
        assert!(data_dir.ends_with("Clarity"));
    }

    #[test]
    fn test_data_root_is_the_install_folder() {
        let root = std::env::temp_dir().join("clarity_root_install");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let dir = AppState::resolve_data_dir(
            None,
            Some(root.as_path()),
            Path::new("C:\\Users\\someone\\AppData\\Local"),
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
        let dir = AppState::resolve_data_dir(None, Some(root.as_path()), Path::new("C:\\nowhere"));
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

    #[cfg(windows)]
    #[test]
    fn test_drive_root_recognises_windows_prefixes() {
        assert_eq!(drive_root(Path::new("D:\\Clarity")).as_deref(), Some("D:\\"));
        assert_eq!(drive_root(Path::new("C:")).as_deref(), Some("C:\\"));
        assert_eq!(drive_root(Path::new("\\\\server\\share\\x")).as_deref(), Some("\\\\server\\share\\"));
        assert_eq!(drive_root(Path::new("relative/path")), None);
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
