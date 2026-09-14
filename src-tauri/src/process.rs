use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::Duration;
use winapi::um::handleapi::CloseHandle;
use winapi::um::jobapi2::{
    AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject, TerminateJobObject,
};
use winapi::um::winnt::{
    JobObjectExtendedLimitInformation, HANDLE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// Finds an available TCP port on 127.0.0.1 starting from `start` for up to `max_attempts`.
pub fn find_available_port(start: u16, max_attempts: u16) -> Option<u16> {
    if max_attempts == 0 {
        return None;
    }

    for attempt in 0..max_attempts {
        let port = match start.checked_add(attempt) {
            Some(p) => p,
            None => break,
        };

        if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
            drop(listener);
            return Some(port);
        }
    }

    None
}

/// Resolves the Python executable to run.
///
/// Priority:
/// 1. `%LOCALAPPDATA%\Clarity\env\Scripts\python.exe` (production isolated venv)
/// 2. `.venv\Scripts\python.exe` (project root development venv)
/// 3. `..\.venv\Scripts\python.exe` (development venv relative to src-tauri)
/// 4. `%VIRTUAL_ENV%\Scripts\python.exe` (active shell venv if set)
/// 5. System `"python.exe"` fallback
pub fn resolve_python_path(app_data_dir: &Path) -> PathBuf {
    let installed_env = app_data_dir.join("env").join("Scripts").join("python.exe");
    if installed_env.exists() {
        return installed_env;
    }

    let dev_venv = Path::new(".venv").join("Scripts").join("python.exe");
    if dev_venv.exists() {
        return dev_venv;
    }

    let parent_dev_venv = Path::new("..").join(".venv").join("Scripts").join("python.exe");
    if parent_dev_venv.exists() {
        return parent_dev_venv;
    }

    if let Ok(venv_val) = std::env::var("VIRTUAL_ENV") {
        let venv_python = PathBuf::from(venv_val).join("Scripts").join("python.exe");
        if venv_python.exists() {
            return venv_python;
        }
    }

    PathBuf::from("python.exe")
}

/// Builds the backend `tokio::process::Command` with all required flags and environment variables.
pub fn build_backend_command(
    python_bin: &Path,
    port: u16,
    app_data_dir: &Path,
    resources_dir: &Path,
) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(python_bin);

    cmd.args([
        "-m",
        "video_upscaler.web.server",
        "--port",
        &port.to_string(),
        "--no-browser",
    ]);

    // Prepend resources_dir to PATH so bundled ffmpeg.exe and ffprobe.exe are found immediately
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = if current_path.is_empty() {
        resources_dir.to_string_lossy().to_string()
    } else {
        format!("{};{}", resources_dir.display(), current_path)
    };
    cmd.env("PATH", new_path);

    // Isolated models directory and desktop mode flag
    cmd.env("CLARITY_MODELS_DIR", app_data_dir.join("models"));
    cmd.env("CLARITY_DESKTOP_MODE", "1");

    // Hide console window on Windows (CREATE_NO_WINDOW = 0x08000000)
    cmd.creation_flags(0x08000000);

    cmd
}

/// Creates a Windows Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.
pub fn create_kill_on_close_job() -> Result<HANDLE, String> {
    unsafe {
        let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
        if job.is_null() {
            return Err(format!(
                "CreateJobObjectW failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        let res = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &mut info as *mut _ as winapi::um::winnt::PVOID,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );

        if res == 0 {
            let err = std::io::Error::last_os_error();
            CloseHandle(job);
            return Err(format!("SetInformationJobObject failed: {}", err));
        }

        Ok(job)
    }
}

pub struct BackendProcessManager {
    job_handle: HANDLE,
    child: tokio::process::Child,
    pub port: u16,
}

// HANDLE is an OS-level pointer that can safely be moved across threads.
// All &self methods are read-only and thread-safe.
unsafe impl Send for BackendProcessManager {}
unsafe impl Sync for BackendProcessManager {}

impl BackendProcessManager {
    /// Spawns the Python backend server bound to the specified port and assigns it to a Windows Job Object.
    pub async fn spawn(
        port: u16,
        app_data_dir: &Path,
        resources_dir: &Path,
    ) -> Result<Self, String> {
        let python_bin = resolve_python_path(app_data_dir);
        let job_handle = create_kill_on_close_job()?;

        let mut cmd = build_backend_command(&python_bin, port, app_data_dir, resources_dir);

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                unsafe {
                    CloseHandle(job_handle);
                }
                return Err(format!(
                    "Failed to spawn python backend at {:?}: {}",
                    python_bin, e
                ));
            }
        };

        let raw_proc_handle = match child.raw_handle() {
            Some(h) => h as HANDLE,
            None => {
                unsafe {
                    CloseHandle(job_handle);
                }
                return Err("Failed to obtain raw handle from child process".to_string());
            }
        };

        let assign_res = unsafe { AssignProcessToJobObject(job_handle, raw_proc_handle) };
        if assign_res == 0 {
            let err = std::io::Error::last_os_error();
            unsafe {
                CloseHandle(job_handle);
            }
            let mut child = child;
            let _ = child.start_kill();
            return Err(format!("AssignProcessToJobObject failed: {}", err));
        }

        Ok(BackendProcessManager {
            job_handle,
            child,
            port,
        })
    }

    /// Polls `http://127.0.0.1:{port}/api/system/info` until it returns 200 OK or times out.
    pub async fn wait_for_port_ready(port: u16, timeout_secs: u64) -> Result<(), String> {
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()
        {
            Ok(c) => c,
            Err(e) => return Err(format!("Failed to build HTTP client: {}", e)),
        };

        let url = format!("http://127.0.0.1:{}/api/system/info", port);
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(timeout_secs);
        let poll_interval = Duration::from_millis(200);

        while start.elapsed() < timeout {
            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
            tokio::time::sleep(poll_interval).await;
        }

        Err(format!(
            "Backend process failed to respond at {} within {} seconds",
            url, timeout_secs
        ))
    }

    /// Polls `http://127.0.0.1:{port}/api/system/info` until it returns 200 OK or times out.
    pub async fn wait_until_ready(&self, timeout_secs: u64) -> Result<(), String> {
        Self::wait_for_port_ready(self.port, timeout_secs).await
    }

    /// Terminates the backend process and closes the associated Job Object handle.
    pub fn terminate(&mut self) {
        if !self.job_handle.is_null() {
            unsafe {
                TerminateJobObject(self.job_handle, 1);
                CloseHandle(self.job_handle);
            }
            self.job_handle = std::ptr::null_mut();
        }
        let _ = self.child.start_kill();
    }
}

impl Drop for BackendProcessManager {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener as TokioTcpListener;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn test_process_manager_is_send_and_sync() {
        assert_send::<BackendProcessManager>();
        assert_sync::<BackendProcessManager>();
    }

    #[test]
    fn test_find_available_port() {
        let port = find_available_port(7860, 50);
        assert!(port.is_some());
        let p = port.unwrap();
        assert!(p >= 7860);
    }

    #[test]
    fn test_find_available_port_occupied() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind random port");
        let bound_port = listener.local_addr().unwrap().port();

        // With max_attempts = 1, it should fail to bind the already-bound port
        let found = find_available_port(bound_port, 1);
        assert!(found.is_none());

        // With max_attempts = 2, it should attempt next port(s)
        let found_next = find_available_port(bound_port, 2);
        if let Some(next) = found_next {
            assert_ne!(next, bound_port);
        }
    }

    #[test]
    fn test_find_available_port_exhausted() {
        let found = find_available_port(7860, 0);
        assert!(found.is_none());
    }

    #[test]
    fn test_find_available_port_overflow_safety() {
        let found = find_available_port(65535, 10);
        // Should not panic on overflow
        assert!(found.is_some() || found.is_none());
    }

    #[test]
    fn test_resolve_python_path_with_installed_venv() {
        let temp_dir = std::env::temp_dir().join(format!("clarity_test_{}", std::process::id()));
        let venv_scripts = temp_dir.join("env").join("Scripts");
        std::fs::create_dir_all(&venv_scripts).unwrap();
        let python_exe = venv_scripts.join("python.exe");
        std::fs::write(&python_exe, b"").unwrap();

        let resolved = resolve_python_path(&temp_dir);
        assert_eq!(resolved, python_exe);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_resolve_python_path_fallback() {
        let non_existent = Path::new("C:\\non_existent_clarity_path_12345");
        let resolved = resolve_python_path(non_existent);
        assert!(
            resolved.ends_with("python.exe") || resolved == PathBuf::from("python"),
            "Resolved path was: {:?}",
            resolved
        );
    }

    #[test]
    fn test_create_kill_on_close_job() {
        let job = create_kill_on_close_job().expect("Failed to create job object");
        assert!(!job.is_null());
        unsafe {
            CloseHandle(job);
        }
    }

    #[test]
    fn test_build_backend_command_properties() {
        let python = Path::new("python.exe");
        let app_data = Path::new("C:\\ClarityData");
        let resources = Path::new("C:\\ClarityResources");

        let cmd = build_backend_command(python, 7865, app_data, resources);
        let std_cmd = cmd.as_std();

        let args: Vec<&std::ffi::OsStr> = std_cmd.get_args().collect();
        assert_eq!(args[0], "-m");
        assert_eq!(args[1], "video_upscaler.web.server");
        assert_eq!(args[2], "--port");
        assert_eq!(args[3], "7865");
        assert_eq!(args[4], "--no-browser");

        let envs: std::collections::HashMap<_, _> = std_cmd
            .get_envs()
            .filter_map(|(k, v)| v.map(|val| (k.to_os_string(), val.to_os_string())))
            .collect();

        assert_eq!(
            envs.get(std::ffi::OsStr::new("CLARITY_DESKTOP_MODE")),
            Some(&std::ffi::OsString::from("1"))
        );
        assert_eq!(
            envs.get(std::ffi::OsStr::new("CLARITY_MODELS_DIR")),
            Some(&std::ffi::OsString::from(app_data.join("models")))
        );

        let path_val = envs
            .get(std::ffi::OsStr::new("PATH"))
            .expect("PATH env must be set");
        let path_str = path_val.to_string_lossy();
        assert!(path_str.starts_with(&resources.to_string_lossy().to_string()));
    }

    #[tokio::test]
    async fn test_wait_until_ready_success() {
        let listener = TokioTcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        // Spawn mock HTTP server responding 200 OK to /api/system/info
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = socket.read(&mut buf).await;
                let response =
                    "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/json\r\n\r\n{}";
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });

        let dummy_child = tokio::process::Command::new("cmd")
            .args(["/c", "exit 0"])
            .spawn()
            .unwrap();

        let manager = BackendProcessManager {
            job_handle: std::ptr::null_mut(),
            child: dummy_child,
            port,
        };

        let res = manager.wait_until_ready(5).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_wait_until_ready_timeout() {
        // Find an unused port and do NOT run anything on it
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let free_port = listener.local_addr().unwrap().port();
        drop(listener);

        let dummy_child = tokio::process::Command::new("cmd")
            .args(["/c", "ping -n 10 127.0.0.1 > nul"])
            .spawn()
            .unwrap();

        let mut manager = BackendProcessManager {
            job_handle: std::ptr::null_mut(),
            child: dummy_child,
            port: free_port,
        };

        let start = std::time::Instant::now();
        let res = manager.wait_until_ready(1).await;
        assert!(res.is_err());
        assert!(start.elapsed() >= Duration::from_millis(900));

        manager.terminate();
    }

    #[tokio::test]
    async fn test_job_object_process_lifecycle_and_terminate() {
        let job = create_kill_on_close_job().expect("create job failed");

        let mut cmd = tokio::process::Command::new("cmd");
        cmd.args(["/c", "ping -n 10 127.0.0.1 > nul"]);
        cmd.creation_flags(0x08000000);

        let child = cmd.spawn().expect("spawn ping failed");
        let raw_h = child.raw_handle().expect("raw handle") as HANDLE;

        let assign_res = unsafe { AssignProcessToJobObject(job, raw_h) };
        assert_ne!(assign_res, 0, "AssignProcessToJobObject failed");

        let mut manager = BackendProcessManager {
            job_handle: job,
            child,
            port: 7860,
        };

        assert!(!manager.job_handle.is_null());
        manager.terminate();
        assert!(manager.job_handle.is_null());

        // Second terminate call should be safe / idempotent
        manager.terminate();
        assert!(manager.job_handle.is_null());
    }
}
