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

/// What answered (or didn't) on an occupied port's `/api/health` probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthProbeOutcome {
    /// Our backend answered with its liveness payload: safe to adopt.
    Healthy,
    /// Nothing answered: dead holder or zombie socket.
    Unreachable,
    /// Something answered that is not our backend (e.g. a dev server).
    Foreign,
}

/// What boot should do about the configured port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortHolderDecision {
    /// Port is free, or its holder is dead: run the normal strict-port spawn path.
    Spawn,
    /// A healthy Clarity backend already owns the port: navigate to it.
    Adopt,
    /// Another application owns the port: fail loudly instead of colliding.
    OccupiedByForeign,
}

/// Decides adopt-vs-spawn-vs-error for the configured port without touching
/// the network itself: `probe` runs only when the port is occupied, so tests
/// stub it and production passes the real `/api/health` check.
///
/// Never returns "kill the holder": an occupied port may belong to the user's
/// own dev server, which the shell must not terminate.
pub fn classify_port_holder(
    port_occupied: bool,
    probe: impl FnOnce() -> HealthProbeOutcome,
) -> PortHolderDecision {
    if !port_occupied {
        return PortHolderDecision::Spawn;
    }
    match probe() {
        HealthProbeOutcome::Healthy => PortHolderDecision::Adopt,
        HealthProbeOutcome::Unreachable => PortHolderDecision::Spawn,
        HealthProbeOutcome::Foreign => PortHolderDecision::OccupiedByForeign,
    }
}

/// True when nothing can bind `port` on loopback right now.
pub fn is_port_occupied(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_err()
}

/// Probes `http://127.0.0.1:{port}/api/health` to decide whether the process
/// holding the port is our backend. Short timeout: boot must not stall on it.
///
/// Only a 2xx carrying our `{"status": "ok"}` liveness payload counts as
/// healthy — anything else that answers is treated as foreign, so the shell
/// never adopts a stranger's server.
pub async fn probe_backend_health(port: u16) -> HealthProbeOutcome {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(client) => client,
        Err(_) => return HealthProbeOutcome::Unreachable,
    };
    let url = format!("http://127.0.0.1:{port}/api/health");
    let response = match client.get(&url).send().await {
        Ok(response) => response,
        Err(_) => return HealthProbeOutcome::Unreachable,
    };
    if !response.status().is_success() {
        return HealthProbeOutcome::Foreign;
    }
    match response.json::<serde_json::Value>().await {
        Ok(body) if body.get("status") == Some(&serde_json::Value::from("ok")) => {
            HealthProbeOutcome::Healthy
        }
        _ => HealthProbeOutcome::Foreign,
    }
}

/// Resolves the Python interpreter used to run the studio backend.
///
/// Priority:
/// 1. `%LOCALAPPDATA%\Clarity\env\Scripts\python.exe` (provisioned by first-run setup)
/// 2. `.venv\Scripts\python.exe` / `..\.venv\...` (repository development venv)
/// 3. `%VIRTUAL_ENV%\Scripts\python.exe` (active shell venv)
///
/// `None` means the environment has not been provisioned: the caller must run
/// the setup wizard rather than launch a process that cannot import its own
/// dependencies and then time out for no apparent reason.
pub fn resolve_python_path(app_data_dir: &Path) -> Option<PathBuf> {
    let installed_env = app_data_dir.join("env").join("Scripts").join("python.exe");
    if installed_env.exists() {
        return Some(installed_env);
    }

    for dev_venv in [
        PathBuf::from(".venv").join("Scripts").join("python.exe"),
        PathBuf::from("..").join(".venv").join("Scripts").join("python.exe"),
    ] {
        if dev_venv.exists() {
            return Some(dev_venv);
        }
    }

    if let Ok(venv_val) = std::env::var("VIRTUAL_ENV") {
        let venv_python = PathBuf::from(venv_val).join("Scripts").join("python.exe");
        if venv_python.exists() {
            return Some(venv_python);
        }
    }

    None
}

/// Location of the studio backend log for a given data directory.
pub fn backend_log_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("logs").join("backend.log")
}

/// Builds the backend `tokio::process::Command` with all required flags and environment variables.
///
/// Output is appended to `<data>/logs/backend.log`; with the console hidden there
/// is nowhere else for a startup crash to go.
pub fn build_backend_command(
    python_bin: &Path,
    port: u16,
    app_data_dir: &Path,
    resources_dir: &Path,
) -> Result<tokio::process::Command, String> {
    let mut cmd = tokio::process::Command::new(python_bin);

    cmd.args([
        "-m",
        "video_upscaler.web.server",
        "--port",
        &port.to_string(),
        "--no-browser",
        // The shell reserved this exact port and is polling it. Without this the
        // server silently drifts to the next free port and the shell waits on a
        // dead one.
        "--strict-port",
    ]);

    // Prepend resources_dir to PATH so bundled ffmpeg.exe and ffprobe.exe are found immediately
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = if current_path.is_empty() {
        resources_dir.to_string_lossy().to_string()
    } else {
        format!("{};{}", resources_dir.display(), current_path)
    };
    cmd.env("PATH", new_path);

    // Prepend resources_dir and resources_dir/src to PYTHONPATH so video_upscaler is always discoverable
    let current_pypath = std::env::var("PYTHONPATH").unwrap_or_default();
    let src_dir = resources_dir.join("src");
    let new_pypath = if current_pypath.is_empty() {
        format!("{};{}", src_dir.display(), resources_dir.display())
    } else {
        format!("{};{};{}", src_dir.display(), resources_dir.display(), current_pypath)
    };
    cmd.env("PYTHONPATH", new_pypath);

    // One variable owns the whole writable layout (input/output/models/tools/cache),
    // so the shell and the app cannot disagree about where user media lives.
    cmd.env("CLARITY_DATA_DIR", app_data_dir);
    cmd.env("CLARITY_DESKTOP_MODE", "1");
    cmd.env("CLARITY_FFMPEG", resources_dir.join("ffmpeg.exe"));
    cmd.env("CLARITY_FFPROBE", resources_dir.join("ffprobe.exe"));

    let log_path = backend_log_path(app_data_dir);
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Failed to create log directory: {e}"))?;
    }
    let logfile = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("Failed to open backend log {log_path:?}: {e}"))?;
    cmd.stdout(std::process::Stdio::from(logfile));
    cmd.stderr(std::process::Stdio::from(
        std::fs::OpenOptions::new().create(true).append(true).open(&log_path)
            .map_err(|e| format!("Failed to open backend log {log_path:?}: {e}"))?,
    ));

    // Hide console window on Windows (CREATE_NO_WINDOW = 0x08000000)
    cmd.creation_flags(0x08000000);

    Ok(cmd)
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

/// A Job Object that kills everything assigned to it when dropped. Used for
/// short-lived helpers (the setup wizard) that outlive their own command.
pub struct ChildJob {
    job_handle: HANDLE,
}

// SAFETY: the only state is a kernel job-object handle. Kernel handles may be
// used and closed from any thread of the owning process and this type holds no
// reference into process memory, so moving it across an await point is sound.
// Without this the boot future is not `Send` and Tauri's async runtime refuses
// to spawn it.
unsafe impl Send for ChildJob {}

impl ChildJob {
    /// Creates a kill-on-close job and assigns `child` to it.
    pub fn attach(child: &tokio::process::Child) -> Result<Self, String> {
        let job_handle = create_kill_on_close_job()?;
        let raw = match child.raw_handle() {
            Some(h) => h as HANDLE,
            None => {
                unsafe { CloseHandle(job_handle) };
                return Err("Failed to obtain raw handle from child process".to_string());
            }
        };

        if unsafe { AssignProcessToJobObject(job_handle, raw) } == 0 {
            let err = std::io::Error::last_os_error();
            unsafe { CloseHandle(job_handle) };
            return Err(format!("AssignProcessToJobObject failed: {err}"));
        }

        Ok(Self { job_handle })
    }
}

impl Drop for ChildJob {
    fn drop(&mut self) {
        if !self.job_handle.is_null() {
            unsafe {
                TerminateJobObject(self.job_handle, 0);
                CloseHandle(self.job_handle);
            }
            self.job_handle = std::ptr::null_mut();
        }
    }
}

pub struct BackendProcessManager {
    job_handle: HANDLE,
    child: tokio::process::Child,
    pub port: u16,
    pub log_path: PathBuf,
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
        let python_bin = match resolve_python_path(app_data_dir) {
            Some(path) => path,
            None => {
                return Err(
                    "The Clarity runtime is not provisioned yet — run first-time setup first.".to_string(),
                )
            }
        };
        let job_handle = create_kill_on_close_job()?;

        let mut cmd = build_backend_command(&python_bin, port, app_data_dir, resources_dir)?;
        let log_path = backend_log_path(app_data_dir);

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
            log_path,
        })
    }

    /// OS process id of the backend child, for the boot log.
    pub fn child_id(&self) -> Option<u32> {
        self.child.id()
    }

    /// Polls `http://127.0.0.1:{port}/api/health` until it answers, the backend
    /// process dies, or the timeout elapses.
    ///
    /// `/api/health` is used rather than `/api/system/info` on purpose: the
    /// latter imports torch and probes the GPU, so on a cold install it can take
    /// longer than the whole readiness budget and report a healthy server dead.
    pub async fn wait_until_ready(&mut self, timeout_secs: u64) -> Result<(), String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

        let url = format!("http://127.0.0.1:{}/api/health", self.port);
        let log_path = self.log_path.display().to_string();
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(timeout_secs);

        loop {
            if let Some(status) = self
                .child
                .try_wait()
                .map_err(|e| format!("Failed to poll backend process: {e}"))?
            {
                return Err(format!(
                    "The Clarity server exited ({status}) before it was ready. See {log_path}"
                ));
            }

            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    return Ok(());
                }
            }

            if start.elapsed() > timeout {
                return Err(format!(
                    "The Clarity server did not answer on {} within {} seconds. See {log_path}",
                    self.port, timeout_secs
                ));
            }

            tokio::time::sleep(Duration::from_millis(300)).await;
        }
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
    use std::path::Path;
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
    fn test_classify_free_port_skips_probe_and_spawns() {
        // A free port must never trigger a network probe: the stub panics if called.
        let decision = classify_port_holder(false, || panic!("probe must not run on a free port"));
        assert_eq!(decision, PortHolderDecision::Spawn);
    }

    #[test]
    fn test_classify_healthy_holder_is_adopted() {
        let decision = classify_port_holder(true, || HealthProbeOutcome::Healthy);
        assert_eq!(decision, PortHolderDecision::Adopt);
    }

    #[test]
    fn test_classify_dead_holder_falls_through_to_spawn_path() {
        // A dead holder answers nothing: proceed to the strict-port spawn path,
        // which errors loudly if the port is still held.
        let decision = classify_port_holder(true, || HealthProbeOutcome::Unreachable);
        assert_eq!(decision, PortHolderDecision::Spawn);
    }

    #[test]
    fn test_classify_foreign_holder_is_an_error_not_a_spawn() {
        // Another app's server (e.g. a dev server): fail loudly instead of
        // colliding with it — and never kill the foreign holder.
        let decision = classify_port_holder(true, || HealthProbeOutcome::Foreign);
        assert_eq!(decision, PortHolderDecision::OccupiedByForeign);
    }

    #[test]
    fn test_resolve_python_path_with_installed_venv() {
        let temp_dir = std::env::temp_dir().join(format!("clarity_test_{}", std::process::id()));
        let venv_scripts = temp_dir.join("env").join("Scripts");
        std::fs::create_dir_all(&venv_scripts).unwrap();
        let python_exe = venv_scripts.join("python.exe");
        std::fs::write(&python_exe, b"").unwrap();

        let resolved = resolve_python_path(&temp_dir);
        assert_eq!(resolved, Some(python_exe));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_resolve_python_path_without_provisioned_env() {
        // No env/ directory: only the repository development venvs may match, and
        // a bare temp dir must never silently fall back to "python.exe" on PATH —
        // that interpreter cannot import the app, which used to surface as a
        // readiness timeout with no explanation.
        let non_existent = std::env::temp_dir().join(format!("clarity_absent_{}", std::process::id()));
        match resolve_python_path(&non_existent) {
            None => {}
            Some(path) => assert!(
                path.to_string_lossy().contains(".venv") || path.to_string_lossy().contains("VIRTUAL_ENV"),
                "unexpected interpreter: {:?}",
                path
            ),
        }
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
        let temp_dir = std::env::temp_dir().join(format!("clarity_cmd_env_{}", std::process::id()));
        let python = Path::new("python.exe");
        let app_data = temp_dir.as_path();
        let resources = Path::new("C:\\ClarityResources");

        let cmd = build_backend_command(python, 7865, app_data, resources).expect("build command");
        // The log file handle must be created under the data directory.
        assert!(backend_log_path(app_data).parent().unwrap().is_dir());

        let std_cmd = cmd.as_std();

        let args: Vec<&std::ffi::OsStr> = std_cmd.get_args().collect();
        assert_eq!(args[0], "-m");
        assert_eq!(args[1], "video_upscaler.web.server");
        assert_eq!(args[2], "--port");
        assert_eq!(args[3], "7865");
        assert_eq!(args[4], "--no-browser");
        assert_eq!(
            args[5], "--strict-port",
            "backend must bind the reserved port or fail loudly"
        );

        let envs: std::collections::HashMap<_, _> = std_cmd
            .get_envs()
            .filter_map(|(k, v)| v.map(|val| (k.to_os_string(), val.to_os_string())))
            .collect();

        assert_eq!(
            envs.get(std::ffi::OsStr::new("CLARITY_DESKTOP_MODE")),
            Some(&std::ffi::OsString::from("1"))
        );
        assert_eq!(
            envs.get(std::ffi::OsStr::new("CLARITY_DATA_DIR")),
            Some(&std::ffi::OsString::from(app_data))
        );
        assert!(
            envs.contains_key(std::ffi::OsStr::new("CLARITY_FFMPEG")),
            "bundled ffmpeg must be pinned by absolute path"
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

        // Spawn mock HTTP server responding 200 OK to /api/health
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
            .args(["/c", "ping -n 30 127.0.0.1 > nul"])
            .spawn()
            .unwrap();

        let mut manager = BackendProcessManager {
            job_handle: std::ptr::null_mut(),
            child: dummy_child,
            port,
            log_path: backend_log_path(std::env::temp_dir().as_path()),
        };

        let res = manager.wait_until_ready(5).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_wait_until_ready_detects_early_exit() {
        // A backend that dies at startup must be reported immediately, with the
        // log path, instead of burning the whole readiness timeout.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let free_port = listener.local_addr().unwrap().port();
        drop(listener);

        let dummy_child = tokio::process::Command::new("cmd")
            .args(["/c", "exit 3"])
            .spawn()
            .unwrap();

        let mut manager = BackendProcessManager {
            job_handle: std::ptr::null_mut(),
            child: dummy_child,
            port: free_port,
            log_path: backend_log_path(std::env::temp_dir().as_path()),
        };

        let start = std::time::Instant::now();
        let err = manager
            .wait_until_ready(30)
            .await
            .expect_err("exited process is never ready");
        assert!(start.elapsed() < Duration::from_secs(10), "error was too slow: {err}");
        assert!(err.contains("exited"), "unexpected message: {err}");
        assert!(err.contains("backend.log"), "error must name the log: {err}");
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
            log_path: backend_log_path(std::env::temp_dir().as_path()),
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
            log_path: backend_log_path(std::env::temp_dir().as_path()),
        };

        assert!(!manager.job_handle.is_null());
        manager.terminate();
        assert!(manager.job_handle.is_null());

        // Second terminate call should be safe / idempotent
        manager.terminate();
        assert!(manager.job_handle.is_null());
    }

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
}
