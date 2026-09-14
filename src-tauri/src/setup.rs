use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Progress event emitted during environment provisioning and setup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetupProgressEvent {
    pub stage: String,
    pub percent: f32,
    pub speed: String,
    pub message: String,
}

/// Thread-safe monotonic progress tracker that prevents progress percentage regressions.
#[derive(Debug)]
pub struct MonotonicProgressTracker {
    max_millipct: AtomicU32,
}

impl MonotonicProgressTracker {
    pub fn new(initial_pct: f32) -> Self {
        Self {
            max_millipct: AtomicU32::new((initial_pct * 1000.0).max(0.0) as u32),
        }
    }

    pub fn update(&self, new_pct: f32) -> f32 {
        let new_milli = (new_pct * 1000.0).max(0.0) as u32;
        self.max_millipct.fetch_max(new_milli, Ordering::SeqCst);
        let max_val = self.max_millipct.load(Ordering::SeqCst);
        (max_val as f32) / 1000.0
    }

    pub fn current(&self) -> f32 {
        (self.max_millipct.load(Ordering::SeqCst) as f32) / 1000.0
    }
}

/// Checks whether the production isolated Python environment and marker file exist in app_data_dir.
pub fn is_setup_complete(app_data_dir: &Path) -> bool {
    let marker = app_data_dir.join(".setup_complete");
    let python_win = app_data_dir.join("env").join("Scripts").join("python.exe");
    let python_unix = app_data_dir.join("env").join("bin").join("python");

    marker.is_file() && (python_win.is_file() || python_unix.is_file())
}

/// Checks whether a local development environment (.venv) is present in the current or parent directory.
pub fn has_dev_environment() -> bool {
    let dev_venv = Path::new(".venv").join("Scripts").join("python.exe");
    if dev_venv.is_file() {
        return true;
    }

    let parent_dev_venv = Path::new("..").join(".venv").join("Scripts").join("python.exe");
    if parent_dev_venv.is_file() {
        return true;
    }

    if let Ok(venv_val) = std::env::var("VIRTUAL_ENV") {
        if Path::new(&venv_val).join("Scripts").join("python.exe").is_file() {
            return true;
        }
    }

    false
}

/// Parses uv stdout/stderr lines to extract download/installation percentage and speed.
///
/// Returns `Some((percent, speed))` where speed is formatted like `"24.5MB/s"` or `"18.2 MB/s"`.
/// If only percentage is present, speed is an empty string `""`.
/// If no valid percentage is found, returns `None`.
pub fn parse_uv_progress_line(line: &str) -> Option<(f32, String)> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let mut percent: Option<f32> = None;
    let mut speed: Option<String> = None;

    // Search for percentage: token ending with '%'
    for word in &words {
        let clean = word.trim_matches(|c: char| {
            c == '[' || c == ']' || c == '(' || c == ')' || c == '{' || c == '}' || c == ','
        });
        if let Some(pct_str) = clean.strip_suffix('%') {
            if let Ok(val) = pct_str.parse::<f32>() {
                if (0.0..=100.0).contains(&val) {
                    percent = Some(val);
                    break;
                }
            }
        }
    }

    if let Some(pct) = percent {
        // Search for transfer speed ending with "/s"
        for (i, word) in words.iter().enumerate() {
            let clean = word.trim_matches(|c: char| {
                c == '(' || c == ')' || c == '[' || c == ']' || c == ',' || c == '{' || c == '}'
            });
            let lower = clean.to_lowercase();
            if lower.ends_with("/s") {
                // Check if numeric value is in the same word (e.g., "24.5MB/s", "12.3MiB/s")
                let prefix = &clean[..clean.len() - 2];
                let num_part = prefix.trim_end_matches(|c: char| c.is_alphabetic());
                if !num_part.is_empty() && num_part.parse::<f32>().is_ok() {
                    speed = Some(clean.to_string());
                    break;
                } else if i > 0 {
                    // Check if previous word is the numeric value (e.g., "18.2" "MB/s")
                    let prev = words[i - 1].trim_matches(|c: char| {
                        c == '(' || c == ')' || c == '[' || c == ']' || c == ','
                    });
                    if prev.parse::<f32>().is_ok() {
                        speed = Some(format!("{} {}", prev, clean));
                        break;
                    }
                }
            }
        }

        Some((pct, speed.unwrap_or_default()))
    } else {
        None
    }
}

/// Resolves the uv binary to use, checking `resources_dir` first, then `PATH`.
pub fn resolve_uv_bin(resources_dir: &Path) -> PathBuf {
    let bundled_win = resources_dir.join("uv.exe");
    if bundled_win.is_file() {
        return bundled_win;
    }
    let bundled_unix = resources_dir.join("uv");
    if bundled_unix.is_file() {
        return bundled_unix;
    }

    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let cand_win = dir.join("uv.exe");
            if cand_win.is_file() {
                return cand_win;
            }
            let cand_unix = dir.join("uv");
            if cand_unix.is_file() {
                return cand_unix;
            }
        }
    }

    if cfg!(windows) {
        PathBuf::from("uv.exe")
    } else {
        PathBuf::from("uv")
    }
}

/// Resolves the project root containing `pyproject.toml`.
pub fn resolve_project_root(resources_dir: &Path) -> PathBuf {
    if resources_dir.join("pyproject.toml").is_file() {
        return resources_dir.to_path_buf();
    }
    if let Some(parent) = resources_dir.parent() {
        if parent.join("pyproject.toml").is_file() {
            return parent.to_path_buf();
        }
        if let Some(grandparent) = parent.parent() {
            if grandparent.join("pyproject.toml").is_file() {
                return grandparent.to_path_buf();
            }
        }
    }
    if Path::new("pyproject.toml").is_file() {
        return PathBuf::from(".");
    }
    if Path::new("..").join("pyproject.toml").is_file() {
        return PathBuf::from("..");
    }
    if let Ok(current_dir) = std::env::current_dir() {
        if current_dir.join("pyproject.toml").is_file() {
            return current_dir;
        }
        if let Some(parent) = current_dir.parent() {
            if parent.join("pyproject.toml").is_file() {
                return parent.to_path_buf();
            }
            if let Some(grandparent) = parent.parent() {
                if grandparent.join("pyproject.toml").is_file() {
                    return grandparent.to_path_buf();
                }
            }
        }
    }
    PathBuf::from(".")
}

/// Searches an installation directory for the installed Python executable.
///
/// The answer is canonicalised on purpose. `uv python install --install-dir`
/// stores the runtime in a versioned folder and adds a versionless directory
/// symlink as an alias; a sorted scan meets the alias first (`-` sorts before
/// `.`). Passing that alias onwards to `uv venv` makes uv create the venv's
/// `Scripts/python.exe` link through a reparse point, which Windows refuses with
/// `ERROR_ACCESS_DENIED` — the failure that looked like a permissions problem.
pub fn find_python_executable(dir: &Path) -> Option<PathBuf> {
    find_python_executable_raw(dir).map(canonical_executable)
}

fn find_python_executable_raw(dir: &Path) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }

    let direct_win = dir.join("python.exe");
    if direct_win.is_file() {
        return Some(direct_win);
    }
    let direct_unix = dir.join("bin").join("python");
    if direct_unix.is_file() {
        return Some(direct_unix);
    }
    let scripts_win = dir.join("Scripts").join("python.exe");
    if scripts_win.is_file() {
        return Some(scripts_win);
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|read| read.flatten().map(|entry| entry.path()).collect())
        .unwrap_or_default();
    // Filesystem order is not stable across platforms; sort so retries match.
    entries.sort();

    for path in entries {
        // `.temp` and `.lock` are uv's own staging artefacts, not runtimes.
        if !path.is_dir() || path.file_name().is_some_and(|name| name.to_string_lossy().starts_with('.'))
        {
            continue;
        }
        for candidate in [
            path.join("python.exe"),
            path.join("install").join("python.exe"),
            path.join("Scripts").join("python.exe"),
            path.join("bin").join("python"),
        ] {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

/// Resolve symlinks and junctions, dropping the `\\?\` verb that Windows
/// `canonicalize` prepends so the path stays usable in argv and error text.
fn canonical_executable(path: PathBuf) -> PathBuf {
    let resolved = match std::fs::canonicalize(&path) {
        Ok(real) => real,
        Err(_) => return path,
    };
    if !resolved.is_file() {
        return path;
    }
    let text = resolved.to_string_lossy().to_string();
    match text.strip_prefix(r"\\?\") {
        Some(plain) => PathBuf::from(plain),
        None => resolved,
    }
}

/// Resolves the python executable inside the target virtual environment.
pub fn resolve_venv_python(env_dir: &Path) -> PathBuf {
    let win = env_dir.join("Scripts").join("python.exe");
    if win.is_file() {
        return win;
    }
    let unix = env_dir.join("bin").join("python");
    if unix.is_file() {
        return unix;
    }
    if cfg!(windows) {
        win
    } else {
        unix
    }
}

/// Processes an async stream, parsing `\r` and `\n` delimited output lines in real-time.
///
/// Employs `tracker` to guarantee monotonically increasing progress, and retains recent
/// output lines in `output_history` for diagnostics on failure.
pub async fn process_stream<R, F>(
    mut reader: R,
    stage: &str,
    base_pct: f32,
    scale_pct: f32,
    tracker: Arc<MonotonicProgressTracker>,
    output_history: Arc<Mutex<VecDeque<String>>>,
    on_progress: Arc<F>,
) where
    R: tokio::io::AsyncRead + Unpin,
    F: Fn(SetupProgressEvent) + Send + Sync + 'static,
{
    use tokio::io::AsyncReadExt;
    let mut buf = [0u8; 1024];
    let mut line_buf = String::new();

    while let Ok(n) = reader.read(&mut buf).await {
        if n == 0 {
            break;
        }
        let chunk = String::from_utf8_lossy(&buf[..n]);
        for ch in chunk.chars() {
            if ch == '\n' || ch == '\r' {
                let trimmed = line_buf.trim();
                if !trimmed.is_empty() {
                    {
                        let mut history = output_history.lock().unwrap();
                        if history.len() >= 15 {
                            history.pop_front();
                        }
                        history.push_back(trimmed.to_string());
                    }

                    if let Some((uv_pct, speed)) = parse_uv_progress_line(trimmed) {
                        let mapped_pct =
                            base_pct + (uv_pct.clamp(0.0, 100.0) / 100.0) * scale_pct;
                        let rounded = (mapped_pct * 10.0).round() / 10.0;
                        let pct = tracker.update(rounded);
                        on_progress(SetupProgressEvent {
                            stage: stage.to_string(),
                            percent: pct,
                            speed,
                            message: trimmed.to_string(),
                        });
                    } else {
                        let pct = tracker.current();
                        on_progress(SetupProgressEvent {
                            stage: stage.to_string(),
                            percent: pct,
                            speed: String::new(),
                            message: trimmed.to_string(),
                        });
                    }
                }
                line_buf.clear();
            } else {
                line_buf.push(ch);
            }
        }
    }

    let trimmed = line_buf.trim();
    if !trimmed.is_empty() {
        {
            let mut history = output_history.lock().unwrap();
            if history.len() >= 15 {
                history.pop_front();
            }
            history.push_back(trimmed.to_string());
        }

        if let Some((uv_pct, speed)) = parse_uv_progress_line(trimmed) {
            let mapped_pct = base_pct + (uv_pct.clamp(0.0, 100.0) / 100.0) * scale_pct;
            let rounded = (mapped_pct * 10.0).round() / 10.0;
            let pct = tracker.update(rounded);
            on_progress(SetupProgressEvent {
                stage: stage.to_string(),
                percent: pct,
                speed,
                message: trimmed.to_string(),
            });
        }
    }
}

/// Spawns a command, attaches progress monitoring to stdout and stderr, and waits for completion.
async fn run_command_with_progress<F>(
    mut cmd: tokio::process::Command,
    stage: &str,
    base_pct: f32,
    scale_pct: f32,
    on_progress: Arc<F>,
) -> Result<(), String>
where
    F: Fn(SetupProgressEvent) + Send + Sync + 'static,
{
    #[cfg(windows)]
    {
        // CREATE_NO_WINDOW: eliminate flickering console popups
        cmd.creation_flags(0x08000000);
    }

    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn command for stage '{}': {}", stage, e))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let tracker = Arc::new(MonotonicProgressTracker::new(base_pct));
    let output_history = Arc::new(Mutex::new(VecDeque::with_capacity(16)));

    let p1 = on_progress.clone();
    let s1 = stage.to_string();
    let t1 = tracker.clone();
    let h1 = output_history.clone();
    let stdout_handle = tokio::spawn(async move {
        if let Some(reader) = stdout {
            process_stream(reader, &s1, base_pct, scale_pct, t1, h1, p1).await;
        }
    });

    let p2 = on_progress.clone();
    let s2 = stage.to_string();
    let t2 = tracker.clone();
    let h2 = output_history.clone();
    let stderr_handle = tokio::spawn(async move {
        if let Some(reader) = stderr {
            process_stream(reader, &s2, base_pct, scale_pct, t2, h2, p2).await;
        }
    });

    let (status, _, _) = tokio::join!(child.wait(), stdout_handle, stderr_handle);
    let status = status.map_err(|e| format!("Failed to wait on stage '{}': {}", stage, e))?;

    if !status.success() {
        let history = output_history.lock().unwrap();
        let diagnostic = if history.is_empty() {
            "No diagnostic output recorded.".to_string()
        } else {
            history.iter().cloned().collect::<Vec<_>>().join("\n  ")
        };
        return Err(format!(
            "Stage '{}' command failed with exit code: {:?}.\nRecent output:\n  {}",
            stage,
            status.code(),
            diagnostic
        ));
    }

    Ok(())
}

/// Provisions the standalone Python runtime that the setup wizard needs.
///
/// Everything past this point — venv, dependencies, weights, verification — is
/// driven by `video_upscaler.desktop` over HTTP, because provisioning is product
/// logic and the window should not own it. This is the one step the shell must
/// do itself: it has to happen before any interpreter exists to run the wizard.
pub async fn ensure_python_runtime(
    app_data_dir: &Path,
    resources_dir: &Path,
    on_progress: impl Fn(SetupProgressEvent) + Send + Sync + 'static,
) -> Result<PathBuf, String> {
    let on_progress = Arc::new(on_progress);
    let python_install_dir = app_data_dir.join("python");
    std::fs::create_dir_all(&python_install_dir)
        .map_err(|e| format!("Failed to create python install directory: {}", e))?;

    if let Some(existing) = find_python_executable(&python_install_dir) {
        on_progress(SetupProgressEvent {
            stage: "python".to_string(),
            percent: 100.0,
            speed: String::new(),
            message: "Python runtime already installed".to_string(),
        });
        return Ok(existing);
    }

    let uv_bin = resolve_uv_bin(resources_dir);
    let mut cmd = tokio::process::Command::new(&uv_bin);
    cmd.args([
        "python",
        "install",
        "3.11",
        "--no-bin",
        "--install-dir",
        &python_install_dir.to_string_lossy(),
    ]);
    cmd.env("UV_PYTHON_INSTALL_DIR", &python_install_dir);
    cmd.env("PYTHONUNBUFFERED", "1");

    run_command_with_progress(cmd, "python", 0.0, 100.0, on_progress).await?;

    find_python_executable(&python_install_dir).ok_or_else(|| {
        format!(
            "uv reported success but no interpreter was found under {}",
            python_install_dir.display()
        )
    })
}

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
    // stdout and stderr are different types, so box them to a common reader.
    for pipe in [
        child.stdout.take().map(|p| Box::new(p) as Box<dyn tokio::io::AsyncRead + Unpin + Send>),
        child.stderr.take().map(|p| Box::new(p) as Box<dyn tokio::io::AsyncRead + Unpin + Send>),
    ]
    .into_iter()
    .flatten()
    {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn test_setup_progress_event_serialization() {
        let event = SetupProgressEvent {
            stage: "pip".to_string(),
            percent: 65.5,
            speed: "24.5MB/s".to_string(),
            message: "Installing torch".to_string(),
        };

        let json = serde_json::to_string(&event).expect("serialize event");
        let deserialized: SetupProgressEvent =
            serde_json::from_str(&json).expect("deserialize event");
        assert_eq!(event, deserialized);
    }

    #[test]
    fn test_monotonic_progress_tracker() {
        let tracker = MonotonicProgressTracker::new(10.0);
        assert_eq!(tracker.current(), 10.0);

        // Advance to 35.5
        assert_eq!(tracker.update(35.5), 35.5);
        assert_eq!(tracker.current(), 35.5);

        // Attempt regression to 20.0 - should remain at 35.5
        assert_eq!(tracker.update(20.0), 35.5);
        assert_eq!(tracker.current(), 35.5);

        // Advance to 75.0
        assert_eq!(tracker.update(75.0), 75.0);
        assert_eq!(tracker.current(), 75.0);
    }

    #[test]
    fn test_is_setup_complete_missing_all() {
        let temp_dir = std::env::temp_dir().join("clarity_test_setup_missing_all");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        assert!(!is_setup_complete(&temp_dir));
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_is_setup_complete_missing_python() {
        let temp_dir = std::env::temp_dir().join("clarity_test_setup_missing_py");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Write marker file only
        std::fs::write(temp_dir.join(".setup_complete"), b"1").unwrap();

        assert!(!is_setup_complete(&temp_dir));
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_is_setup_complete_missing_marker() {
        let temp_dir = std::env::temp_dir().join("clarity_test_setup_missing_marker");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let scripts_dir = temp_dir.join("env").join("Scripts");
        std::fs::create_dir_all(&scripts_dir).unwrap();
        std::fs::write(scripts_dir.join("python.exe"), b"mock").unwrap();

        assert!(!is_setup_complete(&temp_dir));
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_is_setup_complete_both_present() {
        let temp_dir = std::env::temp_dir().join("clarity_test_setup_both_present");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let scripts_dir = temp_dir.join("env").join("Scripts");
        std::fs::create_dir_all(&scripts_dir).unwrap();
        std::fs::write(scripts_dir.join("python.exe"), b"mock").unwrap();
        std::fs::write(temp_dir.join(".setup_complete"), b"1").unwrap();

        assert!(is_setup_complete(&temp_dir));
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_parse_uv_progress_line_standard() {
        let line = "Downloading python-3.11.9-windows-x86_64.tar.gz (25.4MB) 85% 24.5MB/s 0s";
        let parsed = parse_uv_progress_line(line);
        assert_eq!(parsed, Some((85.0, "24.5MB/s".to_string())));
    }

    #[test]
    fn test_parse_uv_progress_line_with_brackets_and_mibs() {
        let line = "[========================================] 45.5% 12.3MiB/s";
        let parsed = parse_uv_progress_line(line);
        assert_eq!(parsed, Some((45.5, "12.3MiB/s".to_string())));
    }

    #[test]
    fn test_parse_uv_progress_line_separated_speed() {
        let line = "Fetching packages [ 50% ] (18.2 MB/s)";
        let parsed = parse_uv_progress_line(line);
        assert_eq!(parsed, Some((50.0, "18.2 MB/s".to_string())));
    }

    #[test]
    fn test_parse_uv_progress_line_percent_only() {
        let line = "Extracting packages 100%";
        let parsed = parse_uv_progress_line(line);
        assert_eq!(parsed, Some((100.0, "".to_string())));
    }

    #[test]
    fn test_parse_uv_progress_line_zero_percent() {
        let line = "Starting download 0%";
        let parsed = parse_uv_progress_line(line);
        assert_eq!(parsed, Some((0.0, "".to_string())));
    }

    #[test]
    fn test_parse_uv_progress_line_out_of_range_percent() {
        let line = "Invalid percent 150%";
        assert_eq!(parse_uv_progress_line(line), None);
    }

    #[test]
    fn test_parse_uv_progress_line_non_progress_text() {
        let line = "Resolved 42 packages in 12ms";
        assert_eq!(parse_uv_progress_line(line), None);

        let line2 = "error: package not found";
        assert_eq!(parse_uv_progress_line(line2), None);

        let line3 = "";
        assert_eq!(parse_uv_progress_line(line3), None);

        let line4 = "   \t \r \n ";
        assert_eq!(parse_uv_progress_line(line4), None);
    }

    #[test]
    fn test_resolve_project_root_finds_pyproject() {
        let root = resolve_project_root(Path::new("non_existent_resources"));
        let pyproject = root.join("pyproject.toml");
        assert!(pyproject.is_file(), "Expected pyproject.toml at {:?}", pyproject);
    }

    #[test]
    fn test_resolve_project_root_grandparent_resolution() {
        let temp_root = std::env::temp_dir().join("clarity_test_grandparent_root");
        let _ = std::fs::remove_dir_all(&temp_root);
        let nested_resources = temp_root.join("src-tauri").join("resources");
        std::fs::create_dir_all(&nested_resources).unwrap();
        std::fs::write(temp_root.join("pyproject.toml"), b"# dummy").unwrap();

        let resolved = resolve_project_root(&nested_resources);
        assert_eq!(resolved, temp_root);

        let _ = std::fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn test_resolve_project_root_direct_in_resources() {
        let temp_resources = std::env::temp_dir().join("clarity_test_direct_resources");
        let _ = std::fs::remove_dir_all(&temp_resources);
        std::fs::create_dir_all(&temp_resources).unwrap();
        std::fs::write(temp_resources.join("pyproject.toml"), b"# dummy").unwrap();

        let resolved = resolve_project_root(&temp_resources);
        assert_eq!(resolved, temp_resources);

        let _ = std::fs::remove_dir_all(&temp_resources);
    }

    #[test]
    fn test_resolve_uv_bin_returns_valid_or_fallback() {
        let uv_path = resolve_uv_bin(Path::new("non_existent_resources"));
        assert!(!uv_path.as_os_str().is_empty());
    }

    #[test]
    fn test_find_python_executable_in_nested_dir() {
        let temp_dir = std::env::temp_dir().join("clarity_test_find_python");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let nested = temp_dir.join("cpython-3.11.9-windows-x86_64-none");
        std::fs::create_dir_all(&nested).unwrap();
        let py_bin = nested.join("python.exe");
        std::fs::write(&py_bin, b"mock").unwrap();

        let found = find_python_executable(&temp_dir);
        assert_eq!(found, Some(canonical_executable(py_bin)));
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_find_python_executable_skips_uv_staging_dirs() {
        let temp_dir = std::env::temp_dir().join("clarity_test_find_python_staging");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let real = temp_dir.join("cpython-3.11.15-windows-x86_64-none");
        let staging = temp_dir.join(".temp").join("download");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(real.join("python.exe"), b"real").unwrap();
        // Sorting would not put `.temp` last, and it does hold a python.exe.
        std::fs::write(staging.join("python.exe"), b"partial").unwrap();

        let found = find_python_executable(&temp_dir);
        assert_eq!(found, Some(canonical_executable(real.join("python.exe"))));
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_canonical_executable_drops_the_windows_verbatim_prefix() {
        let temp_dir = std::env::temp_dir().join("clarity_test_canonical_python");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let py = temp_dir.join("python.exe");
        std::fs::write(&py, b"mock").unwrap();

        let resolved = canonical_executable(py.clone());
        assert!(resolved.is_file(), "canonical path must still exist: {resolved:?}");
        assert!(
            !resolved.to_string_lossy().starts_with(r"\\?\"),
            "uv must not be handed a verbatim path: {resolved:?}"
        );
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_find_python_executable_resolves_the_versionless_alias() {
        // uv installs the runtime under the versioned name and adds a symlink
        // alias without the patch version. Only the real path may reach uv.
        let temp_dir = std::env::temp_dir().join("clarity_test_alias_python");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let real = temp_dir.join("cpython-3.11.15-windows-x86_64-none");
        std::fs::create_dir_all(&real).unwrap();
        let target = real.join("python.exe");
        std::fs::write(&target, b"mock").unwrap();

        let alias = temp_dir.join("cpython-3.11-windows-x86_64-none");
        #[cfg(windows)]
        {
            if std::os::windows::fs::symlink_dir(&real, &alias).is_err() {
                // No developer mode / privilege: nothing to assert here.
                let _ = std::fs::remove_dir_all(&temp_dir);
                return;
            }
        }
        #[cfg(not(windows))]
        {
            if std::os::unix::fs::symlink(&real, &alias).is_err() {
                let _ = std::fs::remove_dir_all(&temp_dir);
                return;
            }
        }

        let found = find_python_executable(&temp_dir).expect("alias interpreter must be found");
        assert_eq!(
            canonical_executable(found.clone()),
            canonical_executable(target),
            "the alias must resolve to the versioned runtime"
        );
        assert!(
            !found.to_string_lossy().contains("3.11-windows"),
            "found path is still the alias: {found:?}"
        );
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_find_python_executable_empty_dir() {
        let temp_dir = std::env::temp_dir().join("clarity_test_find_python_empty");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        assert_eq!(find_python_executable(&temp_dir), None);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_resolve_venv_python_path() {
        let env_dir = Path::new("dummy_env");
        let py = resolve_venv_python(env_dir);
        if cfg!(windows) {
            assert_eq!(py, env_dir.join("Scripts").join("python.exe"));
        } else {
            assert_eq!(py, env_dir.join("bin").join("python"));
        }
    }

    #[tokio::test]
    async fn test_process_stream_emits_events_and_preserves_monotonic_progress() {
        use std::io::Cursor;
        // Output with progress, then non-progress informational line, then regression attempt
        let sample_output =
            "Downloading 50% 10.0MB/s\rResolved 42 dependencies\nDownloading 30% 5.0MB/s\rCompleted 100%\n";
        let cursor = Cursor::new(sample_output.as_bytes());

        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();

        let tracker = Arc::new(MonotonicProgressTracker::new(10.0));
        let history = Arc::new(Mutex::new(VecDeque::new()));

        process_stream(
            cursor,
            "test_stage",
            10.0,
            20.0,
            tracker,
            history.clone(),
            Arc::new(move |evt| {
                events_clone.lock().unwrap().push(evt);
            }),
        )
        .await;

        let captured = events.lock().unwrap().clone();
        assert!(!captured.is_empty());
        assert_eq!(captured[0].stage, "test_stage");
        // base_pct 10.0 + (50% * 20.0) = 20.0
        assert_eq!(captured[0].percent, 20.0);
        assert_eq!(captured[0].speed, "10.0MB/s");

        // The second event was a non-progress line ("Resolved 42 dependencies")
        // It must NOT have regressed to base_pct (10.0), but remained at 20.0!
        assert_eq!(captured[1].message, "Resolved 42 dependencies");
        assert_eq!(captured[1].percent, 20.0);

        // The third event was a regression attempt (30% -> mapped 16.0%)
        // It must stay at 20.0!
        assert_eq!(captured[2].percent, 20.0);

        // History buffer must record recent lines
        let recorded = history.lock().unwrap();
        assert!(recorded.contains(&"Resolved 42 dependencies".to_string()));
    }

    #[tokio::test]
    async fn test_ensure_python_runtime_reuses_existing_interpreter() {
        let temp_app_data = std::env::temp_dir().join("clarity_setup_reuse_app_data");
        let temp_resources = std::env::temp_dir().join("clarity_setup_reuse_res");
        let _ = std::fs::remove_dir_all(&temp_app_data);
        let _ = std::fs::remove_dir_all(&temp_resources);
        std::fs::create_dir_all(&temp_app_data).unwrap();
        std::fs::create_dir_all(&temp_resources).unwrap();

        // A real interpreter is already there: uv must not be invoked at all, so
        // even a garbage uv cannot break a relaunch.
        let python_dir = temp_app_data.join("python");
        std::fs::create_dir_all(&python_dir).unwrap();
        let existing = python_dir.join(if cfg!(windows) { "python.exe" } else { "python" });
        std::fs::write(&existing, b"").unwrap();
        std::fs::write(
            if cfg!(windows) { temp_resources.join("uv.exe") } else { temp_resources.join("uv") },
            b"not_an_executable",
        )
        .unwrap();

        let res = ensure_python_runtime(&temp_app_data, &temp_resources, |_evt| {}).await;
        assert_eq!(res.as_deref(), Ok(existing.as_path()));

        let _ = std::fs::remove_dir_all(&temp_app_data);
        let _ = std::fs::remove_dir_all(&temp_resources);
    }

    #[tokio::test]
    async fn test_ensure_python_runtime_fails_with_diagnostics() {
        let temp_app_data = std::env::temp_dir().join("clarity_setup_fail_app_data");
        let temp_resources = std::env::temp_dir().join("clarity_setup_fail_res");
        let _ = std::fs::remove_dir_all(&temp_app_data);
        let _ = std::fs::remove_dir_all(&temp_resources);
        std::fs::create_dir_all(&temp_app_data).unwrap();
        std::fs::create_dir_all(&temp_resources).unwrap();

        std::fs::write(
            if cfg!(windows) { temp_resources.join("uv.exe") } else { temp_resources.join("uv") },
            b"not_an_executable",
        )
        .unwrap();

        let res = ensure_python_runtime(&temp_app_data, &temp_resources, |_evt| {}).await;
        assert!(res.is_err(), "expected failure, got {:?}", res);
        assert!(temp_app_data.join("python").is_dir(), "install dir must exist");
        // No runtime, no marker: the wizard must stay reachable afterwards.
        assert!(!temp_app_data.join(".setup_complete").exists());

        let _ = std::fs::remove_dir_all(&temp_app_data);
        let _ = std::fs::remove_dir_all(&temp_resources);
    }

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
}
