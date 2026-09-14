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

/// Checks whether the Python environment and setup have already been completed.
///
/// Setup is deemed complete if and only if:
/// 1. The marker file `%LOCALAPPDATA%\Clarity\.setup_complete` exists.
/// 2. The virtual environment's Python executable exists.
pub fn is_setup_complete(app_data_dir: &Path) -> bool {
    let marker = app_data_dir.join(".setup_complete");
    let python_win = app_data_dir.join("env").join("Scripts").join("python.exe");
    let python_unix = app_data_dir.join("env").join("bin").join("python");

    marker.is_file() && (python_win.is_file() || python_unix.is_file())
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
pub fn find_python_executable(dir: &Path) -> Option<PathBuf> {
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

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let cand_win = path.join("python.exe");
                if cand_win.is_file() {
                    return Some(cand_win);
                }
                let cand_install = path.join("install").join("python.exe");
                if cand_install.is_file() {
                    return Some(cand_install);
                }
                let cand_scripts = path.join("Scripts").join("python.exe");
                if cand_scripts.is_file() {
                    return Some(cand_scripts);
                }
                let cand_bin = path.join("bin").join("python");
                if cand_bin.is_file() {
                    return Some(cand_bin);
                }
            }
        }
    }

    None
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

/// Runs the complete first-run setup orchestrator:
///
/// 1. Creates `%LOCALAPPDATA%\Clarity\{python, env, models}`.
/// 2. Invokes `uv.exe python install 3.11 --no-bin --install-dir <app_data_dir>/python`.
/// 3. Invokes `uv.exe venv --allow-existing <app_data_dir>/env --python <installed_python>`.
/// 4. Invokes `uv.exe pip install` pointing to the project root and appropriate PyTorch index.
/// 5. Invokes `main.py --download-models essential` to fetch Real-CUGAN and AMT-S weights.
/// 6. Writes `%LOCALAPPDATA%\Clarity\.setup_complete`.
pub async fn run_setup<F>(
    app_data_dir: &Path,
    resources_dir: &Path,
    target: crate::hardware::GpuTarget,
    on_progress: F,
) -> Result<(), String>
where
    F: Fn(SetupProgressEvent) + Send + Sync + 'static,
{
    let on_progress = Arc::new(on_progress);

    // Step 1: Create directories
    let python_install_dir = app_data_dir.join("python");
    let env_dir = app_data_dir.join("env");
    let models_dir = app_data_dir.join("models");

    std::fs::create_dir_all(&python_install_dir)
        .map_err(|e| format!("Failed to create python install directory: {}", e))?;
    std::fs::create_dir_all(&env_dir)
        .map_err(|e| format!("Failed to create virtual environment directory: {}", e))?;
    std::fs::create_dir_all(&models_dir)
        .map_err(|e| format!("Failed to create models directory: {}", e))?;

    on_progress(SetupProgressEvent {
        stage: "init".to_string(),
        percent: 5.0,
        speed: String::new(),
        message: "Created runtime and models directories".to_string(),
    });

    let uv_bin = resolve_uv_bin(resources_dir);

    // Step 2: uv python install 3.11 --no-bin --install-dir <python_install_dir>
    on_progress(SetupProgressEvent {
        stage: "python".to_string(),
        percent: 10.0,
        speed: String::new(),
        message: "Downloading and installing Python 3.11 runtime...".to_string(),
    });

    let mut py_install_cmd = tokio::process::Command::new(&uv_bin);
    py_install_cmd.args([
        "python",
        "install",
        "3.11",
        "--no-bin",
        "--install-dir",
        &python_install_dir.to_string_lossy(),
    ]);

    run_command_with_progress(py_install_cmd, "python", 10.0, 15.0, on_progress.clone()).await?;

    // Step 3: uv venv --allow-existing <env_dir> --python <installed_or_discovered_python>
    on_progress(SetupProgressEvent {
        stage: "venv".to_string(),
        percent: 25.0,
        speed: String::new(),
        message: "Creating isolated virtual environment...".to_string(),
    });

    let mut venv_cmd = tokio::process::Command::new(&uv_bin);
    venv_cmd.args(["venv", "--allow-existing"]).arg(&env_dir);

    if let Some(installed_py) = find_python_executable(&python_install_dir) {
        venv_cmd.arg("--python").arg(&installed_py);
    } else {
        venv_cmd.arg("--python").arg("3.11");
    }

    venv_cmd.env("UV_PYTHON_INSTALL_DIR", &python_install_dir);

    run_command_with_progress(venv_cmd, "venv", 25.0, 10.0, on_progress.clone()).await?;

    // Step 4: uv pip install into that venv
    on_progress(SetupProgressEvent {
        stage: "dependencies".to_string(),
        percent: 35.0,
        speed: String::new(),
        message: "Installing Clarity dependencies and PyTorch...".to_string(),
    });

    let venv_python = resolve_venv_python(&env_dir);
    let mut pip_cmd = tokio::process::Command::new(&uv_bin);
    pip_cmd.args(["pip", "install", "--python", &venv_python.to_string_lossy()]);

    if let crate::hardware::GpuTarget::NvidiaCuda { .. } = target {
        pip_cmd.args(["--extra-index-url", "https://download.pytorch.org/whl/cu126"]);
    }

    let project_root = resolve_project_root(resources_dir);
    pip_cmd.args(["-e", &project_root.to_string_lossy()]);

    run_command_with_progress(pip_cmd, "dependencies", 35.0, 45.0, on_progress.clone()).await?;

    // Step 5: Download essential models
    on_progress(SetupProgressEvent {
        stage: "models".to_string(),
        percent: 80.0,
        speed: String::new(),
        message: "Fetching essential AI model weights (Real-CUGAN & AMT-S)...".to_string(),
    });

    let main_py = project_root.join("main.py");
    let mut model_cmd = tokio::process::Command::new(&venv_python);
    if main_py.is_file() {
        model_cmd.arg(&main_py).args(["--download-models", "essential"]);
    } else {
        model_cmd.args(["-m", "video_upscaler.cli", "--download-models", "essential"]);
    }
    model_cmd.env("CLARITY_MODELS_DIR", &models_dir);
    model_cmd.env("PYTHONUNBUFFERED", "1");

    run_command_with_progress(model_cmd, "models", 80.0, 19.0, on_progress.clone()).await?;

    // Step 6: Write marker file .setup_complete
    let marker = app_data_dir.join(".setup_complete");
    std::fs::write(&marker, b"1")
        .map_err(|e| format!("Failed to write setup completion marker: {}", e))?;

    on_progress(SetupProgressEvent {
        stage: "complete".to_string(),
        percent: 100.0,
        speed: String::new(),
        message: "Setup completed successfully! Ready to launch.".to_string(),
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::GpuTarget;
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
        assert_eq!(found, Some(py_bin));
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
    async fn test_run_setup_invalid_binary_fails_with_diagnostics() {
        let temp_app_data = std::env::temp_dir().join("clarity_test_setup_diag_app_data");
        let temp_resources = std::env::temp_dir().join("clarity_test_setup_diag_res");
        let _ = std::fs::remove_dir_all(&temp_app_data);
        let _ = std::fs::remove_dir_all(&temp_resources);
        std::fs::create_dir_all(&temp_app_data).unwrap();
        std::fs::create_dir_all(&temp_resources).unwrap();

        // Create an invalid dummy executable that fails
        let fake_uv = if cfg!(windows) {
            temp_resources.join("uv.exe")
        } else {
            temp_resources.join("uv")
        };
        std::fs::write(&fake_uv, b"not_an_executable").unwrap();

        let res = run_setup(
            &temp_app_data,
            &temp_resources,
            GpuTarget::CpuFallback,
            |_evt| {},
        )
        .await;

        assert!(res.is_err());
        let err_msg = res.unwrap_err();
        // Error message should identify the failure
        assert!(
            err_msg.contains("failed") || err_msg.contains("Failed"),
            "Expected failure message, got: {}",
            err_msg
        );

        // Verify directories were created
        assert!(temp_app_data.join("python").is_dir());
        assert!(temp_app_data.join("env").is_dir());
        assert!(temp_app_data.join("models").is_dir());
        // Verify setup marker was NOT written on failure
        assert!(!temp_app_data.join(".setup_complete").exists());

        let _ = std::fs::remove_dir_all(&temp_app_data);
        let _ = std::fs::remove_dir_all(&temp_resources);
    }

    #[tokio::test]
    async fn test_run_setup_retry_venv_allow_existing() {
        let temp_app_data = std::env::temp_dir().join("clarity_test_setup_retry_app_data");
        let temp_resources = std::env::temp_dir().join("clarity_test_setup_retry_res");
        let _ = std::fs::remove_dir_all(&temp_app_data);
        let _ = std::fs::remove_dir_all(&temp_resources);
        std::fs::create_dir_all(&temp_app_data).unwrap();
        std::fs::create_dir_all(&temp_resources).unwrap();

        // Pre-create the env directory simulating a prior partial run
        let existing_env = temp_app_data.join("env");
        std::fs::create_dir_all(&existing_env).unwrap();
        std::fs::write(existing_env.join("partial.txt"), b"dummy").unwrap();

        let fake_uv = if cfg!(windows) {
            temp_resources.join("uv.exe")
        } else {
            temp_resources.join("uv")
        };
        std::fs::write(&fake_uv, b"not_a_binary").unwrap();

        let res = run_setup(
            &temp_app_data,
            &temp_resources,
            GpuTarget::CpuFallback,
            |_evt| {},
        )
        .await;

        assert!(res.is_err());
        // Ensure existing env was preserved and not broken by directory creation
        assert!(existing_env.join("partial.txt").exists());

        let _ = std::fs::remove_dir_all(&temp_app_data);
        let _ = std::fs::remove_dir_all(&temp_resources);
    }
}
