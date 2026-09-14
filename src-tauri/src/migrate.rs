//! One-time adoption of data folders created by earlier builds.
//!
//! Two historical locations exist: `<install drive>\Clarity-data` (a rule that
//! put data on the install drive but outside the install folder) and
//! `%LOCALAPPDATA%\Clarity` (what the installer hook used). Both can hold an
//! interpreter, a venv and gigabytes of weights, so the fix is to move them into
//! the install folder rather than download them again — but only when a rename
//! can do it, because copying 5 GB across volumes during startup is worse than
//! the duplication it avoids.

use std::path::{Path, PathBuf};

use crate::state::drive_root;

/// Folders an earlier build may have provisioned into.
pub fn legacy_roots(install_dir: &Path, local_dir: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(root) = drive_root(install_dir) {
        roots.push(Path::new(&root).join("Clarity-data"));
    }
    roots.push(local_dir.join("Clarity"));
    roots
}

/// True when both paths sit on the same Windows volume, so `rename` is instant.
pub fn same_volume(a: &Path, b: &Path) -> bool {
    match (drive_root(a), drive_root(b)) {
        (Some(left), Some(right)) => left.eq_ignore_ascii_case(&right),
        _ => false,
    }
}

/// Move every entry of `legacy` into `target` that is not already there, then
/// remove `legacy` if it ended up empty. Returns the moved entry names.
pub fn merge_into(legacy: &Path, target: &Path) -> Result<Vec<String>, String> {
    let entries = std::fs::read_dir(legacy)
        .map_err(|e| format!("cannot read {}: {e}", legacy.display()))?;
    let mut entries: Vec<std::fs::DirEntry> = entries
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("cannot read {}: {e}", legacy.display()))?;
    // The marker must land absolute-last (after setup.json): is_setup_complete
    // requires marker + env interpreter, so the target reads complete only
    // once every other entry has already moved.
    entries.sort_by_key(|entry| match entry.file_name().to_string_lossy().as_ref() {
        ".setup_complete" => 2,
        "setup.json" => 1,
        _ => 0,
    });
    let mut moved = Vec::new();

    for entry in entries {
        let from = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        let to = target.join(&name);
        // Never replace program files, and never mix two envs or model sets.
        if to.exists() {
            continue;
        }
        std::fs::rename(&from, &to)
            .map_err(|e| format!("cannot move {} to {}: {e}", from.display(), to.display()))?;
        moved.push(name);
    }

    let _ = std::fs::remove_dir(legacy);
    Ok(moved)
}

/// Adopt a legacy folder when it holds a finished runtime and the install folder
/// does not. Same volume only; everything else is left for the user to delete.
pub fn adopt_legacy_data(install_dir: &Path, local_dir: &Path) -> Vec<String> {
    if crate::setup::is_setup_complete(install_dir) {
        return Vec::new();
    }
    let mut log = Vec::new();
    for legacy in legacy_roots(install_dir, local_dir) {
        if !legacy.is_dir() || legacy == install_dir || !same_volume(&legacy, install_dir) {
            continue;
        }
        if !crate::setup::is_setup_complete(&legacy) {
            continue;
        }
        match merge_into(&legacy, install_dir) {
            Ok(moved) if !moved.is_empty() => {
                log.push(format!("adopted {} from {}", moved.join(", "), legacy.display()))
            }
            Ok(_) => {}
            Err(err) => log.push(format!("could not adopt {}: {err}", legacy.display())),
        }
    }
    log
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_moves_entries_without_overwriting_the_program() {
        let root = std::env::temp_dir().join("clarity_merge");
        let _ = std::fs::remove_dir_all(&root);
        let legacy = root.join("Clarity-data");
        let target = root.join("Clarity");
        std::fs::create_dir_all(legacy.join("models")).unwrap();
        std::fs::write(legacy.join("models").join("amt-s.pth"), b"weights").unwrap();
        std::fs::write(legacy.join("setup.json"), b"{}").unwrap();
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("clarity-desktop.exe"), b"program").unwrap();
        std::fs::write(legacy.join("clarity-desktop.exe"), b"stale").unwrap();

        let moved = merge_into(&legacy, &target).unwrap();

        assert!(moved.iter().any(|name| name == "models"));
        assert!(target.join("models").join("amt-s.pth").is_file());
        assert!(target.join("setup.json").is_file());
        assert_eq!(
            std::fs::read(target.join("clarity-desktop.exe")).unwrap(),
            b"program",
            "the installed program must survive a merge"
        );
        assert!(!legacy.join("models").exists());
        assert!(!legacy.join("setup.json").exists());
        // The skipped stale duplicate is left behind, so the folder survives
        // with only it; the installed program itself is untouched.
        assert!(legacy.is_dir());
        assert_eq!(
            std::fs::read(legacy.join("clarity-desktop.exe")).unwrap(),
            b"stale"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_merge_removes_an_emptied_legacy_folder() {
        let root = std::env::temp_dir().join("clarity_merge_emptied");
        let _ = std::fs::remove_dir_all(&root);
        let legacy = root.join("Clarity-data");
        let target = root.join("Clarity");
        std::fs::create_dir_all(legacy.join("models")).unwrap();
        std::fs::write(legacy.join("setup.json"), b"{}").unwrap();
        std::fs::write(legacy.join(".setup_complete"), b"done").unwrap();
        std::fs::create_dir_all(&target).unwrap();

        let moved = merge_into(&legacy, &target).unwrap();

        assert_eq!(moved.len(), 3);
        assert_eq!(moved.last().map(String::as_str), Some(".setup_complete"));
        assert!(!legacy.exists(), "an emptied legacy folder is removed");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_same_volume_compares_drive_roots() {
        assert!(same_volume(Path::new("D:\\Clarity-data"), Path::new("D:\\Clarity")));
        assert!(!same_volume(Path::new("C:\\Users\\x"), Path::new("D:\\Clarity")));
        assert!(!same_volume(Path::new("relative"), Path::new("D:\\Clarity")));
    }

    #[test]
    fn test_legacy_roots_are_the_two_historical_locations() {
        let roots = legacy_roots(
            Path::new("D:\\Clarity"),
            Path::new("C:\\Users\\x\\AppData\\Local"),
        );
        assert!(roots.contains(&PathBuf::from("D:\\Clarity-data")));
        assert!(roots.contains(&PathBuf::from("C:\\Users\\x\\AppData\\Local\\Clarity")));
    }

    #[test]
    fn test_adoption_skips_a_legacy_folder_that_was_never_finished() {
        // `local_dir.join("Clarity")` is the second legacy candidate, so this
        // exercises adopt_legacy_data end to end without touching a drive root.
        let root = std::env::temp_dir().join("clarity_merge_partial");
        let _ = std::fs::remove_dir_all(&root);
        let legacy = root.join("Clarity");
        std::fs::create_dir_all(legacy.join("env").join("Scripts")).unwrap();
        std::fs::write(legacy.join("env").join("Scripts").join("python.exe"), b"").unwrap();
        // No .setup_complete: the install that made this folder never finished.
        let install = root.join("install");
        std::fs::create_dir_all(&install).unwrap();

        let log = adopt_legacy_data(&install, &root);

        assert!(log.is_empty());
        assert!(legacy.join("env").is_dir(), "a half-finished folder is left alone");
        assert!(!install.join("env").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
