use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuTarget {
    NvidiaCuda { device_name: String },
    CpuFallback,
}

/// Parses the output of `nvidia-smi -L` to extract the primary GPU device name.
///
/// Typical output format from nvidia-smi:
/// `GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-...)`
///
/// Returns `Some(device_name)` if a valid GPU is found, or `None` otherwise.
pub fn parse_nvidia_smi_output(output: &str) -> Option<String> {
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(after_gpu) = trimmed.strip_prefix("GPU ") {
            if let Some(colon_idx) = after_gpu.find(':') {
                let index_part = after_gpu[..colon_idx].trim();
                if !index_part.is_empty() && index_part.chars().all(|c| c.is_ascii_digit()) {
                    let mut device_info = after_gpu[colon_idx + 1..].trim();

                    // Strip trailing UUID segment if present: e.g. " (UUID: GPU-...)"
                    if let Some(uuid_idx) = device_info.find("(UUID:") {
                        device_info = device_info[..uuid_idx].trim();
                    } else if let Some(uuid_idx) = device_info.to_lowercase().find("(uuid:") {
                        device_info = device_info[..uuid_idx].trim();
                    }

                    if !device_info.is_empty() {
                        return Some(device_info.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Helper function to invoke `nvidia-smi -L`, checking standard paths on Windows.
fn run_nvidia_smi() -> Option<String> {
    let candidate_binaries = if cfg!(windows) {
        vec![
            "nvidia-smi".to_string(),
            r"C:\Windows\System32\nvidia-smi.exe".to_string(),
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe".to_string(),
        ]
    } else {
        vec!["nvidia-smi".to_string()]
    };

    for bin in candidate_binaries {
        let mut cmd = std::process::Command::new(&bin);
        cmd.arg("-L");

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        if let Ok(output) = cmd.output() {
            if output.status.success() {
                return Some(String::from_utf8_lossy(&output.stdout).to_string());
            }
        }
    }

    None
}

/// Detects the target GPU platform, returning `GpuTarget::NvidiaCuda` if an NVIDIA GPU
/// is detected via `nvidia-smi -L`, or `GpuTarget::CpuFallback` otherwise.
pub fn detect_gpu_target() -> GpuTarget {
    if let Some(stdout) = run_nvidia_smi() {
        if let Some(device_name) = parse_nvidia_smi_output(&stdout) {
            return GpuTarget::NvidiaCuda { device_name };
        }
    }
    GpuTarget::CpuFallback
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nvidia_smi_output_standard_rtx3080() {
        let sample = "GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-5c4d3e21-0000-1111-2222-333344445555)\n";
        assert_eq!(
            parse_nvidia_smi_output(sample),
            Some("NVIDIA GeForce RTX 3080".to_string())
        );
    }

    #[test]
    fn test_parse_nvidia_smi_output_rtx3050() {
        let sample = "GPU 0: NVIDIA GeForce RTX 3050 (UUID: GPU-9a833cd5-201e-3ef9-1771-ded8ce2e36e7)\r\n";
        assert_eq!(
            parse_nvidia_smi_output(sample),
            Some("NVIDIA GeForce RTX 3050".to_string())
        );
    }

    #[test]
    fn test_parse_nvidia_smi_output_multi_gpu() {
        let sample = "\
GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-1111-2222-3333)
GPU 1: NVIDIA GeForce RTX 3080 Ti (UUID: GPU-4444-5555-6666)
";
        assert_eq!(
            parse_nvidia_smi_output(sample),
            Some("NVIDIA GeForce RTX 4090".to_string())
        );
    }

    #[test]
    fn test_parse_nvidia_smi_output_without_uuid() {
        let sample = "GPU 0: NVIDIA A100-PCIE-40GB\n";
        assert_eq!(
            parse_nvidia_smi_output(sample),
            Some("NVIDIA A100-PCIE-40GB".to_string())
        );
    }

    #[test]
    fn test_parse_nvidia_smi_output_with_preceding_warnings() {
        let sample = "\
WARNING: info message from driver
GPU 0: Tesla T4 (UUID: GPU-9999-8888)
";
        assert_eq!(
            parse_nvidia_smi_output(sample),
            Some("Tesla T4".to_string())
        );
    }

    #[test]
    fn test_parse_nvidia_smi_output_empty_string() {
        assert_eq!(parse_nvidia_smi_output(""), None);
    }

    #[test]
    fn test_parse_nvidia_smi_output_driver_error() {
        let sample = "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n";
        assert_eq!(parse_nvidia_smi_output(sample), None);
    }

    #[test]
    fn test_parse_nvidia_smi_output_malformed_gpu_line() {
        let sample = "GPU: Invalid Line Without Index\nGPU invalid: No colon index";
        assert_eq!(parse_nvidia_smi_output(sample), None);
    }

    #[test]
    fn test_parse_nvidia_smi_output_empty_device_name() {
        let sample = "GPU 0: (UUID: GPU-1234)\n";
        assert_eq!(parse_nvidia_smi_output(sample), None);
    }

    #[test]
    fn test_detect_gpu_target_returns_valid_variant() {
        let target = detect_gpu_target();
        match &target {
            GpuTarget::NvidiaCuda { device_name } => {
                assert!(!device_name.is_empty());
            }
            GpuTarget::CpuFallback => (),
        }
    }

    #[test]
    fn test_gpu_target_serialization_roundtrip() {
        let cuda = GpuTarget::NvidiaCuda {
            device_name: "NVIDIA GeForce RTX 3080".to_string(),
        };
        let cuda_json = serde_json::to_string(&cuda).expect("serialize cuda");
        let cuda_deser: GpuTarget = serde_json::from_str(&cuda_json).expect("deserialize cuda");
        assert_eq!(cuda, cuda_deser);

        let cpu = GpuTarget::CpuFallback;
        let cpu_json = serde_json::to_string(&cpu).expect("serialize cpu");
        let cpu_deser: GpuTarget = serde_json::from_str(&cpu_json).expect("deserialize cpu");
        assert_eq!(cpu, cpu_deser);
    }

    #[test]
    fn test_parse_nvidia_smi_output_spacing_variations() {
        let sample = "  GPU  0 :   NVIDIA GeForce RTX 4080 (UUID: GPU-abc)  \n";
        assert_eq!(
            parse_nvidia_smi_output(sample),
            Some("NVIDIA GeForce RTX 4080".to_string())
        );
    }

    #[test]
    fn test_detect_gpu_target_on_current_machine() {
        let target = detect_gpu_target();
        if let Some(stdout) = run_nvidia_smi() {
            if let Some(expected_name) = parse_nvidia_smi_output(&stdout) {
                assert_eq!(
                    target,
                    GpuTarget::NvidiaCuda {
                        device_name: expected_name
                    }
                );
                return;
            }
        }
        assert_eq!(target, GpuTarget::CpuFallback);
    }
}

