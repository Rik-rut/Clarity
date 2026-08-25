# Third-Party Notices

Clarity (the application code in this repository) is MIT-licensed — see
`LICENSE`. It builds on the following third-party projects, which keep their
own licenses. Model weights are downloaded at runtime from the Clarity model
hub (`https://huggingface.co/Rik-rut/clarity-models`) and are redistributed
from the original releases listed below.

| Component | Used for | Files | License | Source |
|---|---|---|---|---|
| **Real-CUGAN** | Upscale action | `up2x/up3x/up4x-*.pth`, vendored `upcunet_v3.py` | MIT | https://github.com/bilibili/ailab (Real-CUGAN release) |
| **AMT** | Slow-motion action | `amt-s/amt-l/amt-g.pth`, vendored `src/video_upscaler/amt/` | **CC BY-NC 4.0 (non-commercial)** | https://github.com/MCG-NKU/AMT |
| **MultiPassDedup** | Interpolate action (GMFSS) | `train_log_pg104/*`, vendored `src/video_upscaler/multipass_dedup/` | MIT (see vendored LICENSE) | https://github.com/routineLife1/MultiPassDedup |
| **Practical-RIFE** | Interpolate action (RIFE) | `rife48.pkl` | MIT | https://github.com/hzwer/Practical-RIFE |
| **RAFT** | Flow estimator for GIMM | `raft-things.pth` | BSD 3-Clause | https://github.com/princeton-vl/RAFT |
| **GIMM-VFI** | Interpolate action (GIMM) | `gimmvfi_r/f_arb_lpips.pt`, vendored gimm sources | **S-Lab License 1.0 (non-commercial)** | https://github.com/GSeanCDAT/GIMM-VFI |
| **MatAnyone2** | Matting action (foreground + alpha matte) | `matanyone2.pth`, vendored `src/video_upscaler/matanyone2/vendor/` | **S-Lab License 1.0 (non-commercial)** | https://github.com/pq-yang/MatAnyone2 |
| **realcugan-ncnn-vulkan** | Vulkan fallback backend | downloaded per-OS into `tools/ncnn/` | MIT | https://github.com/nihui/realcugan-ncnn-vulkan |

## Non-commercial restrictions

Three bundled components restrict commercial use:

1. **AMT** (Slow-motion) — CC BY-NC 4.0: sharing and adapting is permitted
   with attribution for non-commercial purposes only.
2. **GIMM-VFI** (Interpolate, GIMM model) — S-Lab License 1.0:
   redistribution and use are permitted for non-commercial purposes;
   commercial use requires contacting the contributors.
3. **MatAnyone2** (Matting) — S-Lab License 1.0: redistribution and use are
   permitted for non-commercial purposes; commercial use requires contacting
   the contributors.

The **Upscale (Real-CUGAN)** and **Interpolate (RIFE/GMFSS)** paths carry no
non-commercial restriction beyond attribution requirements noted above.

If you redistribute Clarity or its model hub contents, keep this notices file
intact alongside your distribution.
