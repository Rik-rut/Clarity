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
| **MatAnyone2** | Matting action (foreground + alpha matte) | `matanyone2.pth`, vendored `src/video_upscaler/matanyone2/vendor/` | **S-Lab License 1.0 (non-commercial)** | https://github.com/pq-yang/MatAnyone2 |
| **realcugan-ncnn-vulkan** | Vulkan fallback backend | downloaded per-OS into `tools/ncnn/` | MIT | https://github.com/nihui/realcugan-ncnn-vulkan |
| **aria2** | Multi-connection, resumable model-weight downloads | bundled `src-tauri/resources/aria2c.exe` (v1.37.0), run as a separate process | **GPL-2.0** | https://github.com/aria2/aria2 |

## aria2 (GPL-2.0)

Clarity ships the official Windows build of aria2 v1.37.0 in the installer and
runs it as a separate process; it is not linked into Clarity's code. The exact
archive staged is `aria2-1.37.0-win-64bit-build1.zip`, sha256
`67d015301eef0b612191212d564c5bb0a14b5b9c4796b76454276a4d28d9b288`, from
https://github.com/aria2/aria2/releases/tag/release-1.37.0 . That release
archive is the corresponding source offer; the binary is statically linked
against gmp, expat, sqlite3, zlib, c-ares and libssh2 (see `README.mingw` and
`LICENSE.OpenSSL` inside the archive). Requests for the source can also be made
to the upstream project at the URL above.

## Non-commercial restrictions

Two bundled components restrict commercial use:

1. **AMT** (Slow-motion) — CC BY-NC 4.0: sharing and adapting is permitted
   with attribution for non-commercial purposes only.
2. **MatAnyone2** (Matting) — S-Lab License 1.0: redistribution and use are
   permitted for non-commercial purposes; commercial use requires contacting
   the contributors.

The **Upscale (Real-CUGAN)** and **Interpolate (RIFE/GMFSS)** paths carry no
non-commercial restriction beyond attribution requirements noted above.

If you redistribute Clarity or its model hub contents, keep this notices file
intact alongside your distribution.
