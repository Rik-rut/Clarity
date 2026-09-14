<#
.SYNOPSIS
    Prepares desktop runtime assets (uv.exe, ffmpeg.exe, ffprobe.exe, and app icons)
    for Clarity desktop packaging with Tauri v2.

.DESCRIPTION
    Ensures standalone binaries and desktop icons are properly placed in:
    - src-tauri/resources/uv.exe
    - src-tauri/resources/ffmpeg.exe
    - src-tauri/resources/ffprobe.exe
    - src-tauri/icons/ (icon.ico, 32x32.png, 128x128.png, 128x128@2x.png, icon.png)

    The script is idempotent. It skips downloads if working binaries and icons are
    already present, unless -Force is specified. It checks local caches before
    downloading over the network unless -ForceDownload is specified.

.PARAMETER Force
    Overwrites existing resources and regenerates icons even if already present.

.PARAMETER ForceDownload
    Bypasses local binary detection and forces fresh downloads from upstream releases.

.EXAMPLE
    .\tools\prepare_desktop.ps1
    .\tools\prepare_desktop.ps1 -Force
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$ForceDownload
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$ResourcesDir = Join-Path $RepoRoot "src-tauri\resources"
$IconsDir = Join-Path $RepoRoot "src-tauri\icons"
$BrandLogo = Join-Path $RepoRoot "src\video_upscaler\web\static\clarity.jpg"

function Write-Info ($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

function Write-Success ($msg) {
    Write-Host "[SUCCESS] $msg" -ForegroundColor Green
}

function Write-Warn ($msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

function Write-Err ($msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

function Test-BinaryExecutable {
    param(
        [Parameter(Mandatory=$true)][string]$FilePath,
        [Parameter(Mandatory=$true)][string]$Argument,
        [int]$MinSizeBytes = 1048576
    )
    if (-not (Test-Path $FilePath)) { return $false }
    $fileItem = Get-Item $FilePath
    if ($fileItem.Length -lt $MinSizeBytes) { return $false }
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $FilePath
        $psi.Arguments = $Argument
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $proc = [System.Diagnostics.Process]::Start($psi)
        $finished = $proc.WaitForExit(6000)
        if (-not $finished) {
            $proc.Kill()
            return $false
        }
        return ($proc.ExitCode -eq 0)
    } catch {
        return $false
    }
}

function Download-ArchiveFile {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$DestinationPath
    )
    $parent = Split-Path -Parent $DestinationPath
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        Write-Info "Downloading via curl: $Url"
        & curl.exe -fSL --retry 3 --retry-delay 2 $Url -o $DestinationPath
        if ($LASTEXITCODE -eq 0 -and (Test-Path $DestinationPath) -and ((Get-Item $DestinationPath).Length -gt 0)) {
            return $true
        }
        Write-Warn "curl download failed or produced empty file. Retrying with PowerShell..."
    }

    Write-Info "Downloading via Invoke-WebRequest: $Url"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13
        Invoke-WebRequest -Uri $Url -OutFile $DestinationPath -UseBasicParsing -TimeoutSec 600
        return ((Test-Path $DestinationPath) -and ((Get-Item $DestinationPath).Length -gt 0))
    } catch {
        Write-Err "Download error: $_"
        return $false
    }
}

function Extract-ArchiveFile {
    param(
        [Parameter(Mandatory=$true)][string]$ArchivePath,
        [Parameter(Mandatory=$true)][string]$ExtractDir
    )
    if (-not (Test-Path $ExtractDir)) {
        New-Item -ItemType Directory -Path $ExtractDir -Force | Out-Null
    }

    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if ($tar) {
        & tar.exe -xf $ArchivePath -C $ExtractDir
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
    }

    Expand-Archive -Path $ArchivePath -DestinationPath $ExtractDir -Force
    return $true
}

# Ensure target directories exist
if (-not (Test-Path $ResourcesDir)) {
    New-Item -ItemType Directory -Path $ResourcesDir -Force | Out-Null
}
if (-not (Test-Path $IconsDir)) {
    New-Item -ItemType Directory -Path $IconsDir -Force | Out-Null
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Clarity Desktop Preparation & Asset Bundler    " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# -------------------------------------------------------------
# 1. uv.exe setup
# -------------------------------------------------------------
Write-Host "`n[1/3] Checking uv.exe..." -ForegroundColor White
$uvTarget = Join-Path $ResourcesDir "uv.exe"
$uvValid = Test-BinaryExecutable -FilePath $uvTarget -Argument "--version" -MinSizeBytes 1048576

if ($uvValid -and -not $Force) {
    $uvVer = (& "$uvTarget" --version).Trim()
    Write-Success "uv.exe already verified at $uvTarget ($uvVer)"
} else {
    $uvAcquired = $false

    if (-not $ForceDownload) {
        $localUvCandidates = @(
            "$env:USERPROFILE\.local\bin\uv.exe",
            ((Get-Command uv.exe -ErrorAction SilentlyContinue).Source)
        ) | Where-Object { $_ -and (Test-Path $_) }

        foreach ($cand in $localUvCandidates) {
            if (Test-BinaryExecutable -FilePath $cand -Argument "--version" -MinSizeBytes 1048576) {
                Write-Info "Found verified local uv.exe at $cand. Copying to resources..."
                Copy-Item -Path $cand -Destination $uvTarget -Force
                $uvAcquired = $true
                break
            }
        }
    }

    if (-not $uvAcquired) {
        Write-Info "Downloading Astral uv.exe release for x86_64 Windows MSVC..."
        $tempZip = Join-Path ([System.IO.Path]::GetTempPath()) ("uv_desktop_" + [System.Guid]::NewGuid().ToString("N") + ".zip")
        $uvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
        
        $dlOk = Download-ArchiveFile -Url $uvUrl -DestinationPath $tempZip
        if (-not $dlOk) {
            throw "Failed to download uv release from $uvUrl"
        }

        $tempExtract = Join-Path ([System.IO.Path]::GetTempPath()) ("uv_extract_" + [System.Guid]::NewGuid().ToString("N"))
        try {
            Extract-ArchiveFile -ArchivePath $tempZip -ExtractDir $tempExtract
            $extractedUv = Get-ChildItem -Path $tempExtract -Filter "uv.exe" -Recurse | Select-Object -First 1
            if (-not $extractedUv) {
                throw "uv.exe not found in downloaded archive."
            }
            Copy-Item -Path $extractedUv.FullName -Destination $uvTarget -Force
        } finally {
            if (Test-Path $tempZip) { Remove-Item $tempZip -Force -ErrorAction SilentlyContinue }
            if (Test-Path $tempExtract) { Remove-Item $tempExtract -Recurse -Force -ErrorAction SilentlyContinue }
        }
    }

    if (-not (Test-BinaryExecutable -FilePath $uvTarget -Argument "--version" -MinSizeBytes 1048576)) {
        throw "Verification failed for $uvTarget after installation."
    }
    $uvVer = (& "$uvTarget" --version).Trim()
    Write-Success "uv.exe successfully configured: $uvVer"
}

# -------------------------------------------------------------
# 2. ffmpeg.exe and ffprobe.exe setup
# -------------------------------------------------------------
Write-Host "`n[2/3] Checking ffmpeg.exe & ffprobe.exe..." -ForegroundColor White
$ffmpegTarget = Join-Path $ResourcesDir "ffmpeg.exe"
$ffprobeTarget = Join-Path $ResourcesDir "ffprobe.exe"

$ffmpegValid = (Test-BinaryExecutable -FilePath $ffmpegTarget -Argument "-version" -MinSizeBytes 10485760) -and
               (Test-BinaryExecutable -FilePath $ffprobeTarget -Argument "-version" -MinSizeBytes 10485760)

if ($ffmpegValid -and -not $Force) {
    Write-Success "ffmpeg.exe and ffprobe.exe already verified at $ResourcesDir"
} else {
    $ffmpegAcquired = $false

    if (-not $ForceDownload) {
        $localFfmpegCandidates = @(
            "$env:USERPROFILE\scoop\apps\ffmpeg\current\bin\ffmpeg.exe",
            ((Get-Command ffmpeg.exe -ErrorAction SilentlyContinue).Source)
        ) | Where-Object { $_ -and (Test-Path $_) }

        foreach ($cand in $localFfmpegCandidates) {
            $candDir = Split-Path -Parent $cand
            $candProbe = Join-Path $candDir "ffprobe.exe"
            if ((Test-BinaryExecutable -FilePath $cand -Argument "-version" -MinSizeBytes 10485760) -and
                (Test-BinaryExecutable -FilePath $candProbe -Argument "-version" -MinSizeBytes 10485760)) {
                Write-Info "Found verified local FFmpeg at $candDir. Copying to resources..."
                Copy-Item -Path $cand -Destination $ffmpegTarget -Force
                Copy-Item -Path $candProbe -Destination $ffprobeTarget -Force
                $ffmpegAcquired = $true
                break
            }
        }
    }

    if (-not $ffmpegAcquired) {
        Write-Info "Downloading FFmpeg standalone release essentials..."
        $tempZip = Join-Path ([System.IO.Path]::GetTempPath()) ("ffmpeg_desktop_" + [System.Guid]::NewGuid().ToString("N") + ".zip")
        $primaryUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        $fallbackUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

        $dlOk = Download-ArchiveFile -Url $primaryUrl -DestinationPath $tempZip
        if (-not $dlOk) {
            Write-Warn "Primary FFmpeg download failed. Trying fallback URL: $fallbackUrl"
            $dlOk = Download-ArchiveFile -Url $fallbackUrl -DestinationPath $tempZip
        }
        if (-not $dlOk) {
            throw "Failed to download FFmpeg from both primary and fallback mirrors."
        }

        $tempExtract = Join-Path ([System.IO.Path]::GetTempPath()) ("ffmpeg_extract_" + [System.Guid]::NewGuid().ToString("N"))
        try {
            Extract-ArchiveFile -ArchivePath $tempZip -ExtractDir $tempExtract
            $foundFfmpeg = Get-ChildItem -Path $tempExtract -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
            $foundFfprobe = Get-ChildItem -Path $tempExtract -Filter "ffprobe.exe" -Recurse | Select-Object -First 1

            if (-not $foundFfmpeg -or -not $foundFfprobe) {
                throw "ffmpeg.exe or ffprobe.exe not found in downloaded archive."
            }

            Copy-Item -Path $foundFfmpeg.FullName -Destination $ffmpegTarget -Force
            Copy-Item -Path $foundFfprobe.FullName -Destination $ffprobeTarget -Force
        } finally {
            if (Test-Path $tempZip) { Remove-Item $tempZip -Force -ErrorAction SilentlyContinue }
            if (Test-Path $tempExtract) { Remove-Item $tempExtract -Recurse -Force -ErrorAction SilentlyContinue }
        }
    }

    if (-not (Test-BinaryExecutable -FilePath $ffmpegTarget -Argument "-version" -MinSizeBytes 10485760)) {
        throw "Verification failed for $ffmpegTarget after installation."
    }
    if (-not (Test-BinaryExecutable -FilePath $ffprobeTarget -Argument "-version" -MinSizeBytes 10485760)) {
        throw "Verification failed for $ffprobeTarget after installation."
    }
    Write-Success "ffmpeg.exe and ffprobe.exe successfully configured in $ResourcesDir"
}

# -------------------------------------------------------------
# 3. Application Icons Setup
# -------------------------------------------------------------
Write-Host "`n[3/3] Checking application icons in src-tauri/icons/..." -ForegroundColor White
$icoTarget = Join-Path $IconsDir "icon.ico"
$png32Target = Join-Path $IconsDir "32x32.png"
$png128Target = Join-Path $IconsDir "128x128.png"
$png256Target = Join-Path $IconsDir "128x128@2x.png"
$png512Target = Join-Path $IconsDir "icon.png"

$iconsExist = (Test-Path $icoTarget) -and ((Get-Item $icoTarget).Length -gt 1024) -and
              (Test-Path $png32Target) -and ((Get-Item $png32Target).Length -gt 100) -and
              (Test-Path $png128Target) -and ((Get-Item $png128Target).Length -gt 100)

if ($iconsExist -and -not $Force) {
    Write-Success "Application icons already exist and are valid. (Use -Force to regenerate)"
} else {
    Write-Info "Generating application icons for desktop packaging..."
    $iconsGenerated = $false

    # Approach 1: Try Python Pillow (using uv or system python)
    $hasUv = Test-BinaryExecutable -FilePath $uvTarget -Argument "--version" -MinSizeBytes 1048576
    
    $pyScript = @"
import os, sys
from PIL import Image

src_img = sys.argv[1]
icons_dir = sys.argv[2]
os.makedirs(icons_dir, exist_ok=True)

if os.path.exists(src_img):
    img = Image.open(src_img).convert("RGBA")
else:
    # Procedural fallback
    img = Image.new("RGBA", (512, 512), (18, 24, 38, 255))

sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
img.save(os.path.join(icons_dir, "icon.ico"), format="ICO", sizes=sizes)
img.resize((32, 32), Image.Resampling.LANCZOS).save(os.path.join(icons_dir, "32x32.png"), format="PNG")
img.resize((128, 128), Image.Resampling.LANCZOS).save(os.path.join(icons_dir, "128x128.png"), format="PNG")
img.resize((256, 256), Image.Resampling.LANCZOS).save(os.path.join(icons_dir, "128x128@2x.png"), format="PNG")
img.resize((512, 512), Image.Resampling.LANCZOS).save(os.path.join(icons_dir, "icon.png"), format="PNG")
print("ICONS_OK")
"@

    $tempPy = Join-Path ([System.IO.Path]::GetTempPath()) ("make_icons_" + [System.Guid]::NewGuid().ToString("N") + ".py")
    Set-Content -Path $tempPy -Value $pyScript -Encoding utf8
    try {
        if ($hasUv) {
            $pyOutput = & "$uvTarget" run python $tempPy $BrandLogo $IconsDir 2>&1
            if ($pyOutput -match "ICONS_OK") {
                $iconsGenerated = $true
            }
        }
        if (-not $iconsGenerated) {
            $pySys = Get-Command python.exe -ErrorAction SilentlyContinue
            if ($pySys) {
                $pyOutput = & python $tempPy $BrandLogo $IconsDir 2>&1
                if ($pyOutput -match "ICONS_OK") {
                    $iconsGenerated = $true
                }
            }
        }
    } finally {
        if (Test-Path $tempPy) { Remove-Item $tempPy -Force -ErrorAction SilentlyContinue }
    }

    # Approach 2: Fallback to System.Drawing in PowerShell if Python is unavailable
    if (-not $iconsGenerated) {
        Write-Info "Falling back to System.Drawing for icon generation..."
        Add-Type -AssemblyName System.Drawing
        
        $srcBmp = $null
        if (Test-Path $BrandLogo) {
            $srcBmp = [System.Drawing.Bitmap]::FromFile($BrandLogo)
        } else {
            $srcBmp = New-Object System.Drawing.Bitmap 512, 512
            $g = [System.Drawing.Graphics]::FromImage($srcBmp)
            $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(255, 18, 24, 38))
            $g.FillRectangle($brush, 0, 0, 512, 512)
            $brush.Dispose(); $g.Dispose()
        }

        try {
            function Resize-Bmp ($orig, $w, $h) {
                $dest = New-Object System.Drawing.Bitmap $w, $h
                $g = [System.Drawing.Graphics]::FromImage($dest)
                $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
                $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
                $g.DrawImage($orig, 0, 0, $w, $h)
                $g.Dispose()
                return $dest
            }

            $b32 = Resize-Bmp $srcBmp 32 32
            $b32.Save($png32Target, [System.Drawing.Imaging.ImageFormat]::Png)
            $b32.Dispose()

            $b128 = Resize-Bmp $srcBmp 128 128
            $b128.Save($png128Target, [System.Drawing.Imaging.ImageFormat]::Png)
            $b128.Dispose()

            $b256 = Resize-Bmp $srcBmp 256 256
            $b256.Save($png256Target, [System.Drawing.Imaging.ImageFormat]::Png)

            $b512 = Resize-Bmp $srcBmp 512 512
            $b512.Save($png512Target, [System.Drawing.Imaging.ImageFormat]::Png)
            $b512.Dispose()

            $hIcon = $b256.GetHicon()
            $icon = [System.Drawing.Icon]::FromHandle($hIcon)
            $fs = New-Object System.IO.FileStream $icoTarget, ([System.IO.FileMode]::Create)
            $icon.Save($fs)
            $fs.Close()
            $fs.Dispose()
            $icon.Dispose()
            $b256.Dispose()
            $iconsGenerated = $true
        } finally {
            if ($srcBmp) { $srcBmp.Dispose() }
        }
    }

    if (-not ((Test-Path $icoTarget) -and (Test-Path $png32Target) -and (Test-Path $png128Target))) {
        throw "Failed to generate valid icons in $IconsDir"
    }
    Write-Success "Icons generated successfully in $IconsDir"
}

# -------------------------------------------------------------
# Final Verification & Output
# -------------------------------------------------------------
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "   Preparation Complete & Verified Successfully   " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  uv binary:     $uvTarget" -ForegroundColor Gray
Write-Host "  uv version:    $((& "$uvTarget" --version).Trim())" -ForegroundColor Green
Write-Host "  ffmpeg binary: $ffmpegTarget" -ForegroundColor Gray
Write-Host "  ffmpeg ver:    $(((& "$ffmpegTarget" -version)[0]).Trim())" -ForegroundColor Green
Write-Host "  ffprobe bin:   $ffprobeTarget" -ForegroundColor Gray
Write-Host "  ffprobe ver:   $(((& "$ffprobeTarget" -version)[0]).Trim())" -ForegroundColor Green
Write-Host "  icons dir:     $IconsDir" -ForegroundColor Gray
Get-ChildItem -Path $IconsDir | ForEach-Object {
    Write-Host ("    - " + $_.Name + " (" + $_.Length + " bytes)") -ForegroundColor Gray
}
Write-Host ""
