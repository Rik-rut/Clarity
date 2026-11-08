[CmdletBinding()]
param(
    [string]$Repo = "Rik-rut/Clarity",
    [string]$ReleaseNotes = "Clarity release update",
    [string]$BundleDir = "src-tauri/target/release/bundle/nsis",
    [string]$OutFile = ""
)

$ErrorActionPreference = "Stop"

# 1. Version check across tauri.conf.json, Cargo.toml, package.json
$tauriConfPath = "src-tauri/tauri.conf.json"
$cargoTomlPath = "src-tauri/Cargo.toml"
$packageJsonPath = "package.json"

if (-not (Test-Path $tauriConfPath)) { throw "Cannot find $tauriConfPath" }
if (-not (Test-Path $cargoTomlPath)) { throw "Cannot find $cargoTomlPath" }
if (-not (Test-Path $packageJsonPath)) { throw "Cannot find $packageJsonPath" }

$tauriConf = Get-Content $tauriConfPath -Raw | ConvertFrom-Json
$tauriVer = $tauriConf.version

$packageJson = Get-Content $packageJsonPath -Raw | ConvertFrom-Json
$packageVer = $packageJson.version

$cargoContent = Get-Content $cargoTomlPath -Raw
if ($cargoContent -match 'version\s*=\s*"([^"]+)"') {
    $cargoVer = $matches[1]
} else {
    throw "Could not parse version from $cargoTomlPath"
}

if ($tauriVer -ne $packageVer -or $tauriVer -ne $cargoVer) {
    throw "Version mismatch detected: tauri.conf.json ($tauriVer), package.json ($packageVer), Cargo.toml ($cargoVer). All three must match."
}

Write-Host "Verified consistent version: v$tauriVer across all configuration files."

# 2. Check for release notes file if passed as path
if (Test-Path $ReleaseNotes -PathType Leaf) {
    $ReleaseNotes = Get-Content $ReleaseNotes -Raw
}

# 3. Locate the installer executable in BundleDir
if (-not (Test-Path $BundleDir)) {
    throw "Bundle directory does not exist: $BundleDir"
}

$installer = Get-ChildItem -Path $BundleDir -Filter "*$tauriVer*setup*.exe" | Select-Object -First 1
if (-not $installer) {
    $installer = Get-ChildItem -Path $BundleDir -Filter "*$tauriVer*.exe" | Select-Object -First 1
}
if (-not $installer) {
    $installer = Get-ChildItem -Path $BundleDir -Filter "*setup*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
if (-not $installer) {
    $installer = Get-ChildItem -Path $BundleDir -Filter "*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
if (-not $installer) {
    throw "No installer executable found in $BundleDir. Please build the installer first (e.g. npm run desktop:build)."
}

$sigPath = "$($installer.FullName).sig"
if (-not (Test-Path $sigPath)) {
    throw "Missing signature file: $sigPath. The build must be signed with the updater private key (TAURI_SIGNING_PRIVATE_KEY or TAURI_SIGNING_PRIVATE_KEY_PATH)."
}

$signature = (Get-Content $sigPath -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($signature)) {
    throw "Signature file $sigPath is empty."
}

$pubDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$assetUrl = "https://github.com/$Repo/releases/download/v$tauriVer/$($installer.Name)"

$manifest = [ordered]@{
    "version" = $tauriVer
    "notes" = $ReleaseNotes
    "pub_date" = $pubDate
    "platforms" = [ordered]@{
        "windows-x86_64" = [ordered]@{
            "signature" = $signature
            "url" = $assetUrl
        }
    }
}

if ([string]::IsNullOrWhiteSpace($OutFile)) {
    $OutFile = Join-Path $BundleDir "latest.json"
}

$outParent = Split-Path -Parent $OutFile
if (-not [string]::IsNullOrWhiteSpace($outParent) -and -not (Test-Path $outParent)) {
    New-Item -ItemType Directory -Path $outParent -Force | Out-Null
}

$jsonText = $manifest | ConvertTo-Json -Depth 5
# UTF-8 without BOM: the updater parses latest.json with serde_json, which rejects a leading BOM.
[System.IO.File]::WriteAllText($OutFile, $jsonText, (New-Object System.Text.UTF8Encoding($false)))

Write-Host "Successfully generated updater manifest: $OutFile"
Write-Host "Asset download URL: $assetUrl"
