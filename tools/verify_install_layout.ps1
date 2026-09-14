<#
.SYNOPSIS
    Asserts the single-directory contract after a sandbox install or uninstall.
.EXAMPLE
    tools\verify_install_layout.ps1 -InstallDir D:\clarity-sbx -ExpectTensorrt
    tools\verify_install_layout.ps1 -InstallDir D:\clarity-sbx -Uninstalled
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$InstallDir,
    [switch]$Uninstalled,
    [switch]$ExpectCleanProfile,
    [switch]$ExpectTensorrt
)

$ErrorActionPreference = 'Stop'
$script:failures = @()

function Test-Condition([bool]$Ok, [string]$What) {
    if ($Ok) { Write-Host "  ok    $What" }
    else { Write-Host "  FAIL  $What"; $script:failures += $What }
}

$required = @('python', 'env', 'models', 'logs', 'input', 'output')
$marker   = Join-Path $InstallDir '.setup_complete'
$venvPy   = Join-Path $InstallDir 'env\Scripts\python.exe'
$drive    = (Split-Path -Qualifier $InstallDir)

if ($Uninstalled) {
    foreach ($dir in $required) {
        Test-Condition (-not (Test-Path (Join-Path $InstallDir $dir))) "removed $dir\"
    }
    Test-Condition (-not (Test-Path $marker)) 'removed .setup_complete'
    Test-Condition (-not (Test-Path (Join-Path $InstallDir 'clarity-desktop.exe'))) 'removed the program'
} else {
    foreach ($dir in $required) {
        Test-Condition (Test-Path (Join-Path $InstallDir $dir)) "exists $dir\"
    }
    Test-Condition (Test-Path $marker) 'marker written'
    Test-Condition (Test-Path $venvPy) 'venv interpreter present'
    Test-Condition (@(Get-ChildItem (Join-Path $InstallDir 'models') -File).Count -gt 0) 'models downloaded'

    # The whole point: nothing provisioned anywhere else.
    Test-Condition (-not (Test-Path (Join-Path $drive 'Clarity-data'))) 'no <drive>\Clarity-data sibling'
    if ($ExpectCleanProfile) {
        Test-Condition (-not (Test-Path "$env:LOCALAPPDATA\Clarity")) 'nothing in %LOCALAPPDATA%\Clarity'
    }

    # The venv must be based on the interpreter inside the install folder, not
    # on uv's roaming copy in %APPDATA%\uv.
    $home_line = (Get-Content (Join-Path $InstallDir 'env\pyvenv.cfg') | Where-Object { $_ -match '^home' }) -join ''
    Test-Condition ($home_line -like "*$InstallDir\python*") "pyvenv.cfg home is inside the install folder ($home_line)"

    if ($ExpectTensorrt) {
        Test-Condition (Test-Path (Join-Path $InstallDir 'env\Lib\site-packages\tensorrt')) 'tensorrt installed'
    }
}

if ($script:failures.Count -gt 0) {
    Write-Host "`n$($script:failures.Count) check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "`nAll layout checks passed" -ForegroundColor Green
exit 0
