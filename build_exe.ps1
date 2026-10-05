# Builds XrealHeadFusion.exe (PyInstaller, one-folder), installs it for the current user
# under %LOCALAPPDATA%\Programs\XrealHeadFusion and puts a shortcut on the Desktop.
# The build venv and intermediates live in %LOCALAPPDATA%\XrealHeadFusion so nothing heavy
# lands in this (possibly cloud-synced) project folder. Requires uv.
#
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1            # build + install + shortcut
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1 -NoInstall # build only
param([switch]$NoInstall)
$ErrorActionPreference = "Stop"

$root    = $PSScriptRoot
$base    = Join-Path $env:LOCALAPPDATA "XrealHeadFusion"
$venv    = Join-Path $base "venv"
$work    = Join-Path $base "build"
$install = Join-Path $env:LOCALAPPDATA "Programs\XrealHeadFusion"
$py      = Join-Path $venv "Scripts\python.exe"
$model   = Join-Path $root "models\face_landmarker.task"
$modelUrl = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if (-not (Test-Path $py)) { uv venv --python 3.12 $venv; if ($LASTEXITCODE) { throw "uv venv failed" } }
uv pip install --python $py -r (Join-Path $root "requirements.txt") pyinstaller
if ($LASTEXITCODE) { throw "package install failed" }

if (-not (Test-Path $model)) {
    New-Item -ItemType Directory -Force (Split-Path $model) | Out-Null
    Write-Host "Downloading MediaPipe face landmarker model..."
    Invoke-WebRequest $modelUrl -OutFile $model
}

& $py -m PyInstaller --noconfirm --clean --onedir --console --name XrealHeadFusion `
    --icon (Join-Path $root "assets\app.ico") `
    --collect-all mediapipe `
    --add-data "$model;models" `
    --distpath (Join-Path $work "dist") --workpath (Join-Path $work "work") --specpath $work `
    (Join-Path $root "tracker.py")
if ($LASTEXITCODE) { throw "PyInstaller failed" }

$built = Join-Path $work "dist\XrealHeadFusion"
Write-Host "Built: $built"
if ($NoInstall) { return }

if (Test-Path $install) { Remove-Item -Recurse -Force $install }
Copy-Item -Recurse $built $install
$exe = Join-Path $install "XrealHeadFusion.exe"

$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $desktop "XREAL Head Fusion.lnk"))
$lnk.TargetPath = $exe
$lnk.WorkingDirectory = $install
$lnk.IconLocation = "$exe,0"
$lnk.Description = "Webcam face tracking + XREAL One Pro IMU head-orientation fusion"
$lnk.Save()
Write-Host "Installed: $exe"
Write-Host "Desktop shortcut: XREAL Head Fusion.lnk"
