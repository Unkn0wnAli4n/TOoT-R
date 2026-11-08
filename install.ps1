# TOoT:R installer for Windows (double-click install.cmd, or run:
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Nup C:\path\game.nup] [-Dir C:\Games\TOoT-R])
#
# Sets up MSYS2 (CLANG64) with the build tools and libraries, asks for your own dump of the game and
# for the folder to install into (default %LOCALAPPDATA%\tootr), builds the game there, and creates
# Start menu and desktop shortcuts. The shortcut checks GitHub for a newer version on every start and
# rebuilds automatically (set OOT_RECOMP_NO_UPDATE=1 to skip).
param(
    [string]$Nup = "",
    [string]$Dir = "",
    [string]$Msys = "C:\msys64",
    [string]$Repo = "Unkn0wnAli4n/TOoT-R"  # (keep in sync with installer/repo.txt)
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
function Say($msg) { Write-Host "[TOoT:R] $msg" -ForegroundColor Cyan }

if (-not [Environment]::Is64BitOperatingSystem) { Say "a 64-bit Windows is required"; exit 1 }

# ---- MSYS2 ---------------------------------------------------------------------------------------
$bash = Join-Path $Msys "usr\bin\bash.exe"
if (-not (Test-Path $bash)) {
    Say "installing MSYS2 (build environment) into $Msys"
    $installed = $false
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id MSYS2.MSYS2 -e --silent --accept-package-agreements --accept-source-agreements --location $Msys
        $installed = Test-Path $bash
    }
    if (-not $installed) {
        $setup = Join-Path $env:TEMP "msys2-installer.exe"
        Invoke-WebRequest "https://repo.msys2.org/distrib/msys2-x86_64-latest.exe" -OutFile $setup
        & $setup in --confirm-command --accept-messages --root ($Msys -replace '\\', '/') | Out-Null
        Remove-Item $setup -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path $bash)) { Say "MSYS2 could not be installed; install it from https://www.msys2.org and run this again"; exit 1 }
}

function Msys($cmd) {
    $env:MSYSTEM = "CLANG64"
    $env:CHERE_INVOKING = "1"
    & $bash -lc $cmd
    if ($LASTEXITCODE -ne 0) { throw "MSYS2 command failed: $cmd" }
}
Say "updating MSYS2 and installing the build tools (this takes a while the first time)"
& $bash -lc "pacman -Syu --noconfirm" | Out-Host   # (may update the core and stop; run twice)
& $bash -lc "pacman -Syu --noconfirm" | Out-Host
$packages = "clang lld cmake ninja python pkgconf sdl3 opus ffmpeg vulkan-loader vulkan-headers fmt zstd lz4" -split " " |
    ForEach-Object { "mingw-w64-clang-x86_64-$_" }
Msys ("pacman -S --needed --noconfirm " + ($packages -join " "))

# ---- source ---------------------------------------------------------------------------------------
$source = $PSScriptRoot
$temp = $null
if (-not (Test-Path (Join-Path $source "tools\build_game.py"))) {
    Say "downloading $Repo"
    $temp = Join-Path $env:TEMP ("tootr-" + [guid]::NewGuid())
    New-Item -ItemType Directory $temp | Out-Null
    $zip = Join-Path $temp "src.zip"
    Invoke-WebRequest "https://codeload.github.com/$Repo/zip/refs/heads/main" -OutFile $zip
    Expand-Archive $zip -DestinationPath $temp
    $source = (Get-ChildItem $temp -Directory | Select-Object -First 1).FullName
}

# ---- the game -------------------------------------------------------------------------------------
if (-not $Nup) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = "Select your dump of The Legend of Zelda: Ocarina of Time (Switch 2, v131072)"
    $dialog.Filter = "Switch 2 package (*.nup)|*.nup|All files (*.*)|*.*"
    if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { Say "no game file selected"; exit 1 }
    $Nup = $dialog.FileName
}
if (-not (Test-Path $Nup)) { Say "no such file: $Nup"; exit 1 }
if (-not $Dir) {
    $default = Join-Path $env:LOCALAPPDATA "tootr"
    Add-Type -AssemblyName System.Windows.Forms
    $folder = New-Object System.Windows.Forms.FolderBrowserDialog
    $folder.Description = "Choose where to install TOoT:R (a TOoT-R folder is made there). Cancel: $default"
    $folder.ShowNewFolderButton = $true
    if ($folder.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK -and $folder.SelectedPath) { $Dir = Join-Path $folder.SelectedPath "TOoT-R" }
    else { $Dir = $default }
}
$Dir = [System.IO.Path]::GetFullPath($Dir)
Say "installing into $Dir"

# ---- build ----------------------------------------------------------------------------------------
$clang64 = Join-Path $Msys "clang64\bin"
$env:PATH = "$clang64;" + (Join-Path $Msys "usr\bin") + ";$env:PATH"
$env:MSYSTEM = "CLANG64"
$python = Join-Path $clang64 "python.exe"
& $python (Join-Path $source "installer\launcher.py") install --source $source --nup $Nup --dir $Dir
if ($LASTEXITCODE -ne 0) { Say "the build failed; see the messages above"; exit 1 }
if ($temp) { Remove-Item -Recurse -Force $temp -ErrorAction SilentlyContinue }

# ---- shortcuts ------------------------------------------------------------------------------------
$data = $Dir
$runner = Join-Path $data "tootr.cmd"
@"
@echo off
rem tootr: checks for updates, then starts the game.
set "PATH=$clang64;$Msys\usr\bin;%PATH%"
set MSYSTEM=CLANG64
set "OOT_RECOMP_HOME=$data"
cd /d "$data"
"$python" "$data\source\installer\launcher.py" run %*
if errorlevel 1 pause
"@ | Set-Content -Encoding ASCII $runner

$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath("Programs"), [Environment]::GetFolderPath("Desktop"))) {
    $link = $shell.CreateShortcut((Join-Path $dir "TOoT-R.lnk"))
    $link.TargetPath = $runner
    $link.WorkingDirectory = $data
    $link.Description = "Native PC port built from your own copy of the game"
    $link.Save()
}
Say "installed. Start the game from the Start menu or the desktop shortcut."
