@echo off
rem TOoT:R installer for Windows: double-click. Runs install.ps1 beside it, or downloads it from GitHub
rem (keep the project name in sync with installer/repo.txt).
setlocal
set "SCRIPT=%~dp0install.ps1"
if not exist "%SCRIPT%" (
    set "SCRIPT=%TEMP%\tootr-install.ps1"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest 'https://raw.githubusercontent.com/Unkn0wnAli4n/TOoT-R/main/install.ps1' -OutFile $env:TEMP\tootr-install.ps1"
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
pause
