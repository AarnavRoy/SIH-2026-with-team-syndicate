@echo off
title SignalScope Forensic Engine Launcher
echo ===================================================
echo   SignalScope - AI Media Forensics & Verification
echo   SIH 2026 [Internal Hackathon] Team Syndicate
echo ===================================================
echo.

:: Check for Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found on your system PATH!
    echo Please install Python 3.10+ from python.org or Anaconda.
    pause
    exit /b 1
)

echo [*] Checking required dependencies...
python -c "import fastapi, uvicorn, torch, torchvision, PIL, numpy" >nul 2>nul
if %errorlevel% neq 0 (
    echo [*] Installing missing dependencies from requirements.txt...
    python -m pip install -r requirements.txt
) else (
    echo [OK] All core dependencies verified.
)

echo.
echo [*] Launching SignalScope Forensic Engine on http://127.0.0.1:8000...
echo [*] Opening default web browser...

:: Open browser after 2 second delay
start "" timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000

:: Start uvicorn server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
