@echo off
REM run.bat — one-click launcher for the Human Tracking System
REM Double-click this file from Windows Explorer, or run it from a terminal.
REM First run sets up everything automatically (venv + dependencies).
REM Every run after that just launches straight away.

setlocal

cd /d "%~dp0"

REM ── Check Python is installed and reachable ─────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [ERROR] Python was not found on this system.
    echo   Install Python 3.9-3.11 from https://www.python.org/downloads/
    echo   and make sure "Add Python to PATH" is checked during setup.
    echo.
    pause
    exit /b 1
)

REM ── First-run setup: create venv and install dependencies ───────────────
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo   First run detected — setting up the environment.
    echo   This only happens once and may take a few minutes.
    echo.

    python -m venv venv
    if errorlevel 1 (
        echo   [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )

    call venv\Scripts\activate.bat

    echo   Installing dependencies...
    pip install --upgrade pip >nul
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo   [ERROR] Dependency installation failed. Check your internet
        echo   connection and try running this file again.
        echo.
        pause
        exit /b 1
    )

    echo.
    echo   Setup complete.
    echo.
) else (
    call venv\Scripts\activate.bat
)

REM ── Check the required MediaPipe model files are present ────────────────
if not exist "pose_landmarker_full.task" (
    echo.
    echo   [ERROR] pose_landmarker_full.task is missing from this folder.
    echo   See README.md for the download link.
    echo.
    pause
    exit /b 1
)
if not exist "hand_landmarker.task" (
    echo.
    echo   [ERROR] hand_landmarker.task is missing from this folder.
    echo   See README.md for the download link.
    echo.
    pause
    exit /b 1
)

echo.
echo   Human Tracking System
echo   ----------------------
echo   [i] Boot AI     [Space] Warmup     [Tab] Analytics
echo   [h] Help         [ESC/q] Quit
echo.

python main.py %*

if errorlevel 1 (
    echo.
    echo   Program exited with an error. See messages above.
    pause
)

endlocal