@echo off
TITLE LUMINARK ANTIKYTHERA - LOCAL LAUNCH
COLOR 0A

echo.
echo    LUMINARK ANTIKYTHERA
echo    QUANTUM LAUNCHER
echo.
echo ========================================================
echo   INITIATING LOCAL QUANTUM ENVIRONMENT
echo ========================================================
echo.

:: Check for .env
if not exist .env (
    echo [ERROR] .env file missing! 
    echo Please create one with OPENAI_API_KEY.
    pause
    exit
)

:: Install deps if needed
echo [1/2] Checking Dependencies...
pip install -r requirements.txt > nul 2>&1
echo       Done.

:: Run Server
echo [2/2] Launching API Server on Port 8000...
echo       Access Docs at: http://localhost:8000/docs
echo.
echo       (Press Ctrl+C to stop)
echo.

python luminark_api.py

pause
