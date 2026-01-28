@echo off
title LUMINARK ORACLE ENGINE
echo ==================================================
echo       🌌 LAUNCHING LUMINARK ORACLE ENGINE 🌌
echo ==================================================
echo.
echo [1/2] Starting LUMINARK AI Brain (FastAPI)...
start "LUMINARK API" cmd /k "python api_bridge.py"
echo.
echo [2/2] Opening Antikythera Dashboard...
timeout /t 3 /nobreak > nul
start index.html
echo.
echo ==================================================
echo   LUMINARK IS NOW ACTIVE IN YOUR BROWSER
echo   Keep the API window open for Quantum Analysis.
echo ==================================================
pause
