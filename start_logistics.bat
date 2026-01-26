@echo off
echo Starting Logistics Backend...
start /B python -m uvicorn luminark_enhanced_bridge:app --reload --host 0.0.0.0 --port 8000
timeout 3
echo.
echo ========================================================
echo       LUMINARK REMOTE ACCESS
echo ========================================================
echo To access on Smartphone, connect to same Wi-Fi and go to:
echo.
for /f "tokens=14" %%a in ('ipconfig ^| findstr IPv4') do echo http://%%a:8000
echo.
echo ========================================================
echo.
echo Opening Local Dashboard...
start http://localhost:8000
