@echo off
echo ==================================================
echo       🌌 LAUNCHING LUMINARK BEAST MODE 🌌
echo ==================================================
echo.
echo [1/2] Checking dependencies...
C:\Users\Forev\AppData\Local\Python\bin\python.exe -m pip install -r requirements.txt
echo.
echo [2/2] Starting Dashboard...
echo.
echo Press Ctrl+C in this window to stop the server.
echo.
"C:\Users\Forev\AppData\Local\Python\bin\python.exe" -m streamlit run luminark_dashboard.py
pause
