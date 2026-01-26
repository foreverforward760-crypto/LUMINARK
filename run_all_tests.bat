@echo off
echo ========================================
echo LUMINARK MASTER TEST SUITE
echo ========================================
echo.
echo Running comprehensive automated tests...
echo.

echo [1/2] Testing Safety Protocols (Ma'at + Yunus)...
echo ----------------------------------------
python test_safety_protocols.py
echo.

echo [2/2] Testing Logistics Dashboard API...
echo ----------------------------------------
python test_dashboard.py
echo.

echo ========================================
echo ALL TESTS COMPLETE!
echo ========================================
echo.
echo Check these files for detailed reports:
echo   - safety_test_report.json
echo   - dashboard_test_report.json
echo   - SAFETY_TESTING_REPORT.md
echo.
pause
