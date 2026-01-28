@echo off
REM ============================================
REM LUMINARK v5 - Quick Deploy Script
REM ============================================

echo.
echo ====================================
echo  LUMINARK v5 UPGRADE DEPLOYMENT
echo ====================================
echo.

REM Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git not found. Please install Git first.
    pause
    exit /b 1
)

REM Navigate to project directory
cd /d "%~dp0"

echo [1/5] Backing up current index.html...
if exist index.html (
    copy index.html index.html.backup.%date:~-4%.%date:~-10,2%.%date:~-7,2%_%time:~0,2%.%time:~3,2% >nul
    echo ✓ Backup created
) else (
    echo ⚠ No existing index.html found
)

echo.
echo [2/5] Copying v5 upgraded version...
if exist luminark_v5_upgraded.html (
    copy luminark_v5_upgraded.html index.html >nul
    echo ✓ V5 deployed as index.html
) else (
    echo ERROR: luminark_v5_upgraded.html not found
    pause
    exit /b 1
)

echo.
echo [3/5] Staging changes...
git add index.html >nul 2>&1
echo ✓ Changes staged

echo.
echo [4/5] Committing to Git...
git commit -m "Upgrade: LUMINARK v5.0 - Sidebar, sliders, radar chart, real-time metrics" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Committed to Git
) else (
    echo ⚠ Commit failed (may already be up-to-date)
)

echo.
echo [5/5] Pushing to GitHub...
git push >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Pushed to GitHub
    echo.
    echo ====================================
    echo ✅ DEPLOYMENT SUCCESSFUL!
    echo ====================================
    echo.
    echo Your app will be live on Vercel in ~30 seconds
    echo https://luminark-six.vercel.app/
    echo.
    echo View deployment status:
    echo https://vercel.com/dashboard
    echo.
) else (
    echo ⚠ Push to GitHub failed
    echo Check your internet connection and git credentials
)

echo.
pause
