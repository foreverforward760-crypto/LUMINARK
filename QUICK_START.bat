@echo off
REM ============================================
REM LUMINARK v5 - QUICK START GUIDE (Windows)
REM ============================================

setlocal enabledelayedexpansion
cls

echo.
echo ============================================
echo   LUMINARK v5 - DEPLOYMENT READY
echo   Quick Start Guide (Windows)
echo ============================================
echo.

REM Check if luminark_v5_upgraded.html exists
if not exist "luminark_v5_upgraded.html" (
    echo ERROR: luminark_v5_upgraded.html not found
    echo.
    echo Make sure you're in the LUMINARK directory:
    echo C:\Users\Forev\OneDrive\Documents\GitHub\LUMINARK
    echo.
    pause
    exit /b 1
)

echo ✓ Files verified
echo.

REM Display what's new
echo ============================================
echo WHAT'S NEW IN v5
echo ============================================
echo.
echo ✓ Sidebar Navigation      - Clean left sidebar
echo ✓ Main Stage Display      - Prominent center focus
echo ✓ Temporal Sentiment      - Future/Past emotion
echo ✓ Life Vectors Dropdown   - 8 life focus areas
echo ✓ Emotional State Input   - Detailed feelings
echo ✓ System Metrics Sliders  - 5 interactive sliders
echo   • Complexity, Stability, Tension
echo   • Adaptability, Coherence
echo ✓ Intention Buttons       - 5 reflection modes
echo ✓ Radar Chart             - Real-time visualization
echo ✓ Deep Reflection Panel   - Stage analysis
echo ✓ Oracle Output Panel     - Guidance + tactics
echo ✓ Professional UI         - Cyber-esoteric design
echo.

REM Deployment options
echo ============================================
echo DEPLOYMENT OPTIONS
echo ============================================
echo.
echo [1] FASTEST - Automated (1 minute)
echo     Command: DEPLOY_V5_NOW.bat
echo     Result: Auto-commits + pushes to Vercel
echo.
echo [2] MANUAL - Git commands (2 minutes)
echo     copy luminark_v5_upgraded.html index.html
echo     git add index.html
echo     git commit -m "Upgrade: LUMINARK v5.0"
echo     git push
echo.
echo [3] WEB - Vercel dashboard (3 minutes)
echo     Visit: https://vercel.com/dashboard
echo     Upload: luminark_v5_upgraded.html as index.html
echo.
echo [4] LOCAL TEST - Before deploying
echo     python -m http.server 8000
echo     Open: http://localhost:8000/luminark_v5_upgraded.html
echo.

REM Show documentation
echo ============================================
echo DOCUMENTATION FILES
echo ============================================
echo.
echo 📖 LUMINARK_V5_UPGRADE_GUIDE.md
echo    - Complete feature guide
echo    - Customization instructions
echo    - API integration details
echo.
echo 🎯 STRATEGIC_IMPROVEMENTS_ROADMAP.md
echo    - What needs improvement
echo    - 4-tier improvement plan
echo    - Monetization strategy
echo    - Implementation timeline
echo.
echo ✅ LUMINARK_V5_COMPLETE_SUMMARY.md
echo    - Feature checklist
echo    - Files created
echo    - QA checklist
echo    - Next steps
echo.

REM Next steps
echo ============================================
echo NEXT STEPS
echo ============================================
echo.
echo TODAY:
echo   1. Review luminark_v5_upgraded.html
echo   2. Test locally or deploy directly
echo   3. Verify at https://luminark-six.vercel.app
echo.
echo THIS WEEK:
echo   1. Gather feedback from users
echo   2. Deploy OpenAI Oracle API
echo   3. Add user history tracking
echo.
echo NEXT MONTH:
echo   1. Launch freemium tier ($5/month)
echo   2. Build analytics dashboard
echo   3. Scale to 1,000+ users
echo.

REM Ready?
echo ============================================
echo READY TO DEPLOY?
echo ============================================
echo.
echo Press:
echo   [1] To run automated deployment
echo   [2] To test locally first
echo   [3] To read documentation
echo   [4] To exit
echo.

set /p choice="Choose (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting deployment...
    call DEPLOY_V5_NOW.bat
) else if "%choice%"=="2" (
    echo.
    echo Starting local server on http://localhost:8000
    echo.
    echo Close this window (Ctrl+C) when done testing.
    echo.
    python -m http.server 8000
) else if "%choice%"=="3" (
    echo.
    echo Opening documentation...
    if exist "LUMINARK_V5_UPGRADE_GUIDE.md" (
        start notepad LUMINARK_V5_UPGRADE_GUIDE.md
    ) else (
        echo Documentation files not found
    )
) else (
    echo Exiting...
    exit /b 0
)

echo.
pause
