@echo off
echo ===================================================
echo   LUMINARK DEPLOYMENT SCRIPT
echo ===================================================
echo.
echo 1. Deploying Void Terminal to Netlify...
echo.

cd "c:\Users\Forev\OneDrive\Documents\GitHub\LUMINARK\DEPLOY_ME_NOW"

call netlify deploy --prod

echo.
echo ===================================================
echo   DEPLOYMENT COMPLETE!
echo   Copy the 'Website URL' from above.
echo ===================================================
pause