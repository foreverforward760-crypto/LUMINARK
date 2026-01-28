@echo off
echo PREPARING LUMINARK FOR DEPLOYMENT...
echo.

:: 1. Copy the Professional V6.2 Engine to the root
copy /Y "DEPLOY_ME_NOW\index.html" "index.html"

echo.
echo SUCCESS! 
echo.
echo The new "index.html" is now ready in this folder.
echo You can now deploy this entire folder to Vercel or Netlify.
echo.
pause
