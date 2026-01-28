@echo off
cd /d "c:\Users\Forev\OneDrive\Documents\GitHub\LUMINARK"
echo Pushing HUMAN-CENTRIC LUMINARK Update...
git add luminark_supercharged_v4_COMPLETE.py
git commit -m "UI Overhaul: Earthy language, Multi-select, Journaling, Graph"
git push
echo.
echo Deployment triggered! Check Vercel in 60 seconds.
pause