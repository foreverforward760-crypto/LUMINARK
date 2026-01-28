@echo off
echo Cleaning up LUMINARK workspace...

mkdir "_ARCHIVE" 2>nul
mkdir "docs" 2>nul
mkdir "scripts" 2>nul
mkdir "Web_Deployment_v6" 2>nul

echo Moving Archives...
move /Y "LUMINARK_ARCHIVE" "_ARCHIVE\"
move /Y "LUMINARK_DEPLOY" "_ARCHIVE\"
move /Y "NIKKI_TASK_PACKAGE" "_ARCHIVE\"
move /Y "Markting" "_ARCHIVE\"
move /Y "youtube_content" "_ARCHIVE\"

echo Moving Obsolete Assessment Tools...
move /Y "sap_deep_assessment.html" "_ARCHIVE\"
move /Y "sap_free_assessment.html" "_ARCHIVE\"

echo Moving Documentation...
move /Y "LUMINARK_DOCS" "_ARCHIVE\"
move /Y "*.md" "docs\"
rem Keep README in root if desired, or move it back. Assuming move all.
move /Y "docs\README.md" ".\" 
move /Y "docs\MASTER_INTEGRATION_SUMMARY.md" ".\"

echo Moving Scripts...
move /Y "*.bat" "scripts\"
rem Don't move THIS script yet
move /Y "scripts\CLEANUP_WORKSPACE.bat" ".\"
move /Y "scripts\run_local.bat" ".\"

echo Configuring Deployment Folder...
xcopy /E /I /Y "DEPLOY_ME_NOW" "Web_Deployment_v6"
rmdir /S /Q "DEPLOY_ME_NOW"

echo Cleanup Complete!
pause
