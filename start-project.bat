@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-project.ps1" %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if not "%EXIT_CODE%"=="0" (
    echo Startup failed. Please check the messages above and logs directory.
) else (
    echo Startup finished.
)
echo.
pause
exit /b %EXIT_CODE%
