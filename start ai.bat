@echo off
:loop
REM Navigate to the directory
cd /d "C:\Path\To\Your\Bot\Directory"

REM Run both Python scripts in parallel
start "AI Script" cmd /c "python AI.py"
start "Web Config" cmd /c "python web_config.py"

REM Wait for 4 hours (14400 seconds)
timeout /t 14400 /nobreak

REM Restart the process
goto loop