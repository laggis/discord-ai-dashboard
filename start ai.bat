@echo off
:loop
REM Navigate to the directory
cd /d "C:\Users\LaGgIs\Desktop\AI"

REM Run the Python script
python AI.py

REM Wait for 4 hours (14400 seconds)
timeout /t 14400 /nobreak

REM Restart the process
goto loop