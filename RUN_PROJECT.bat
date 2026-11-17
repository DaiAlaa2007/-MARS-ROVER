@echo off
title Mars Rover Pathfinder - Mission Control (Member 4)
cd /d "%~dp0"
echo ========================================================
echo Launching Mars Rover Pathfinder AI Mission Control...
echo ========================================================
"C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe" main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Error] Project exited with code %ERRORLEVEL%.
    pause
)
