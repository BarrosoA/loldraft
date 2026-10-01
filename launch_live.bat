@echo off
title LolDraft Live Companion
cd /d "%~dp0"
echo ======================================================================
echo               LOLDRAFT LIVE COMPANION (v1.0)
echo         Real-Time Champion Select Machine Learning Advisor
echo ======================================================================
echo.
python connector\lcu_socket.py
if %errorlevel% neq 0 (
    echo.
    echo [LolDraft] Process exited with code %errorlevel%.
)
echo.
pause
