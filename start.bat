@echo off
chcp 65001 >nul
pushd "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set SOLO_MODE=1
echo ============================================
echo    DDALKKAK is starting...
echo    Wait for this line:
echo        Uvicorn running on http://127.0.0.1:8000
echo    Then type  127.0.0.1:8000  in your web browser.
echo    (First run can take 1-2 minutes.)
echo ============================================
echo.
pyembed\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
echo.
echo [Server stopped. If there is a red error above, screenshot it.]
pause
