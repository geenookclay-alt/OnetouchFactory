@echo off
chcp 65001 >nul
pushd "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ============================================
echo    DDALKKAK - Install (one time only)
echo    No Python install needed. Just wait.
echo ============================================
echo.
if not exist "pyembed\python.exe" ( echo [X] pyembed missing. Re-extract the zip into a Windows folder. & pause & exit /b )
echo [1/4] Preparing installer...
pyembed\python.exe pyembed\get-pip.py --no-warn-script-location -q
echo [2/4] Installing packages (3-5 min, needs internet)...
pyembed\python.exe -m pip install -r requirements.txt --timeout 60 --retries 10
if errorlevel 1 ( echo [X] Package install failed. Screenshot and ask support. & pause & exit /b )
echo [3/4] Installing ffmpeg and tools...
if exist "pyembed\Scripts\yt-dlp.exe" copy /Y "pyembed\Scripts\yt-dlp.exe" "pyembed\" >nul
pyembed\python.exe setup_ffmpeg.py
echo [4/4] API key setup...
pyembed\python.exe setup_env.py
echo.
echo ============================================
echo    DONE!  Now double-click  start.bat
echo ============================================
pause
