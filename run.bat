@echo off
REM DWARP launcher. Run install.bat once first.
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo DWARP is not installed yet. Run install.bat first.
    pause
    exit /b 1
)
REM ffmpeg downloaded by install.bat (only when none was on PATH)
if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"
"%~dp0.venv\Scripts\python.exe" "%~dp0server.py" %*
