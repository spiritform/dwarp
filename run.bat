@echo off
REM DWARP launcher — run install.bat once first.
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo DWARP environment not found. Run install.bat first.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0server.py" %*
