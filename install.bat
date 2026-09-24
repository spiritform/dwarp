@echo off
REM DWARP one-click installer. Everything lands inside this folder; nothing touches
REM your system Python. Safe to re-run: finished steps are skipped.
setlocal EnableExtensions
cd /d "%~dp0"

REM ---- pinned downloads (verified by SHA-256) -------------------------------------
set "UV_VER=0.12.18"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VER%/uv-x86_64-pc-windows-msvc.zip"
set "UV_SHA=cae6a3bc25239f83dffb467a4b180508d9da23986c04639ebfa44e43e6a84bff"
set "FF_URL=https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip"
set "FF_SHA=fa7d4d7e795db0e2503f49f105f46ed5852386f0cfdd819899be3b65ebde24fc"
set "RIFE_URL=https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife49.pth"
set "RIFE_SHA=e55fd00f3cc184e3c65961f4bb827a9da022e78eed36b055242c0ac30000d533"
REM PyTorch CUDA build. cu128 needs an NVIDIA driver from 2025 or newer.
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
REM -------------------------------------------------------------------------------

set "FETCH=powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\fetch.ps1""
set "PY=%~dp0.venv\Scripts\python.exe"
REM keep uv's own Python install inside this folder too
set "UV_PYTHON_INSTALL_DIR=%~dp0tools\python"

echo.
echo  DWARP installer
echo  ---------------
where nvidia-smi >nul 2>nul
if errorlevel 1 echo  WARNING: no NVIDIA driver found. DWARP needs an NVIDIA GPU with CUDA.

echo.
echo [1/6] uv (Python package manager)
if not exist "tools\uv\uv.exe" (
    %FETCH% -Url "%UV_URL%" -Sha256 %UV_SHA% -Dest "tools\uv.zip" || goto :fail
    powershell -NoProfile -Command "Expand-Archive -LiteralPath 'tools\uv.zip' -DestinationPath 'tools\uv' -Force" || goto :fail
    del "tools\uv.zip"
) else (
    echo   already have uv
)
set "UV=%~dp0tools\uv\uv.exe"

echo.
echo [2/6] Python 3.12 environment
if not exist "%PY%" (
    "%UV%" venv --python 3.12 .venv || goto :fail
) else (
    echo   already have .venv
)

echo.
echo [3/6] PyTorch with CUDA (about 3 GB the first time)
"%PY%" -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.11') and torch.version.cuda else 1)" >nul 2>nul
if errorlevel 1 (
    "%UV%" pip install --python "%PY%" torch==2.11.0 torchvision==0.26.0 --index-url %TORCH_INDEX% || goto :fail
) else (
    echo   already have torch
)

echo.
echo [4/6] DWARP packages
"%UV%" pip install --python "%PY%" -r requirements.txt || goto :fail

echo.
echo [5/6] ffmpeg and RIFE weights
where ffmpeg >nul 2>nul
if errorlevel 1 (
    if not exist "tools\ffmpeg\bin\ffmpeg.exe" (
        %FETCH% -Url "%FF_URL%" -Sha256 %FF_SHA% -Dest "tools\ffmpeg.zip" || goto :fail
        powershell -NoProfile -Command "Expand-Archive -LiteralPath 'tools\ffmpeg.zip' -DestinationPath 'tools' -Force" || goto :fail
        move /y "tools\ffmpeg-7.1-essentials_build" "tools\ffmpeg" >nul || goto :fail
        del "tools\ffmpeg.zip"
    ) else (
        echo   already have ffmpeg
    )
) else (
    echo   ffmpeg found on PATH
)
%FETCH% -Url "%RIFE_URL%" -Sha256 %RIFE_SHA% -Dest "models\rife\rife49.pth" || goto :fail

echo.
echo [6/6] models folder
"%PY%" "%~dp0scripts\setup_config.py" || goto :fail

echo.
echo  Done. Start DWARP with run.bat
echo.
pause
exit /b 0

:fail
echo.
echo  Install failed. Scroll up for the error; re-running install.bat resumes where it stopped.
echo.
pause
exit /b 1
