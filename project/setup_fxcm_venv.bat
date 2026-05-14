@echo off
REM Creates:
REM   .venv      — Python 3.11 (Streamlit + dashboard deps)
REM   .venv-fc   — Python 3.7 + forexconnect (PyPI only ships Windows wheels for cp37)
title Setup FXCM + dashboard venvs
cd /d "%~dp0"
set PYTHONUTF8=1

where py >nul 2>nul
if errorlevel 1 (
  echo Install the Python Launcher for Windows ^(py^).
  pause
  exit /b 1
)

echo === Python 3.11 for .venv ===
py -3.11 -c "import sys; print(sys.executable)" 2>nul
if errorlevel 1 (
  echo Install Python 3.11 ^(winget: Python.Python.3.11^).
  pause
  exit /b 1
)

echo === Python 3.7 for .venv-fc ^(forexconnect on Windows^) ===
py -3.7 -c "import sys; print(sys.executable)" 2>nul
if errorlevel 1 (
  echo Install Python 3.7 ^(winget: Python.Python.3.7^).
  pause
  exit /b 1
)

if exist ".venv" (
  echo Removing old .venv ...
  rmdir /s /q ".venv"
)
echo Creating .venv ^(3.11^)...
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Failed.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install -U pip
pip install -r requirements.txt
if errorlevel 1 (
  echo pip install requirements.txt failed.
  pause
  exit /b 1
)
call deactivate 2>nul

if exist ".venv-fc" (
  echo Removing old .venv-fc ...
  rmdir /s /q ".venv-fc"
)
echo Creating .venv-fc ^(3.7 + forexconnect^)...
py -3.7 -m venv .venv-fc
if errorlevel 1 (
  echo Failed.
  pause
  exit /b 1
)
call ".venv-fc\Scripts\activate.bat"
python -m pip install -U "pip<24"
pip install -r requirements-fc.txt
if errorlevel 1 (
  echo pip install requirements-fc.txt failed.
  pause
  exit /b 1
)
python -c "import forexconnect; print('forexconnect OK:', forexconnect.__file__)" 2>nul
if errorlevel 1 (
  echo forexconnect import failed in .venv-fc.
  pause
  exit /b 1
)
call deactivate 2>nul

echo.
echo Done. Dashboard: mo_dash.bat  ^(uses .venv 3.11; FXCM orders call .venv-fc 3.7^)
pause
