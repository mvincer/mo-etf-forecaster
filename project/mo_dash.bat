@echo off
REM MO Dashboard — Forecasting Dashboard (Desktop shortcut "mo_dash" points here).
REM FXCM forexconnect: only installs on Python 3.10–3.11 (see requirements.txt).
REM If .venv was made with 3.12+, run setup_fxcm_venv.bat once.
title MO Dashboard
cd /d "%~dp0"
set PYTHONUTF8=1

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(3 if sys.version_info >= (3, 12) else 0)" 2>nul
  if errorlevel 3 (
    echo.
    echo [mo_dash] This .venv uses Python 3.12+ — forexconnect is skipped by pip.
    echo           Recreate with Python 3.11:  setup_fxcm_venv.bat
    echo           Launching with system py -3.11 if available...
    echo.
    goto try_py
  )
  if not errorlevel 1 (
    call ".venv\Scripts\activate.bat"
    python main.py
    if errorlevel 1 (
      echo.
      echo Launch failed. For FXCM trading run: setup_fxcm_venv.bat
      echo Then: pip install -r requirements.txt
      pause
      exit /b 1
    )
    exit /b 0
  )
)

:try_py
py -3.11 main.py
if not errorlevel 1 exit /b 0
py -3.10 main.py
if not errorlevel 1 exit /b 0
py -3 main.py
if not errorlevel 1 exit /b 0
python main.py
if not errorlevel 1 exit /b 0

echo.
echo Could not start the dashboard.
echo 1^) Install Python 3.11 from python.org ^(include "py launcher"^).
echo 2^) In this folder run:  setup_fxcm_venv.bat
echo 3^) Then run:  mo_dash.bat
echo.
pause
exit /b 1
