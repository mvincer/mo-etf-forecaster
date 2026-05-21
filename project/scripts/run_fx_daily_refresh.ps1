# Daily FX refresh — same pipeline as the dashboard's "Refresh forecasts now" button,
# extended with the daily auto-trade check (step 4).
#
# Steps:
#   1) fx_data_collect.run_collect          (raw FX parquets, skips FRED for speed)
#   2) fx_data_collect.run_model_inputs     (fundamentals + technicals panels)
#   3) fxnl.daily_binary_forecast_report    (--align-to-next-bar --inject-live-quote)
#      → writes Mo_Dash\FX\FX forecasts\non-linear FX forecast - daily_binary_fx_forecast\
#         daily_binary_fx_forecast.xlsx
#   4) scripts/daily_trade_check.py         (analysis + reporting; only executes trades
#                                            when Mo_Dash\Dashboard\auto_trade.config.json
#                                            has "enabled": true)
#      → writes daily_trade_check.json next to the workbook for the dashboard panel
#
# All code and data live under Mo_Dash\ (single source of truth). This script may be
# launched from either the Mo_Dash copy of ETF Forecaster (preferred) or the legacy
# sibling — it always resolves paths relative to Mo_Dash.
#
# Logs to:  <FxNlRoot>\daily_binary_fx_forecast.refresh.log
# Exits 0 on success, non-zero on any failure. Designed to be called by Task Scheduler
# at 5:15 PM America/New_York every weekday (the dashboard already enforces a 5 PM NY
# cutoff for data dates — see fxnl.fx_session_calendar).

$ErrorActionPreference = "Stop"

# This script lives at: <EtfRoot>\project\scripts\run_fx_daily_refresh.ps1
$EtfRoot    = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$MoDashRoot = (Resolve-Path (Join-Path $EtfRoot "..\..")).Path  # …\Mo_Dash
$FxCollect  = Join-Path $MoDashRoot "FX\FX forecasts"
$FxNlRoot   = Join-Path $FxCollect "non-linear FX forecast - daily_binary_fx_forecast"
$OutXlsx    = Join-Path $FxNlRoot "daily_binary_fx_forecast.xlsx"
$VenvPy     = Join-Path $EtfRoot "project\.venv\Scripts\python.exe"
$LogPath    = Join-Path $FxNlRoot "daily_binary_fx_forecast.refresh.log"

if (-not (Test-Path $VenvPy))    { throw "Missing venv python: $VenvPy" }
if (-not (Test-Path $FxCollect)) { throw "Missing FX data root: $FxCollect" }
if (-not (Test-Path $FxNlRoot))  { throw "Missing FX NL project: $FxNlRoot" }

New-Item -ItemType Directory -Force -Path (Split-Path $OutXlsx) | Out-Null
# Always write the log as UTF-8 (Add-Content/Tee-Object default to ASCII/UTF-16 on PS 5.1,
# which mixes encodings in the same file and makes the log unreadable later).
"--- scheduled refresh started $(Get-Date -Format o)" | Out-File -FilePath $LogPath -Encoding utf8
"--- runner: $PSCommandPath" | Out-File -FilePath $LogPath -Encoding utf8 -Append
"--- MoDashRoot: $MoDashRoot" | Out-File -FilePath $LogPath -Encoding utf8 -Append

$env:PYTHONUNBUFFERED = "1"
$env:MO_DASH_ROOT = $MoDashRoot
$env:FXNL_PROJECT_ROOT = $FxNlRoot

function Invoke-Step {
    param([string]$StepName, [string]$Cwd, [string[]]$PyArgs)
    "`n=== $StepName (cwd=$Cwd) ===" | Out-File -FilePath $LogPath -Encoding utf8 -Append
    Push-Location $Cwd
    # Python's logging module writes INFO/WARN to stderr by default. With the script-level
    # $ErrorActionPreference = "Stop" in effect, PowerShell would treat ANY stderr line as a
    # fatal terminating error and abort before run_collect could finish. We locally set it to
    # Continue, then check $LASTEXITCODE explicitly to decide success/failure.
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Capture stderr+stdout to a single stream; render as text; append as UTF-8 so the
        # whole log stays in one consistent encoding (Tee-Object's default is UTF-16 on PS5.1).
        & $VenvPy @PyArgs 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $LogPath -Encoding utf8 -Append
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $saved
        Pop-Location
    }
    if ($code -ne 0) { throw "$StepName failed (exit $code)" }
}

Invoke-Step -StepName "run_collect"      -Cwd $FxCollect -PyArgs @("-m","fx_data_collect.run_collect","--skip-fred","--no-chart")
Invoke-Step -StepName "run_model_inputs" -Cwd $FxCollect -PyArgs @("-m","fx_data_collect.run_model_inputs")
Invoke-Step -StepName "daily_report"     -Cwd $FxNlRoot  -PyArgs @(
    "-m","fxnl.daily_binary_forecast_report",
    "--align-to-next-bar","--inject-live-quote",
    "--out-xlsx",$OutXlsx
)

# Step 4: daily auto-trade check. Runs from the ETF Forecaster project root so that
# `python scripts/daily_trade_check.py` can import `utils.*` and `ui.*` without
# extra PYTHONPATH gymnastics. Reads auto_trade.config.json; only EXECUTES trades
# when "enabled": true (default false). Always writes daily_trade_check.json.
$ProjectDir = Join-Path $EtfRoot "project"
Invoke-Step -StepName "daily_trade_check" -Cwd $ProjectDir -PyArgs @("scripts\daily_trade_check.py")

"--- refresh finished OK $(Get-Date -Format o)" | Out-File -FilePath $LogPath -Encoding utf8 -Append
exit 0
