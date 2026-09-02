# ETF Forecaster - Saturday job: full ingest + walk-forward retrain (fast budgets)
$ErrorActionPreference = "Continue"
$py = "C:\Dev\Mo_Dash\app\.venv\Scripts\python.exe"
$repo = "C:\Dev\mo_etf_forecaster"
$log = Join-Path $repo ("logs\retrain_" + (Get-Date -Format "yyyyMMdd") + ".log")
New-Item -ItemType Directory -Force -Path (Join-Path $repo "logs") | Out-Null
Set-Location $repo
& $py -m etf_forecaster.pipeline.run_all --retrain --jobs 10 *>> $log
exit $LASTEXITCODE
