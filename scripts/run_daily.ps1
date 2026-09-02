# ETF Forecaster - daily job: ingest, rebuild features, forecast, PDF, email.
$ErrorActionPreference = "Continue"
$py = "C:\Dev\Mo_Dash\app\.venv\Scripts\python.exe"
$repo = "C:\Dev\mo_etf_forecaster"
$log = Join-Path $repo ("logs\daily_" + (Get-Date -Format "yyyyMMdd") + ".log")
New-Item -ItemType Directory -Force -Path (Join-Path $repo "logs") | Out-Null
Set-Location $repo
& $py -m etf_forecaster.pipeline.run_all --jobs 8 *>> $log
exit $LASTEXITCODE
