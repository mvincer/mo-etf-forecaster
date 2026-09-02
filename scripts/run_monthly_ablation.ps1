# ETF Forecaster - monthly job (first weekend): full staged ablation + Optuna search
$ErrorActionPreference = "Continue"
$py = "C:\Dev\Mo_Dash\app\.venv\Scripts\python.exe"
$repo = "C:\Dev\mo_etf_forecaster"
$log = Join-Path $repo ("logs\ablation_" + (Get-Date -Format "yyyyMMdd") + ".log")
New-Item -ItemType Directory -Force -Path (Join-Path $repo "logs") | Out-Null
Set-Location $repo
& $py -m etf_forecaster.pipeline.run_all --retrain --full-search --jobs 10 *>> $log
exit $LASTEXITCODE
