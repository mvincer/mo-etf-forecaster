# Registers the ETF Forecaster scheduled tasks (run from an elevated PowerShell):
#   ETF_Forecaster_Daily    - weekdays 17:45 (local; machine runs US Eastern)
#   ETF_Forecaster_Weekly   - Saturday 08:00, walk-forward retrain
#   ETF_Forecaster_Monthly  - 1st Saturday 10:00, full ablation grid
$repo = "C:\Dev\mo_etf_forecaster"
$ps = "powershell.exe"

function Register-Job($name, $script, $trigger) {
    $action = New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$repo\scripts\$script`""
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 20) `
        -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    Write-Host "registered $name"
}

$daily = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 17:45
Register-Job "ETF_Forecaster_Daily" "run_daily.ps1" $daily

$weekly = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 08:00
Register-Job "ETF_Forecaster_Weekly" "run_weekly_retrain.ps1" $weekly

# First Saturday of the month (schtasks supports MONTHLY/FIRST directly)
schtasks /Create /F /TN "ETF_Forecaster_Monthly" /SC MONTHLY /MO FIRST /D SAT /ST 10:00 `
    /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"$repo\scripts\run_monthly_ablation.ps1\"" | Out-Null
Write-Host "registered ETF_Forecaster_Monthly"
Write-Host "Done. Verify with: Get-ScheduledTask -TaskName 'ETF_Forecaster_*'"
