# Register a Windows Scheduled Task that runs the FX daily refresh on weekdays at 5:15 PM local.
#
# Idempotent: removes any existing task with the same name and recreates it.
# Runs as the current interactive user (no admin required). If you need it to run while you are
# logged out, run this script as Administrator and pass -RunWhetherLoggedOnOrNot $true.
#
# This assumes the machine's local time zone is America/New_York (the dashboard's data-cutoff
# rule is 5 PM NY). If the machine is in a different time zone, edit $RefreshTime below to the
# local equivalent of 17:15 NY.
#
# Usage:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File `
#     "C:\Users\mrmhr\OneDrive\Documents\Python\Mo_Dash\ETF\ETF Forecaster\project\scripts\setup_fx_daily_refresh_task.ps1"
#
# Remove later:
#
#   Unregister-ScheduledTask -TaskName "FX Daily Refresh (5.15 PM NY)" -Confirm:$false
#
# Note: the task name uses "5.15" instead of "5:15" because Task Scheduler treats ":" as a
# task-path separator (\folder\subfolder), which makes Register-ScheduledTask fail with
# HRESULT 0x80070057 (E_INVALIDARG).

param(
    [bool]$RunWhetherLoggedOnOrNot = $false
)

$ErrorActionPreference = "Stop"

$TaskName    = "FX Daily Refresh (5.15 PM NY)"
$RunnerPath  = (Resolve-Path (Join-Path $PSScriptRoot "run_fx_daily_refresh.ps1")).Path
$WorkingDir  = Split-Path $RunnerPath
$RefreshTime = (Get-Date "17:15")   # 5:15 PM local clock (= 5:15 PM NY if machine TZ is America/New_York)
$UserId      = if ($env:USERDOMAIN) { "$env:USERDOMAIN\$env:USERNAME" } else { $env:USERNAME }

if (-not (Test-Path $RunnerPath)) { throw "Missing runner script: $RunnerPath" }

# Trigger: weekly, Mon-Fri at 17:15 local. Using a Get-Date object avoids HRESULT 0x80070057
# from PowerShell 5.1's looser AM/PM literal parsing.
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At $RefreshTime

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$RunnerPath`"" `
    -WorkingDirectory $WorkingDir

# Drop -RunOnlyIfNetworkAvailable (causes 0x80070057 in some PS5.1 configurations). The runner
# script will fail loudly if the network is down, which is fine for a 5 PM refresh.
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# Idempotent re-register.
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null

if ($RunWhetherLoggedOnOrNot) {
    # Requires Administrator. S4U lets the task run without a logged-on session.
    Register-ScheduledTask `
        -TaskName  $TaskName `
        -Action    $Action `
        -Trigger   $Trigger `
        -Settings  $Settings `
        -User      $UserId `
        -RunLevel  Limited `
        -Description "Runs run_fx_daily_refresh.ps1 weekdays at 5:15 PM local (intended to be NY time)." | Out-Null
    $LogonNote = "$UserId (S4U: runs whether logged on or not; requires Admin to register)"
}
else {
    # Default: interactive user. Works without admin, fires only while user session exists.
    # -User must be passed explicitly here; some PS5.1 configurations reject the cmdlet with
    # HRESULT 0x80070057 when it has to infer the principal from $Settings alone.
    Register-ScheduledTask `
        -TaskName  $TaskName `
        -Action    $Action `
        -Trigger   $Trigger `
        -Settings  $Settings `
        -User      $UserId `
        -RunLevel  Limited `
        -Description "Runs run_fx_daily_refresh.ps1 weekdays at 5:15 PM local (intended to be NY time)." | Out-Null
    $LogonNote = "$UserId (interactive: fires only while you are logged in)"
}

Write-Host "Registered scheduled task: $TaskName"
Write-Host "Trigger : Mon-Fri at $($RefreshTime.ToString('h:mm tt')) local clock"
Write-Host "Runs as : $LogonNote"
Write-Host "Runner  : $RunnerPath"
Write-Host ""
Write-Host "Verify with:"
Write-Host "  Get-ScheduledTask -TaskName `"$TaskName`" | Get-ScheduledTaskInfo"
Write-Host ""
Write-Host "If this machine is not in America/New_York, edit `$RefreshTime in this script."
