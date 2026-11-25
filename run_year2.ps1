param(
 [string]$Python = "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe",
 [string]$Pax = "$env:USERPROFILE\Downloads\AA_PAX_FORECASTING (1).xlsx",
 [string]$Airports = "$env:USERPROFILE\Downloads\AA_AIRPORT_INFO.xlsx",
 [switch]$RebuildData
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Interpreter not found: $Python" }
if (-not (Test-Path '.venv\Scripts\python.exe')) {
 & $Python -m venv .venv
 if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
$YearTwoPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $YearTwoPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if ($RebuildData -or -not (Test-Path 'data\processed\panel.npz')) {
 & $YearTwoPython prepare.py --pax $Pax --airports $Airports
 if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed' }
}
& $YearTwoPython model.py
if ($LASTEXITCODE -ne 0) { throw 'Model run failed' }
& $YearTwoPython -m unittest -v test_model.py
if ($LASTEXITCODE -ne 0) { throw 'Validation failed' }
& $YearTwoPython validate_results.py
if ($LASTEXITCODE -ne 0) { throw 'Output reconciliation failed' }
& $YearTwoPython freeze_forecast.py
if ($LASTEXITCODE -ne 0) { throw 'Forecast snapshot failed' }
& $YearTwoPython evaluate_issue.py
if ($LASTEXITCODE -ne 0) { throw 'Frozen forecast evaluation failed' }
Write-Host 'Second-year results saved to data\processed\results'
