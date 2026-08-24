$ErrorActionPreference = "Stop"

$prefLab = "pref-lab"
if (Test-Path ".\.venv\Scripts\pref-lab.exe") {
    $prefLab = ".\.venv\Scripts\pref-lab.exe"
}

Write-Host "Running pref-lab validate..."
& $prefLab validate data/sample_preferences.jsonl

Write-Host "`nRunning pref-lab evaluate..."
& $prefLab evaluate --config configs/local.yaml

Write-Host "`nDisplaying outputs/metrics.json:"
Get-Content outputs/metrics.json
