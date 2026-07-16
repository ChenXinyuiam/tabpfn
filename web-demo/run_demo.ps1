$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

Write-Host "Starting TriFocus ForceLab demo..."
Write-Host "Open: http://127.0.0.1:8000"
Write-Host "Keep this window open. Press Ctrl+C to stop."

python web-demo\server.py --host 127.0.0.1 --port 8000
