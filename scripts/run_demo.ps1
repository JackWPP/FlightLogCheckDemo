$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is required. Install uv first, then rerun this script."
}

$Port = if ($env:DEMO_PORT) { [int]$env:DEMO_PORT } else { 8003 }

Write-Host "Project: $ProjectRoot"
Write-Host "Installing/checking dependencies with uv..."
uv sync --python 3.12

Write-Host "Starting demo: http://127.0.0.1:$Port/"
uv run --python 3.12 uvicorn formcheck.app:app --host 127.0.0.1 --port $Port
