param([switch]$LiveProvider)
$ErrorActionPreference = 'Stop'
$root = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..\..\backend'
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("no-code-ml-verify-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
try {
  $env:UPLOAD_DIR = Join-Path $tempRoot 'uploads'
  $env:ARTIFACTS_DIR = Join-Path $tempRoot 'artifacts'
  $env:STATE_DIR = Join-Path $tempRoot 'state'
  $env:AI_DB_FILE = Join-Path $env:STATE_DIR 'ai_control_plane.sqlite3'
  Push-Location $root
  python -m pytest tests/test_ml.py tests/test_ai_execution.py tests/test_ai_security_persistence.py -q
  python -m compileall -q ai main.py config.py
  if ($LiveProvider) { python -m pytest tests/test_ai_live_provider.py -q }
  Pop-Location
} finally {
  Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

