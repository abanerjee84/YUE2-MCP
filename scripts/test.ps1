$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    & .venv/Scripts/ruff.exe check src tests scripts/benchmark.py
    if ($LASTEXITCODE) { throw 'Lint failed.' }
    $testRoot = Join-Path $projectRoot ('data/test-' + [guid]::NewGuid().ToString('N'))
    & .venv/Scripts/python.exe -m pytest -q --basetemp $testRoot
    if ($LASTEXITCODE) { throw 'Tests failed.' }
} finally { Pop-Location }
