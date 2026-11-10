param([switch]$DownloadModels, [switch]$RegisterCodex, [switch]$SkipNativeBuild)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first.' }
    if (-not (Test-Path '.venv/Scripts/python.exe')) {
        uv --system-certs venv --python 3.12 .venv
        if ($LASTEXITCODE) { throw 'Python environment creation failed.' }
    }
    uv --system-certs pip install --python .venv/Scripts/python.exe -r requirements.lock.txt
    if ($LASTEXITCODE) { throw 'Locked dependency installation failed.' }
    uv --system-certs pip install --python .venv/Scripts/python.exe -e .
    if ($LASTEXITCODE) { throw 'Application installation failed.' }
    if (-not $SkipNativeBuild) {
        if (-not (Test-Path 'vendor/yue2.cpp/.git')) {
            git clone https://github.com/ServeurpersoCom/yue2.cpp.git vendor/yue2.cpp
            if ($LASTEXITCODE) { throw 'Native clone failed.' }
        }
        git -C vendor/yue2.cpp checkout 5a9500fd2979ab99fc82895877aadf8583acbabb
        if ($LASTEXITCODE) { throw 'Pinned native checkout failed.' }
        git -C vendor/yue2.cpp submodule update --init
        if ($LASTEXITCODE) { throw 'Pinned GGML checkout failed.' }
        & .\scripts\build-native.cmd
        if ($LASTEXITCODE) { throw 'Native CUDA build failed.' }
    }
    & .venv/Scripts/python.exe -m yue_music install-cli
    if ($LASTEXITCODE) { throw 'CLI installation failed.' }
    if ($DownloadModels) {
        & .venv/Scripts/python.exe -m yue_music setup-models
        if ($LASTEXITCODE) { throw 'Model download failed.' }
    }
    if ($RegisterCodex) {
        & .venv/Scripts/python.exe -m yue_music register codex
        if ($LASTEXITCODE) { throw 'MCP registration failed.' }
    }
} finally { Pop-Location }
