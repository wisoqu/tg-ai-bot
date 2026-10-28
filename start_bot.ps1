$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw "Install Python 3.10+ first: https://www.python.org/downloads/" }
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    python -m venv .venv
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
}
if (-not (Test-Path (Join-Path $root ".env"))) {
    & $venvPython manage.py setup
}
& $venvPython manage.py check
& $venvPython main.py
