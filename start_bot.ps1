$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$envFile = Join-Path $root ".env"
$pythonExe = Join-Path $root "venv\\Scripts\\python.exe"
$mainFile = Join-Path $root "main.py"

if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $root ".env.example") $envFile
    throw ".env was missing and has been created from .env.example. Fill required keys and rerun."
}

$envText = Get-Content $envFile -Raw

$tokenMatch = [regex]::Match($envText, "(?m)^TELEGRAM_BOT_TOKEN=(.*)$")
$token = $tokenMatch.Groups[1].Value.Trim()
if (-not $token -or $token -eq "your_telegram_token_here") {
    throw "Set TELEGRAM_BOT_TOKEN in .env before launch."
}

$geminiMatch = [regex]::Match($envText, "(?m)^GEMINI_API_KEY=(.*)$")
$geminiKey = $geminiMatch.Groups[1].Value.Trim()
if (-not $geminiKey -or $geminiKey -eq "your_gemini_api_key_here") {
    throw "Set GEMINI_API_KEY in .env before launch."
}

& $pythonExe $mainFile

