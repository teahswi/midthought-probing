param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$ConfigPath = ".\configs\local_qwen3_8b_windows.yaml",
    [ValidateSet("dry-run", "smoke", "full")]
    [string]$Mode = "dry-run"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

if (!(Test-Path $ModelPath -PathType Container)) {
    throw "Model path does not exist or is not a directory: $ModelPath"
}

$python = ".\.venv\Scripts\python.exe"
if (!(Test-Path $python)) {
    throw "Virtual environment not found. Create it with: py -3.11 -m venv .venv"
}

if ($Mode -eq "dry-run") {
    & $python .\main.py generation --config $ConfigPath --dry-run
    & $python .\main.py hidden-state-extraction --config $ConfigPath --dry-run
    exit $LASTEXITCODE
}

if ($Mode -eq "smoke") {
    & $python .\main.py make-splits --config $ConfigPath
    & $python .\main.py generation --config $ConfigPath --split validation
    exit $LASTEXITCODE
}

& $python .\main.py make-splits --config $ConfigPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python .\main.py generation --config $ConfigPath --split validation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python .\main.py hidden-state-extraction --config $ConfigPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python .\main.py probe-training --config $ConfigPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python .\main.py baseline-evaluation --config $ConfigPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python .\main.py plotting --config $ConfigPath
