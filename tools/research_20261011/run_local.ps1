param([switch]$Sync, [switch]$ShortWindows)
$ErrorActionPreference = 'Stop'
$ResearchRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$ResearchPython = Join-Path $ResearchRoot '.venv_research_20261011/Scripts/python.exe'
Set-Location -LiteralPath $ResearchRoot
$env:PYTHONIOENCODING = 'utf-8'
function Run-Stage([string]$Name) {
    & $ResearchPython (Join-Path $PSScriptRoot $Name)
    if ($LASTEXITCODE -ne 0) { throw "Research stage failed: $Name" }
}
if ($Sync) {
    Run-Stage 'audit_sync.py'
    Run-Stage 'supplement_audit.py'
    Run-Stage 'inspect_supplement.py'
}
Run-Stage 'verify.py'
Run-Stage 'prepare_data.py'
Run-Stage 'regimes.py'
Run-Stage 'train_models.py'
Run-Stage 'evaluate_models.py'
Run-Stage 'risk_refinement.py'
Run-Stage 'portfolio_extension.py'
Run-Stage 'intraday.py'
Run-Stage 'factor_timing.py'
Run-Stage 'legacy_extensions.py'
Run-Stage 'jev_adapter.py'
Run-Stage 'period_audit.py'
if ($ShortWindows) {
    & $ResearchPython (Join-Path $PSScriptRoot 'regimes.py') --window 126
    if ($LASTEXITCODE -ne 0) { throw 'Short-window HMM failed' }
    Run-Stage 'short_windows.py'
}
Run-Stage 'requirements_matrix.py'
Run-Stage 'final_checks.py'
Run-Stage 'recent_diagnosis.py'
Run-Stage 'report_assets.py'
# Jev remains offline. Public Kronos checkpoint probe and DOCX rendering are separate documented stages.
