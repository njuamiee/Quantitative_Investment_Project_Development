param(
    [ValidateSet('Report','Verify','All')][string]$Mode = 'Report',
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$bundledPython = Join-Path $projectRoot '.venv_research_20261011/Scripts/python.exe'
if ($PythonPath) {
    if (Test-Path -LiteralPath $PythonPath) {
        $researchPython = (Resolve-Path -LiteralPath $PythonPath).Path
    } else {
        $explicitCommand = Get-Command -Name $PythonPath -CommandType Application -ErrorAction SilentlyContinue
        if (-not $explicitCommand) { throw "PythonPath not found: $PythonPath" }
        $researchPython = $explicitCommand.Source
    }
} elseif (Test-Path -LiteralPath $bundledPython) {
    $researchPython = $bundledPython
} else {
    $fallbackCommand = Get-Command -Name 'python' -CommandType Application -ErrorAction SilentlyContinue
    if (-not $fallbackCommand) { throw 'Python not found. Supply -PythonPath or install Python on PATH.' }
    $researchPython = $fallbackCommand.Source
}
Push-Location $projectRoot
try {
    function Run-ResearchScript([string]$Name) {
        & $researchPython (Join-Path $PSScriptRoot $Name)
        if ($LASTEXITCODE -ne 0) { throw "Research step failed: $Name" }
    }
    if ($Mode -eq 'All') {
        # Uses the downloaded official documents and the frozen local price/factor snapshot.
        Run-ResearchScript 'parse_industry.py'
        Run-ResearchScript 'risk_metadata.py'
        Run-ResearchScript 'models.py'
        Run-ResearchScript 'experiment.py'
        Run-ResearchScript 'diagnose_models.py'
    }
    if ($Mode -in @('Verify', 'All')) {
        & $researchPython -m unittest discover -s $PSScriptRoot -p 'test_*.py'
        if ($LASTEXITCODE -ne 0) { throw 'Research unit checks failed' }
        Run-ResearchScript 'audit_accounts_ledger.py'
        Run-ResearchScript 'audit_metadata_reproduction.py'
    }
    Run-ResearchScript 'report.py'
} finally { Pop-Location }
