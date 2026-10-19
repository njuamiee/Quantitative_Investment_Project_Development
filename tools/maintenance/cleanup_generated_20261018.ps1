param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspacePath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspacePrefix = $workspacePath.TrimEnd('\') + '\'
$targets = [Collections.Generic.List[string]]::new()
function Add-Target([string]$relative) {
    $candidatePath = [IO.Path]::GetFullPath((Join-Path $workspacePath $relative))
    if (-not $candidatePath.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside workspace: $relative"
    }
    if (Test-Path -LiteralPath $candidatePath) { $targets.Add($candidatePath) }
}
# These folders only contain generated renderings, not authored documents.
Get-ChildItem -LiteralPath $workspacePath -Directory -Force |
    Where-Object Name -Like '.chart-data-*' | ForEach-Object { Add-Target $_.Name }
Get-ChildItem -LiteralPath (Join-Path $workspacePath 'training_runs') -Directory -Recurse |
    Where-Object { $_.Name -like 'docx_qa*' -or $_.Name -like 'pptx_qa*' -or $_.Name -eq 'document_draft' } |
    ForEach-Object { Add-Target ([IO.Path]::GetRelativePath($workspacePath, $_.FullName)) }
foreach ($relative in @(
    'training_runs/learning_20261013/complete_final_0',
    'training_runs/learning_20261013/complete_final_1',
    'training_runs/learning_20261013/complete_qa_0',
    'training_runs/learning_20261013/complete_qa_1',
    'training_runs/learning_20261013/qa_global',
    'training_runs/learning_20261013/qa_lesson1',
    '.ppt_build_20261016/final_slides', '.ppt_build_20261016/slides',
    '.ppt_build_20261016/candidate.pptx', '.ppt_build_20261016/candidate.pptx.inspect.ndjson',
    '.ppt_teacher_20261016/final_slides', '.ppt_teacher_20261016/slides',
    '.ppt_teacher_20261016/source_slides', '.ppt_teacher_20261016/candidate.pptx',
    '.ppt_teacher_20261016/candidate.pptx.inspect.ndjson',
    '.ppt_teacher_20261016/visual_revision/final_slides',
    '.ppt_teacher_20261016/visual_revision/slides',
    '.ppt_teacher_20261016/visual_revision/candidate.pptx',
    '.ppt_teacher_20261016/visual_revision/candidate.pptx.inspect.ndjson',
    '.ppt_teacher_20261016/visual_revision/with_notes/notes_only_candidate.pptx',
    '.ppt_teacher_20261016/visual_revision/with_notes/final_slides',
    '.ppt_teacher_20261016/visual_revision/with_notes/candidate.pptx',
    '.ppt_teacher_20261016/visual_revision/with_notes/candidate.pptx.inspect.ndjson'
)) { Add-Target $relative }
Get-ChildItem -LiteralPath (Join-Path $workspacePath 'tools') -Directory -Recurse |
    Where-Object { $_.Name -in @('__pycache__','.pytest_cache','.mypy_cache','.ruff_cache') } |
    ForEach-Object { Add-Target ([IO.Path]::GetRelativePath($workspacePath, $_.FullName)) }

$tracked = @(& git -C $workspacePath -c core.quotepath=false ls-files)
if ($LASTEXITCODE -ne 0) { throw 'Cannot check tracked files' }
$records = @()
$accepted = @()
foreach ($targetPath in ($targets | Sort-Object -Unique)) {
    if ($accepted | Where-Object { $targetPath.StartsWith($_ + '\', [StringComparison]::OrdinalIgnoreCase) }) { continue }
    $item = Get-Item -LiteralPath $targetPath -Force
    $relative = [IO.Path]::GetRelativePath($workspacePath, $targetPath).Replace('\','/')
    if ($tracked | Where-Object { $_ -eq $relative -or $_.StartsWith($relative + '/') }) {
        throw "Refusing to remove tracked artifact: $relative"
    }
    $allItems = @($item)
    if ($item.PSIsContainer) { $allItems += @(Get-ChildItem -LiteralPath $targetPath -Recurse -Force) }
    if ($allItems | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) {
        throw "Refusing recursive cleanup of a link or junction: $relative"
    }
    $files = @($allItems | Where-Object { -not $_.PSIsContainer })
    $records += [pscustomobject][ordered]@{path=$relative; files=$files.Count; bytes=[long](($files | Measure-Object Length -Sum).Sum)}
    $accepted += $targetPath
}
# Validate every target before making the first deletion. Never traverse node_modules junctions.
if ($Apply) {
    foreach ($targetPath in $accepted) { Remove-Item -LiteralPath $targetPath -Recurse -Force }
}
$result = [ordered]@{
    operation = $(if ($Apply) { 'deleted' } else { 'dry_run' })
    date = (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')
    target_count = $records.Count
    file_count = [int](($records | Measure-Object files -Sum).Sum)
    bytes = [long](($records | Measure-Object bytes -Sum).Sum)
    preserved = @('training_data', 'all experiment ledgers', 'authored PPT sources', 'node_modules junctions', 'virtual environments')
    targets = $records
}
if ($Apply) {
    $manifestPath = Join-Path $PSScriptRoot 'cleanup_20261018.json'
    if (Test-Path -LiteralPath $manifestPath) {
        $previous = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $result['targets'] = @($previous.targets) + @($records)
        $result['target_count'] += $previous.target_count
        $result['file_count'] += $previous.file_count
        $result['bytes'] += $previous.bytes
    }
    $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
}
$result | ConvertTo-Json -Depth 5
