param([string]$InputDocx,[string]$OutputPdf)
$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($InputDocx, $false, $false)
    $document.Fields.Update() | Out-Null
    $document.Repaginate()
    $document.Fields.Update() | Out-Null
    $document.Save()
    $document.ExportAsFixedFormat($OutputPdf, 17)
    Write-Output ('pages=' + $document.ComputeStatistics(2))
} finally {
    if ($null -ne $document) { $document.Close(0); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($null -ne $word) { $word.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
}
