# Edit this target folder before running.
# Examples:
#   tactile/rawdata/batsh3/200ml
#   thermal/raw_bin
#   powershell -ExecutionPolicy Bypass -File Workspace\C_model_training\move_records_bin.ps1

#$TargetRelativeDir = "tactile/rawdata/batsh3"
$TargetRelativeDir = "thermal/raw_bin"
# Set to $true if .bin files may be inside subfolders under Vet6USB_pyqt_records.
$Recurse = $false

# Set to $true to preview actions without moving files.
$DryRun = $false

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$SourceDir = Join-Path $RepoRoot "Vet6USB_pyqt\Vet6USB_pyqt_records"
$TargetRoot = $ScriptDir
$TargetDir = Join-Path $TargetRoot $TargetRelativeDir

if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
    throw "Source folder does not exist: $SourceDir"
}

$resolvedTargetRoot = (Resolve-Path -LiteralPath $TargetRoot).Path
$targetFullPath = [System.IO.Path]::GetFullPath($TargetDir)
if (-not $targetFullPath.StartsWith($resolvedTargetRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Target must stay under $resolvedTargetRoot. Current target: $targetFullPath"
}

if (-not $DryRun -and -not (Test-Path -LiteralPath $targetFullPath -PathType Container)) {
    New-Item -ItemType Directory -Path $targetFullPath | Out-Null
}

$searchOption = if ($Recurse) { "AllDirectories" } else { "TopDirectoryOnly" }
$binFiles = [System.IO.Directory]::EnumerateFiles($SourceDir, "*.bin", $searchOption) |
    Sort-Object

$movedCount = 0
foreach ($file in $binFiles) {
    $fileName = [System.IO.Path]::GetFileName($file)
    $destination = Join-Path $targetFullPath $fileName

    if (Test-Path -LiteralPath $destination) {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($fileName)
        $ext = [System.IO.Path]::GetExtension($fileName)
        $i = 1
        do {
            $destination = Join-Path $targetFullPath ("{0}_{1}{2}" -f $stem, $i, $ext)
            $i += 1
        } while (Test-Path -LiteralPath $destination)
    }

    if ($DryRun) {
        Write-Host "[DRY RUN] $file -> $destination"
    } else {
        Move-Item -LiteralPath $file -Destination $destination
        Write-Host "Moved: $fileName -> $TargetRelativeDir"
    }
    $movedCount += 1
}

Write-Host "Done. Bin files processed: $movedCount"
Write-Host "Source: $SourceDir"
Write-Host "Target: $targetFullPath"
