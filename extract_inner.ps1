$sevenZip = "C:\Program Files\7-Zip\7z.exe"
$zipFiles = Get-ChildItem -Path "data\raw\huflit_logs" -Filter "*.zip" -Recurse

Write-Host "Found $($zipFiles.Count) zip files to extract..."

foreach ($zip in $zipFiles) {
    $outDir = $zip.DirectoryName
    Write-Host "Extracting $($zip.Name)..."
    & $sevenZip x $zip.FullName "-o$outDir" -y | Out-Null
}

Write-Host "All inner zip files extracted successfully!"
