# AGI Service Integrity Check Script
# Validates binary signatures and environment consistency

$InstallPath = "C:\Program Files\AGI"
$ServiceName = "AGI_Service"

Write-Host "Starting AGI Integrity Check..."

# 1. Verify Service Binary
$SvcPath = "$InstallPath\agisvc.exe"
if (Test-Path $SvcPath) {
    $Signature = Get-AuthenticodeSignature $SvcPath
    if ($Signature.Status -ne "Valid") {
        Write-Error "CRITICAL: agisvc.exe signature is invalid!"
        # In production, we would trigger an emergency halt
    } else {
        Write-Host "Binary signature verified."
    }
} else {
    Write-Error "CRITICAL: agisvc.exe not found!"
}

# 2. Check Service Status
$Service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($Service.Status -ne "Running") {
    Write-Warning "AGI_Service is not running. Status: $($Service.Status)"
}

# 3. Verify Model Integrity
$ModelPath = "$InstallPath\models\core_model.onnx"
if (Test-Path $ModelPath) {
    $Hash = Get-FileHash $ModelPath -Algorithm SHA256
    Write-Host "Model Hash: $($Hash.Hash)"
}

Write-Host "Integrity Check Complete."
