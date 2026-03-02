# Windows 11 AGI Service Build and Installation Script
# Requires: Rust, Windows SDK, Visual Studio Build Tools

param(
    [switch]$Release = $true,
    [switch]$Install = $false,
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"

# Configuration
$ServiceName = "AGI_Service"
$ServiceDisplayName = "Ontology AGI Runtime v0.7.0"
$InstallPath = "C:\Program Files\AGI"
$LogPath = "C:\ProgramData\AGI\logs"
$ModelPath = "$InstallPath\models"

# Colors for output
$Green = [ConsoleColor]::Green
$Yellow = [ConsoleColor]::Yellow
$Red = [ConsoleColor]::Red
$Cyan = [ConsoleColor]::Cyan

function Write-Color {
    param($Color, $Text)
    $prev = [Console]::ForegroundColor
    [Console]::ForegroundColor = $Color
    Write-Host $Text
    [Console]::ForegroundColor = $prev
}

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check for admin privileges
if (-not (Test-Admin)) {
    Write-Color $Red "❌ This script requires Administrator privileges"
    exit 1
}

# Clean previous build
if ($Clean) {
    Write-Color $Yellow "🧹 Cleaning previous builds..."
    cargo clean
    Remove-Item -Path $InstallPath -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $LogPath -Recurse -Force -ErrorAction SilentlyContinue
}

# Create directories
Write-Color $Cyan "📁 Creating directories..."
New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
New-Item -ItemType Directory -Path $LogPath -Force | Out-Null
New-Item -ItemType Directory -Path $ModelPath -Force | Out-Null

# Build Rust project
Write-Color $Cyan "🔨 Building AGI Service..."
$BuildArgs = @("build", "--package", "agi-service")
if ($Release) {
    $BuildArgs += "--release"
}

$env:RUSTFLAGS = "-C target-feature=+crt-static"
cargo @BuildArgs

if (-not $?) {
    Write-Color $Red "❌ Build failed"
    exit 1
}

# Copy binaries
Write-Color $Cyan "📦 Installing binaries..."
$BinPath = if ($Release) { "target\release\agi-service.exe" } else { "target\debug\agi-service.exe" }
Copy-Item -Path $BinPath -Destination "$InstallPath\agisvc.exe" -Force

# Copy models (placeholder)
Write-Color $Cyan "🤖 Installing model placeholder..."
$ModelContent = @'
// Placeholder for ONNX model
// In production, download from secure source
'@
Set-Content -Path "$ModelPath\core_model.onnx" -Value $ModelContent -Encoding UTF8

# Create configuration
Write-Color $Cyan "⚙️  Creating configuration..."
$Config = @{
    service = @{
        name = $ServiceName
        display_name = $ServiceDisplayName
        description = "AGI Runtime with Geometric Consensus and Quantum Security"
    }
    security = @{
        apk_hash = "A1B2C3D4E5F678901234567890ABCDEF0123456789ABCDEF0123456789ABCDEF"
        quantum_seed = "7f3b2a1c9e8d5f4a2b3c1d0e9f8a7b6c5d4e3f2a1b0c9e8d7f6a5b4c3d2e1f0a9"
        lyapunov_threshold = 0.05
        coherence_threshold = 0.8
    }
    paths = @{
        install = $InstallPath
        logs = $LogPath
        models = $ModelPath
        substrate = "C:\ProgramData\AGI\substrate.db"
    }
}

$ConfigJson = $Config | ConvertTo-Json -Depth 10
Set-Content -Path "$InstallPath\config.json" -Value $ConfigJson -Encoding UTF8

# Register Windows Service
if ($Install) {
    Write-Color $Cyan "🛠️  Registering Windows Service..."

    # Stop existing service if running
    $Service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($Service) {
        Write-Color $Yellow "⚠️  Stopping existing service..."
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        sc.exe delete $ServiceName
        Start-Sleep -Seconds 2
    }

    # Create new service
    $Result = sc.exe create $ServiceName `
        binPath= "`"$InstallPath\agisvc.exe`" --service" `
        DisplayName= $ServiceDisplayName `
        start= delayed-auto `
        type= own `
        error= normal `
        tag= no

    if ($Result -ne 0) {
        Write-Color $Red "❌ Failed to create service: $Result"
        exit 1
    }

    # Configure service recovery
    sc.exe failure $ServiceName reset= 86400 actions= restart/60000/restart/60000/restart/300000

    # Set service description
    $Description = "Provides AGI capabilities with geometric consensus, quantum security, and paradigm protection."
    Set-Service -Name $ServiceName -Description $Description

    # Configure service SID type
    sc.exe sidtype $ServiceName unrestricted

    # Grant necessary privileges
    Write-Color $Cyan "🔐 Configuring service privileges..."

    # This requires security policy editing
    # In production, would use secedit.exe or Group Policy

    # Start the service
    Write-Color $Cyan "🚀 Starting service..."
    Start-Service -Name $ServiceName

    # Verify service is running
    Start-Sleep -Seconds 3
    $Service = Get-Service -Name $ServiceName
    if ($Service.Status -eq "Running") {
        Write-Color $Green "✅ Service started successfully!"
        Write-Color $Green "📊 Service Name: $ServiceName"
        Write-Color $Green "📈 Status: $($Service.Status)"
        Write-Color $Green "🎯 Startup Type: $($Service.StartType)"
        Write-Color $Green "📁 Logs: $LogPath"
        Write-Color $Green "🏠 Install Path: $InstallPath"
    } else {
        Write-Color $Red "❌ Service failed to start"
        Get-EventLog -LogName Application -Source $ServiceName -Newest 10 |
            Format-List TimeGenerated, EntryType, Message
        exit 1
    }
}

# Create firewall rules
Write-Color $Cyan "🔥 Configuring firewall..."
New-NetFirewallRule -DisplayName "AGI Service Audit Stream" `
    -Direction Inbound `
    -LocalPort 8081 `
    -Protocol TCP `
    -Action Allow `
    -Program "$InstallPath\agisvc.exe" `
    -ErrorAction SilentlyContinue | Out-Null

# Set up event log source
Write-Color $Cyan "📝 Configuring event log..."
New-EventLog -LogName Application -Source $ServiceName -ErrorAction SilentlyContinue

# Create scheduled task for integrity checks
Write-Color $Cyan "⏰ Creating integrity check task..."
$TaskAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File `"$InstallPath\integrity-check.ps1`""

$TaskTrigger = New-ScheduledTaskTrigger -Daily -At 3am
$TaskPrincipal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$TaskSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName "AGI Integrity Check" `
    -Action $TaskAction `
    -Trigger $TaskTrigger `
    -Principal $TaskPrincipal `
    -Settings $TaskSettings `
    -Description "Daily integrity check for AGI Service" `
    -Force `
    -ErrorAction SilentlyContinue

Write-Color $Green "✨ AGI Service installation complete!"
Write-Color $Cyan "📋 Next steps:"
Write-Color $Cyan "   1. Configure quantum seed in $InstallPath\config.json"
Write-Color $Cyan "   2. Replace placeholder model with actual ONNX model"
Write-Color $Cyan "   3. Test service with: Get-Service AGI_Service"
Write-Color $Cyan "   4. View logs with: Get-EventLog -LogName Application -Source AGI_Service"

# Create quick test script
$TestScript = @'
# Test AGI Service
$Service = Get-Service -Name "AGI_Service"
Write-Host "Service Status: $($Service.Status)"
Write-Host "Startup Type: $($Service.StartType)"

# Check recent events
Get-EventLog -LogName Application -Source "AGI_Service" -Newest 5 |
    Format-Table TimeGenerated, EntryType, Message -AutoSize

# Test TCP port
Test-NetConnection -ComputerName localhost -Port 8081
'@

Set-Content -Path "$InstallPath\test-service.ps1" -Value $TestScript -Encoding UTF8
