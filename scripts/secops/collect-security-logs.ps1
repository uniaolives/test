# scripts/secops/collect-security-logs.ps1
<#
.SYNOPSIS
    Coleta logs de segurança dos agentes ARKHE(N) para análise no SIEM.
.DESCRIPTION
    Este script integra-se ao Omega Ledger e coleta métricas de entropia
    para detecção proativa de ameaças.
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$AgentName = "*",

    [Parameter(Mandatory=$false)]
    [int]$HoursBack = 24
)

$LogPath = "C:\arkhe\logs\security"
$DateFilter = (Get-Date).AddHours(-$HoursBack)

Write-Host "🔐 Iniciando coleta de logs de segurança para agente: $AgentName" -ForegroundColor Cyan

# Coleta eventos de handover do sistema
Get-ChildItem -Path $LogPath -Filter "handover_*.log" | ForEach-Object {
    $logFile = $_.FullName
    Get-Content $logFile | Where-Object {
        $_ -match $AgentName -and [datetime]$_ -gt $DateFilter
    } | ForEach-Object {
        # Converte para formato estruturado
        $logEntry = $_ | ConvertFrom-Json

        # Calcula score de risco baseado em entropia
        if ($logEntry.entropy -gt 0.7) {
            $riskScore = "CRÍTICO"
            Write-Warning "Handover com entropia elevada detectado: $($logEntry.handover_id)"
        } else {
            $riskScore = "NORMAL"
        }

        # Saída formatada para ingestão no SIEM
        [PSCustomObject]@{
            Timestamp = $logEntry.timestamp
            AgentFrom = $logEntry.from_agent
            AgentTo = $logEntry.to_agent
            Entropy = $logEntry.entropy
            RiskLevel = $riskScore
            DataSizeKB = $logEntry.data_size / 1KB
        }
    }
} | Export-Csv -Path "C:\arkhe\exports\security_alerts_$(Get-Date -Format 'yyyyMMdd').csv" -NoTypeInformation

Write-Host "✅ Coleta concluída. Relatório exportado." -ForegroundColor Green
