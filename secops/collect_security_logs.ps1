# secops/collect_security_logs.ps1
param(
    [string]$NodeId = "arkhe-node-001",
    [int]$HoursBack = 24
)

# Mocked event collection for systems without Windows Event Logs
$startTime = (Get-Date).AddHours(-$HoursBack)
$events = @(
    @{ Id = 4624; TimeCreated = (Get-Date); Account = "System"; IP = "127.0.0.1" }
    @{ Id = 4625; TimeCreated = (Get-Date); Account = "Admin"; IP = "192.168.1.100" }
)

$handovers = @()
foreach ($ev in $events) {
    # Convert Windows event to Arkhe handover
    $handover = @{
        id = [guid]::NewGuid().ToString()
        emitter = $NodeId
        receiver = "secops-collector"
        type = 1   # excitatory (security information)
        timestamp_physical = [long]($ev.TimeCreated - (Get-Date "1970-01-01")).TotalSeconds
        entropy_cost = 0.01  # fixed cost, can be refined
        payload = @{
            event_id = $ev.Id
            account = $ev.Account
            ip = $ev.IP
        } | ConvertTo-Json
    }
    $handovers += $handover
}

# Send to central collector via REST (mocked)
$body = $handovers | ConvertTo-Json
# Invoke-RestMethod -Uri "http://arkhe-secops:8080/api/handovers" -Method Post -Body $body -ContentType "application/json"

Write-Host "Enviados $($handovers.Count) handovers de segurança."
foreach ($h in $handovers) {
    Write-Host "📦 Handover ID: $($h.id) | Emitter: $($h.emitter) | Payload: $($h.payload)"
}
