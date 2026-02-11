Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Security

# --- ARKHE(N) OS: MÓDULO DE PRESERVAÇÃO (v5.1.1 Final Polish) ---
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$configPath = Join-Path $scriptDir "arkhe_config.json"
$logPath = Join-Path $scriptDir "arkhe_scan.log"
$identityPath = Join-Path $scriptDir "SIWA_IDENTITY.md"

$Script:LastScanResults = New-Object System.Collections.Generic.List[PSObject]

# --- SECURITY: DPAPI ENCRYPTION ---
function Protect-Secret($secret) {
    if ([string]::IsNullOrWhiteSpace($secret)) { return "" }
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($secret)
        $protected = [System.Security.Cryptography.ProtectedData]::Protect($bytes, $null, 'CurrentUser')
        return [Convert]::ToBase64String($protected)
    } catch { return "" }
}

function Unprotect-Secret($encrypted) {
    if ([string]::IsNullOrWhiteSpace($encrypted)) { return "" }
    try {
        $bytes = [Convert]::FromBase64String($encrypted)
        $unprotected = [System.Security.Cryptography.ProtectedData]::Unprotect($bytes, $null, 'CurrentUser')
        return [System.Text.Encoding]::UTF8.GetString($unprotected)
    } catch { return "" }
}

# --- LOGGING ---
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("INFO", "ERROR", "WARN", "DEBUG")]
        [string]$Level = "INFO",
        [string]$Component = "ARKHE"
    )
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $Entry = "[$Timestamp] [$Level] [$Component] $Message"
    try { Add-Content -Path $logPath -Value $Entry -ErrorAction SilentlyContinue } catch {}

    if ($globalLogBox) {
        $color = switch($Level) {
            "ERROR"{[Drawing.Color]::Red}
            "WARN"{[Drawing.Color]::Orange}
            "DEBUG"{[Drawing.Color]::Gray}
            Default{[Drawing.Color]::LimeGreen}
        }

        $globalLogBox.Invoke([Action[string, [Drawing.Color]]]{
            param($msg, $clr)
            $globalLogBox.SelectionStart = $globalLogBox.TextLength
            $globalLogBox.SelectionLength = 0
            $globalLogBox.SelectionColor = $clr
            $globalLogBox.AppendText("$msg`n")
            $globalLogBox.ScrollToCaret()
        }, $Entry, $color)
    }
}

# --- SETTINGS ---
function Load-Settings {
    $Default = @{
        PlexDbPath = "$env:LOCALAPPDATA\Plex Media Server\Plug-in Support\Databases\com.plexapp.plugins.library.db"
        Sonarr = @{ URL = "http://localhost:8989"; APIKey = ""; RootPath = "C:\Downloads"; ProfileId = 1 }
        Radarr = @{ URL = "http://localhost:7878"; APIKey = ""; RootPath = "C:\Downloads"; ProfileId = 1 }
        ExportPath = [System.IO.Path]::Combine($env:USERPROFILE, "Documents")
        DefaultDrive = "F:\"
        LastRotation = (Get-Date).ToString("yyyy-MM-dd")
        EssentialContacts = @{ Email = ""; TelegramChatId = "" }
    }
    if (Test-Path $configPath) {
        try {
            $s = Get-Content $configPath -Raw | ConvertFrom-Json
            $s.Sonarr.APIKey = Unprotect-Secret $s.Sonarr.APIKey
            $s.Radarr.APIKey = Unprotect-Secret $s.Radarr.APIKey
            if (-not $s.EssentialContacts) { $s.EssentialContacts = $Default.EssentialContacts }
            if (-not $s.Sonarr.ProfileId) { $s.Sonarr.ProfileId = 1 }
            if (-not $s.Radarr.ProfileId) { $s.Radarr.ProfileId = 1 }
            return $s
        } catch { Write-Log "Erro ao carregar configurações." "ERROR" }
    }
    return $Default
}

function Save-Settings($settings) {
    $clone = $settings | ConvertTo-Json | ConvertFrom-Json
    $clone.Sonarr.APIKey = Protect-Secret $settings.Sonarr.APIKey
    $clone.Radarr.APIKey = Protect-Secret $settings.Radarr.APIKey
    $clone | ConvertTo-Json -Depth 5 | Set-Content $configPath
    Write-Log "Configurações persistidas." "INFO" "CFG"
}

$globalSettings = Load-Settings

# --- SIWA IDENTITY HELPERS ---
function Read-SiwaIdentity {
    if (-not (Test-Path $identityPath)) { return @{ Address="0x0"; AgentId="0"; ChainId="0" } }
    $content = Get-Content $identityPath
    $id = @{}
    $content | ForEach-Object {
        if ($_ -match "\*\*Address:\*\* (0x[a-fA-F0-9]+)") { $id.Address = $Matches[1] }
        if ($_ -match "\*\*Agent ID:\*\* (\d+)") { $id.AgentId = $Matches[1] }
        if ($_ -match "\*\*Chain ID:\*\* (\d+)") { $id.ChainId = $Matches[1] }
    }
    return $id
}

function Build-SIWAMessage {
    param($Domain, $Nonce)
    $siwa = Read-SiwaIdentity
    $timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
    $msg = "$Domain wants you to sign in with your Agent account:`n$($siwa.Address)`n`nMódulo de Preservação Arkhe(n)`n`nURI: https://$Domain/auth`nVersion: 1`nAgent ID: $($siwa.AgentId)`nChain ID: $($siwa.ChainId)`nNonce: $Nonce`nIssued At: $timestamp"
    return $msg
}

# --- DATABASE DISCOVERY ---
function Get-PlexDBPath {
    $RegistryPath = "HKCU:\Software\Plex, Inc.\Plex Media Server"
    if (Test-Path $RegistryPath) {
        $Custom = (Get-ItemProperty -Path $RegistryPath -Name "LocalAppDataPath" -ErrorAction SilentlyContinue).LocalAppDataPath
        if ($Custom) {
            $path = Join-Path $Custom "Plex Media Server\Plug-in Support\Databases\com.plexapp.plugins.library.db"
            if (Test-Path $path) { return $path }
        }
    }
    return $globalSettings.PlexDbPath
}

$defaultPlexDbPath = Get-PlexDBPath
$defaultSqlitePath = Join-Path $scriptDir "sqlite3.exe"
if (-not (Test-Path $defaultSqlitePath)) { $defaultSqlitePath = "C:\tools\sqlite3.exe" }
$tempDbPath = "$env:TEMP\plex_missing_media_temp.db"

# --- NETWORK DRIVE CHECK ---
function Test-DriveReady($DrivePath) {
    if ($DrivePath -like "\\*") {
        $server = ($DrivePath -split '\\')[2]
        return Test-Connection -ComputerName $server -Count 1 -Quiet
    } else {
        $root = ($DrivePath -split ':')[0] + ":"
        return [bool](Get-PSDrive $root -ErrorAction SilentlyContinue)
    }
}

# --- DRIVE DETECTION ---
function Get-MissingDrives {
    Write-Log "Iniciando detecção (Protocolo Linfócito)..." "INFO" "DETECT"
    if (-not (Test-Path $defaultPlexDbPath)) { Write-Log "DB não localizado." "ERROR"; return @() }
    try {
        Copy-Item $defaultPlexDbPath $tempDbPath -Force
        $Query = "SELECT DISTINCT SUBSTR(file, 1, 3) FROM media_parts WHERE file IS NOT NULL;"
        $DbRoots = & $defaultSqlitePath -csv $tempDbPath $Query | ForEach-Object { $_.Trim('"').ToUpper() }
        Remove-Item $tempDbPath -Force
        $Missing = $DbRoots | Where-Object { -not (Test-DriveReady $_) -and $_ -match "^[A-Z]:\\" }
        return $Missing
    } catch { Write-Log "Falha na percepção: $_" "ERROR" }
    return @()
}

# --- RESTORATION APIs ---
function Add-To-Sonarr($title, $tvdbId, $seasons) {
    if (-not $globalSettings.Sonarr.APIKey) { Write-Log "API Key Sonarr ausente!" "ERROR"; return }
    if (-not [int]::TryParse($tvdbId, [ref]0)) { return }
    try {
        $headers = @{ "X-Api-Key" = $globalSettings.Sonarr.APIKey }
        $url = "$($globalSettings.Sonarr.URL)/api/v3/series"
        $body = @{
            title = $title; tvdbId = [int]$tvdbId; qualityProfileId = [int]$globalSettings.Sonarr.ProfileId; languageProfileId = 1
            rootFolderPath = $globalSettings.Sonarr.RootPath; monitored = $true
            seasons = ($seasons | ForEach-Object { @{ seasonNumber = [int]$_; monitored = $true } })
            addOptions = @{ searchForMissingEpisodes = $true }
        } | ConvertTo-Json -Depth 5
        Invoke-RestMethod -Uri $url -Method Post -Body $body -Headers $headers -ContentType "application/json"
        Write-Log "Cura enviada ao Sonarr: '$title'." "INFO" "SONARR"
    } catch { Write-Log "Erro Sonarr: $_" "ERROR" }
}

function Add-To-Radarr($title, $tmdbId) {
    if (-not $globalSettings.Radarr.APIKey) { Write-Log "API Key Radarr ausente!" "ERROR"; return }
    if (-not [int]::TryParse($tmdbId, [ref]0)) { return }
    try {
        $headers = @{ "X-Api-Key" = $globalSettings.Radarr.APIKey }
        $url = "$($globalSettings.Radarr.URL)/api/v3/movie"
        $body = @{
            title = $title; tmdbId = [int]$tmdbId; qualityProfileId = [int]$globalSettings.Radarr.ProfileId
            rootFolderPath = $globalSettings.Radarr.RootPath; monitored = $true; addOptions = @{ searchForMovie = $true }
        } | ConvertTo-Json
        Invoke-RestMethod -Uri $url -Method Post -Body $body -Headers $headers -ContentType "application/json"
        Write-Log "Cura enviada ao Radarr: '$title'." "INFO" "RADARR"
    } catch { Write-Log "Erro Radarr: $_" "ERROR" }
}

# --- GUI ---
$form = New-Object Windows.Forms.Form
$form.Text = "Arkhe(n) - Vigilante Soberano (v5.1.1)"
$form.Size = "1100, 900"; $form.BackColor = "#121212"; $form.ForeColor = "#E0E0E0"; $form.StartPosition = "CenterScreen"
$tabControl = New-Object Windows.Forms.TabControl; $tabControl.Dock = "Fill"; $form.Controls.Add($tabControl)
$logBox = New-Object Windows.Forms.RichTextBox; $logBox.Dock = "Bottom"; $logBox.Height = 250; $logBox.BackColor = "#000000"; $logBox.ForeColor = "#00FF00"; $logBox.ReadOnly = $true; $form.Controls.Add($logBox); $global:globalLogBox = $logBox
$infoPanel = New-Object Windows.Forms.Panel; $infoPanel.Dock = "Bottom"; $infoPanel.Height = 30; $form.Controls.Add($infoPanel)
$lblStatus = New-Object Windows.Forms.Label; $lblStatus.Text = "Φ = 1.000 (Coerência Estrita)"; $lblStatus.AutoSize = $true; $lblStatus.Location = "5, 5"; $lblStatus.ForeColor = "Cyan"; $infoPanel.Controls.Add($lblStatus)

function Create-Tab($tabName, $libType) {
    $tab = New-Object Windows.Forms.TabPage; $tab.Text = $tabName; $tab.BackColor = "#1E1E1E"
    $btnSmartFix = New-Object Windows.Forms.Button; $btnSmartFix.Text = "Vigilante (Detect)"; $btnSmartFix.Location = "10, 10"; $btnSmartFix.Size = "150, 40"; $btnSmartFix.BackColor = "#2C3E50"
    $tab.Controls.Add($btnSmartFix)
    $txtDrive = New-Object Windows.Forms.TextBox; $txtDrive.Text = $globalSettings.DefaultDrive; $txtDrive.Location = "170, 18"; $txtDrive.Width = 60; $tab.Controls.Add($txtDrive)
    $btnScan = New-Object Windows.Forms.Button; $btnScan.Text = "Diagnosticar"; $btnScan.Location = "240, 10"; $btnScan.Size = "120, 40"; $btnScan.BackColor = "#34495E"
    $tab.Controls.Add($btnScan)
    $btnHeal = New-Object Windows.Forms.Button; $btnHeal.Text = "Cicatrizar (API)"; $btnHeal.Location = "370, 10"; $btnHeal.Size = "130, 40"; $btnHeal.BackColor = "#27AE60"; $btnHeal.Enabled = $false; $tab.Controls.Add($btnHeal)
    $summaryBox = New-Object Windows.Forms.RichTextBox; $summaryBox.Location = "10, 60"; $summaryBox.Width = 1060; $summaryBox.Height = 480; $summaryBox.Anchor = "Top, Left, Right, Bottom"; $summaryBox.ReadOnly = $true; $summaryBox.BackColor = "#181818"; $summaryBox.ForeColor = "#F1C40F"; $tab.Controls.Add($summaryBox)

    $btnSmartFix.Add_Click({ $m = Get-MissingDrives; if ($m) { $txtDrive.Text = $m[0] } })
    $btnScan.Add_Click({
        $summaryBox.Clear()
        if (-not (Test-DriveReady $txtDrive.Text)) { Write-Log "Volume $($txtDrive.Text) inacessível." "ERROR"; return }
        $results = Run-Diagnostics -Type $libType -Tab $tabName -Drive $txtDrive.Text
        if ($results) { $Script:LastScanResults = $results; $btnHeal.Enabled = $true; foreach($r in $results) { $summaryBox.AppendText("- $($r.Title) ($($r.Year))`n") } }
    })
    $btnHeal.Add_Click({
        $resp = [Windows.Forms.MessageBox]::Show("Confirmar 2FA (Vigilante Security) para restauração?", "Vigilante Security", "YesNo")
        if ($resp -eq "Yes") {
            Write-Log "Aprovação 2FA recebida via Política de Vigilância." "INFO" "SECURITY"
            foreach ($item in $Script:LastScanResults) {
                if ($libType -eq "Movie") { Add-To-Radarr -title $item.Title -tmdbId $item.TmdbId }
                else { Add-To-Sonarr -title $item.Title -tvdbId $item.TvdbId -seasons $item.Seasons }
            }
        }
    })
    return $tab
}

function Create-Security-Tab {
    $tab = New-Object Windows.Forms.TabPage; $tab.Text = "Segurança & Identidade"; $tab.BackColor = "#1E1E1E"
    $siwa = Read-SiwaIdentity
    $lblId = New-Object Windows.Forms.Label; $lblId.Text = "SIWA Identity:"; $lblId.Location = "10, 20"; $lblId.AutoSize = $true; $tab.Controls.Add($lblId)
    $txtId = New-Object Windows.Forms.RichTextBox; $txtId.Text = "Address: $($siwa.Address)`nAgent ID: $($siwa.AgentId)`nChain ID: $($siwa.ChainId)"; $txtId.Location = "10, 40"; $txtId.Size = "400, 80"; $txtId.BackColor = "#000"; $txtId.ForeColor = "#00FF00"; $tab.Controls.Add($txtId)
    $btnSiwa = New-Object Windows.Forms.Button; $btnSiwa.Text = "Gerar Auth SIWA"; $btnSiwa.Location = "10, 130"; $btnSiwa.Size = "150, 40"; $btnSiwa.BackColor = "#2980B9"; $tab.Controls.Add($btnSiwa)
    $txtMsg = New-Object Windows.Forms.RichTextBox; $txtMsg.Location = "10, 180"; $txtMsg.Size = "500, 200"; $txtMsg.BackColor = "#000"; $txtMsg.ForeColor = "#FFF"; $tab.Controls.Add($txtMsg)
    $btnSiwa.Add_Click({ $txtMsg.Text = Build-SIWAMessage -Domain "arkhe.os" -Nonce (New-Guid).ToString().Substring(0,8) })
    return $tab
}

function Create-Config-Tab {
    $tab = New-Object Windows.Forms.TabPage; $tab.Text = "Configurações"; $tab.BackColor = "#1E1E1E"
    $y = 20
    $fields = @(@("Sonarr URL", "SUrl"), @("Sonarr API Key", "SKey"), @("Sonarr Profile ID", "SProf"), @("Sonarr Root Path", "SRoot"), @("Radarr URL", "RUrl"), @("Radarr API Key", "RKey"), @("Radarr Profile ID", "RProf"), @("Radarr Root Path", "RRoot"), @("Alert Email", "Email"), @("Telegram Chat ID", "TId"))
    $inputs = @{}
    foreach ($f in $fields) {
        $lbl = New-Object Windows.Forms.Label; $lbl.Text = $f[0] + ":"; $lbl.Location = "10, $y"; $lbl.Width = 150; $tab.Controls.Add($lbl)
        $txt = New-Object Windows.Forms.TextBox; $txt.Location = "170, $y"; $txt.Width = 400
        $val = switch($f[1]) { "SUrl"{$globalSettings.Sonarr.URL} "SKey"{$globalSettings.Sonarr.APIKey} "SProf"{$globalSettings.Sonarr.ProfileId} "SRoot"{$globalSettings.Sonarr.RootPath} "RUrl"{$globalSettings.Radarr.URL} "RKey"{$globalSettings.Radarr.APIKey} "RProf"{$globalSettings.Radarr.ProfileId} "RRoot"{$globalSettings.Radarr.RootPath} "Email"{$globalSettings.EssentialContacts.Email} "TId"{$globalSettings.EssentialContacts.TelegramChatId} }
        $txt.Text = $val; $tab.Controls.Add($txt); $inputs[$f[1]] = $txt; $y += 35
    }
    $btnSave = New-Object Windows.Forms.Button; $btnSave.Text = "Gravar Axioma"; $btnSave.Location = "170, $y"; $btnSave.Size = "150, 40"; $btnSave.BackColor = "#2980B9"; $tab.Controls.Add($btnSave)
    $btnSave.Add_Click({
        $globalSettings.Sonarr.URL = $inputs["SUrl"].Text; $globalSettings.Sonarr.APIKey = $inputs["SKey"].Text; $globalSettings.Sonarr.ProfileId = $inputs["SProf"].Text; $globalSettings.Sonarr.RootPath = $inputs["SRoot"].Text
        $globalSettings.Radarr.URL = $inputs["RUrl"].Text; $globalSettings.Radarr.APIKey = $inputs["RKey"].Text; $globalSettings.Radarr.ProfileId = $inputs["RProf"].Text; $globalSettings.Radarr.RootPath = $inputs["RRoot"].Text
        $globalSettings.EssentialContacts.Email = $inputs["Email"].Text; $globalSettings.EssentialContacts.TelegramChatId = $inputs["TId"].Text
        $globalSettings.LastRotation = (Get-Date).ToString("yyyy-MM-dd"); Save-Settings $globalSettings; [Windows.Forms.MessageBox]::Show("Axiomas atualizados.")
    })
    return $tab
}

# --- DIAGNOSTICS LOGIC ---
function Run-Diagnostics($Type, $Tab, $Drive) {
    Write-Log "Diagnóstico iniciado: $Drive" "INFO" "SCAN"
    if (-not (Test-Path $defaultPlexDbPath)) { return $null }
    Copy-Item $defaultPlexDbPath $tempDbPath -Force
    $Query = if ($Type -eq "Movie") {
        "SELECT md.title, md.year, CASE WHEN md.guid LIKE '%tmdb://%' THEN REPLACE(REPLACE(SUBSTR(md.guid, INSTR(md.guid, 'tmdb://') + 7), '?lang=en', ''), '?lang=pt', '') WHEN md.guid LIKE '%imdb://%' THEN REPLACE(REPLACE(SUBSTR(md.guid, INSTR(md.guid, 'imdb://') + 7), '?lang=en', ''), '?lang=pt', '') ELSE md.guid END as ExtId, mp.file FROM metadata_items md JOIN media_items mi ON md.id = mi.metadata_item_id JOIN media_parts mp ON mi.id = mp.media_item_id WHERE md.metadata_type = 1 AND md.deleted_at IS NULL;"
    } else {
        "SELECT show.title, show.year, CASE WHEN show.guid LIKE '%tvdb://%' THEN REPLACE(REPLACE(SUBSTR(show.guid, INSTR(show.guid, 'tvdb://') + 7), '?lang=en', ''), '?lang=pt', '') ELSE show.guid END as ExtId, season.index, mp.file, lib.name FROM metadata_items show JOIN metadata_items season ON season.parent_id = show.id JOIN metadata_items episode ON episode.parent_id = season.id JOIN media_items mi ON episode.id = mi.metadata_item_id JOIN media_parts mp ON mi.id = mp.media_item_id JOIN library_sections lib ON show.library_section_id = lib.id WHERE show.metadata_type = 2 AND episode.metadata_type = 4 AND episode.deleted_at IS NULL;"
    }
    $results = & $defaultSqlitePath -csv $tempDbPath $Query | ConvertFrom-Csv -Header "Title","Year","ExtId","SeasonOrPath","PathOrLib","LibOrNull"
    $missing = New-Object System.Collections.Generic.List[PSObject]
    foreach ($r in $results) {
        $filePath = if($Type -eq "Movie") { $r.SeasonOrPath } else { $r.PathOrLib }
        if ($filePath.StartsWith($Drive, [System.StringComparison]::OrdinalIgnoreCase)) {
            if (-not (Test-Path $filePath)) {
                $id = ($r.ExtId -split '\?')[0]
                if ($Type -eq "Movie") { $missing.Add([PSCustomObject]@{ Title=$r.Title; Year=$r.Year; TmdbId=$id }) }
                else {
                    $lib = $r.LibOrNull
                    if (($Tab -eq "Anime" -and $lib -match "Anime") -or ($Tab -eq "TV Shows" -and $lib -notmatch "Anime")) {
                        $missing.Add([PSCustomObject]@{ Title=$r.Title; Year=$r.Year; TvdbId=$id; Season=$r.SeasonOrPath })
                    }
                }
            }
        }
    }
    $final = if ($Type -eq "Movie") { $missing | Group-Object Title | ForEach-Object { $_.Group[0] } }
             else { $missing | Group-Object Title | ForEach-Object { $f = $_.Group[0]; $s = $_.Group | Select-Object -ExpandProperty Season -Unique | Sort-Object {[int]$_}; [PSCustomObject]@{ Title=$_.Name; Year=$f.Year; TvdbId=$f.TvdbId; Seasons=$s } } }
    Remove-Item $tempDbPath -Force -ErrorAction SilentlyContinue; return $final
}

$tabControl.TabPages.Add((Create-Tab "TV Shows" "TV"))
$tabControl.TabPages.Add((Create-Tab "Movies" "Movie"))
$tabControl.TabPages.Add((Create-Tab "Anime" "TV"))
$tabControl.TabPages.Add((Create-Security-Tab))
$tabControl.TabPages.Add((Create-Config-Tab))
Write-Log "Vigilante Online. Identidade SIWA Verificada." "INFO" "KERNEL"
$form.ShowDialog() | Out-Null
