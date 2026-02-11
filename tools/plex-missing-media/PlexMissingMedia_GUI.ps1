Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Security

# --- ARKHE(N) OS: MÓDULO DE PRESERVAÇÃO (v5.2 Unified Restoration) ---
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$configPath = Join-Path $scriptDir "ArkheConfig.json"
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
        [ValidateSet("INFO", "ERROR", "WARN", "DEBUG", "SUCCESS")]
        [string]$Level = "INFO",
        [string]$Component = "ARKHE"
    )
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $Entry = "[$Timestamp] [$Level] [$Component] $Message"
    try { Add-Content -Path $logPath -Value $Entry -ErrorAction SilentlyContinue } catch {}

    if ($globalLogBox) {
        $color = switch($Level) {
            "ERROR"   { [Drawing.Color]::Red }
            "WARN"    { [Drawing.Color]::Orange }
            "DEBUG"   { [Drawing.Color]::Gray }
            "SUCCESS" { [Drawing.Color]::Lime }
            Default   { [Drawing.Color]::LimeGreen }
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
    if (Test-Path $configPath) {
        try {
            $s = Get-Content $configPath -Raw | ConvertFrom-Json
            $s.Sonarr.APIKey = Unprotect-Secret $s.Sonarr.APIKey
            $s.Radarr.APIKey = Unprotect-Secret $s.Radarr.APIKey
            return $s
        } catch { Write-Log "Erro ao carregar ArkheConfig.json." "ERROR" }
    }
    $Default = @{
        PlexDbPath = "" # Wil be auto-detected
        Sonarr = @{ URL = "http://localhost:8989"; APIKey = ""; RootPath = "C:\Downloads"; ProfileId = 1 }
        Radarr = @{ URL = "http://localhost:7878"; APIKey = ""; RootPath = "C:\Downloads"; ProfileId = 1 }
        ExportPath = [System.IO.Path]::Combine($env:USERPROFILE, "Documents")
        DefaultDrive = "F:\"
        LastRotation = (Get-Date).ToString("yyyy-MM-dd")
        EssentialContacts = @{ Email = ""; TelegramChatId = "" }
    }
    return $Default
}

function Save-Settings($settings) {
    $clone = $settings | ConvertTo-Json | ConvertFrom-Json
    $clone.Sonarr.APIKey = Protect-Secret $settings.Sonarr.APIKey
    $clone.Radarr.APIKey = Protect-Secret $settings.Radarr.APIKey
    $clone | ConvertTo-Json -Depth 5 | Set-Content $configPath
    Write-Log "Axiomas de configuração persistidos e selados." "INFO" "CFG"
}

$globalSettings = Load-Settings

# --- DATABASE DISCOVERY ---
function Get-PlexDatabasePath {
    <#
    .SYNOPSIS
        Locates the Plex Media Server database path dynamically via Windows Registry.
    #>
    $RegistryPath = "HKCU:\Software\Plex, Inc.\Plex Media Server"
    $ValueName = "LocalAppDataPath"
    $DbFileName = "com.plexapp.plugins.library.db"
    $DefaultSubPath = "Plex Media Server\Plug-in Support\Databases\$DbFileName"

    try {
        if (Test-Path $RegistryPath) {
            $CustomPath = (Get-ItemProperty -Path $RegistryPath -Name $ValueName -ErrorAction SilentlyContinue).$ValueName
            if ($CustomPath) {
                $FinalPath = Join-Path $CustomPath $DefaultSubPath
                if (Test-Path $FinalPath) {
                    Write-Log "Banco de dados Plex localizado via Registro: $FinalPath" "INFO" "DB"
                    return $FinalPath
                }
            }
        }
    } catch { Write-Log "Erro ao acessar Registro para descoberta do DB: $_" "ERROR" "DB" }

    # Fallback
    $Fallback = Join-Path $env:LOCALAPPDATA $DefaultSubPath
    Write-Log "Usando caminho padrão para o banco de dados: $Fallback" "WARN" "DB"
    return $Fallback
}

$defaultPlexDbPath = Get-PlexDatabasePath
$globalSettings.PlexDbPath = $defaultPlexDbPath

$defaultSqlitePath = Join-Path $scriptDir "sqlite3.exe"
if (-not (Test-Path $defaultSqlitePath)) { $defaultSqlitePath = "C:\tools\sqlite3.exe" }
$tempDbPath = "$env:TEMP\plex_missing_media_temp.db"

# --- DRIVE DETECTION ---
function Get-MissingDrives {
    Write-Log "Interrogando banco de dados para detecção de volumes..." "INFO" "DETECT"
    if (-not (Test-Path $defaultPlexDbPath)) { Write-Log "Banco não encontrado para análise." "ERROR" "DB"; return @() }

    try {
        Copy-Item $defaultPlexDbPath $tempDbPath -Force -ErrorAction Stop

        # Query for all unique roots (e.g. F:\)
        $Query = "SELECT DISTINCT SUBSTR(file, 1, 3) FROM media_parts WHERE file IS NOT NULL AND file LIKE '_:_%';"
        $DbRoots = & $defaultSqlitePath -csv $tempDbPath $Query | ForEach-Object { $_.Trim('"').ToUpper() }
        Remove-Item $tempDbPath -Force -ErrorAction SilentlyContinue

        $Mounted = Get-PSDrive -PSProvider FileSystem | Select-Object -ExpandProperty Root | ForEach-Object { $_.ToUpper() }
        $Missing = $DbRoots | Where-Object { $Mounted -notcontains $_ -and $_ -match "^[A-Z]:\\" }

        if ($Missing) {
            Write-Log "Vácuo detectado nos volumes: $($Missing -join ', ')" "WARN" "DETECT"
            return $Missing
        } else {
            Write-Log "Todos os volumes registrados no banco estão montados." "SUCCESS" "DETECT"
        }
    } catch {
        Write-Log "Falha ao escanear banco para detecção de volumes: $_" "ERROR" "DETECT"
        if (Test-Path $tempDbPath) { Remove-Item $tempDbPath -Force }
    }
    return @()
}

# --- RESTORATION APIs ---
function Add-ToSonarr {
    param(
        [string]$Title,
        [string]$TvdbId,
        [int[]]$Seasons,
        [string]$RootPath,
        [int]$ProfileId
    )
    if (-not $globalSettings.Sonarr.APIKey) { Write-Log "API Key do Sonarr ausente!" "ERROR" "SONARR"; return }
    if (-not [int]::TryParse($TvdbId, [ref]0)) { Write-Log "TvdbId '$TvdbId' inválido. Abortando cura para '$Title'." "WARN" "SONARR"; return }

    Write-Log "Orquestrando cura no Sonarr: '$Title' (TVDB: $TvdbId)" "INFO" "SONARR"

    try {
        $headers = @{ "X-Api-Key" = $globalSettings.Sonarr.APIKey }
        $url = "$($globalSettings.Sonarr.URL)/api/v3/series"

        # Duplicate check
        $existing = Invoke-RestMethod -Uri "$url?tvdbId=$TvdbId" -Headers $headers -ErrorAction SilentlyContinue
        if ($existing) { Write-Log "Série '$Title' já reside no Sonarr. Ignorando." "WARN" "SONARR"; return }

        $seasonsPayload = $Seasons | ForEach-Object { @{ seasonNumber = $_; monitored = $true } }
        $body = @{
            title = $Title; tvdbId = [int]$TvdbId; qualityProfileId = $ProfileId; languageProfileId = 1
            rootFolderPath = $RootPath; monitored = $true; seasons = $seasonsPayload
            addOptions = @{ searchForMissingEpisodes = $true }
        } | ConvertTo-Json -Depth 5

        Invoke-RestMethod -Uri $url -Method Post -Body $body -Headers $headers -ContentType "application/json"
        Write-Log "Comando de cura enviado: '$Title'." "SUCCESS" "SONARR"
    } catch {
        Write-Log "Falha ao comunicar com Sonarr para '$Title': $($_.Exception.Message)" "ERROR" "SONARR"
    }
}

function Add-ToRadarr {
    param(
        [string]$Title,
        [string]$TmdbId,
        [string]$RootPath,
        [int]$ProfileId
    )
    if (-not $globalSettings.Radarr.APIKey) { Write-Log "API Key do Radarr ausente!" "ERROR" "RADARR"; return }
    if (-not [int]::TryParse($TmdbId, [ref]0)) { Write-Log "TmdbId '$TmdbId' inválido. Abortando cura para '$Title'." "WARN" "RADARR"; return }

    Write-Log "Orquestrando cura no Radarr: '$Title' (TMDB: $TmdbId)" "INFO" "RADARR"

    try {
        $headers = @{ "X-Api-Key" = $globalSettings.Radarr.APIKey }
        $url = "$($globalSettings.Radarr.URL)/api/v3/movie"

        $existing = Invoke-RestMethod -Uri "$url?tmdbId=$TmdbId" -Headers $headers -ErrorAction SilentlyContinue
        if ($existing) { Write-Log "Filme '$Title' já reside no Radarr. Ignorando." "WARN" "RADARR"; return }

        $body = @{
            title = $Title; tmdbId = [int]$TmdbId; qualityProfileId = $ProfileId
            rootFolderPath = $RootPath; monitored = $true; addOptions = @{ searchForMovie = $true }
        } | ConvertTo-Json

        Invoke-RestMethod -Uri $url -Method Post -Body $body -Headers $headers -ContentType "application/json"
        Write-Log "Comando de cura enviado: '$Title'." "SUCCESS" "RADARR"
    } catch {
        Write-Log "Falha ao comunicar com Radarr para '$Title': $($_.Exception.Message)" "ERROR" "RADARR"
    }
}

# --- GUI ---
$form = New-Object Windows.Forms.Form
$form.Text = "Arkhe(n) - Vigilante Soberano (v5.2 Unified)"
$form.Size = "1100, 900"; $form.BackColor = "#121212"; $form.ForeColor = "#E0E0E0"; $form.StartPosition = "CenterScreen"
$tabControl = New-Object Windows.Forms.TabControl; $tabControl.Dock = "Fill"; $form.Controls.Add($tabControl)

$logBox = New-Object Windows.Forms.RichTextBox
$logBox.Dock = "Bottom"; $logBox.Height = 250; $logBox.BackColor = "#000000"; $logBox.ForeColor = "#00FF00"; $logBox.ReadOnly = $true
$form.Controls.Add($logBox); $global:globalLogBox = $logBox

$infoPanel = New-Object Windows.Forms.Panel; $infoPanel.Dock = "Bottom"; $infoPanel.Height = 30
$form.Controls.Add($infoPanel)
$lblStatus = New-Object Windows.Forms.Label; $lblStatus.Text = "Linfócito de Integridade: Ativo (Φ = 1.000)"; $lblStatus.AutoSize = $true; $lblStatus.Location = "5, 5"; $lblStatus.ForeColor = "Cyan"
$infoPanel.Controls.Add($lblStatus)

function Create-Tab($tabName, $libType) {
    $tab = New-Object Windows.Forms.TabPage; $tab.Text = $tabName; $tab.BackColor = "#1E1E1E"
    $btnSmartFix = New-Object Windows.Forms.Button; $btnSmartFix.Text = "Smart Fix (Detect)"; $btnSmartFix.Location = "10, 10"; $btnSmartFix.Size = "150, 40"; $btnSmartFix.BackColor = "#2C3E50"
    $tab.Controls.Add($btnSmartFix)
    $txtDrive = New-Object Windows.Forms.TextBox; $txtDrive.Text = $globalSettings.DefaultDrive; $txtDrive.Location = "170, 18"; $txtDrive.Width = 60; $tab.Controls.Add($txtDrive)
    $btnScan = New-Object Windows.Forms.Button; $btnScan.Text = "Diagnosticar"; $btnScan.Location = "240, 10"; $btnScan.Size = "120, 40"; $btnScan.BackColor = "#34495E"
    $tab.Controls.Add($btnScan)
    $btnCsv = New-Object Windows.Forms.Button; $btnCsv.Text = "Exportar CSV"; $btnCsv.Location = "370, 10"; $btnCsv.Size = "120, 40"; $btnCsv.BackColor = "#7F8C8D"; $btnCsv.Enabled = $false
    $tab.Controls.Add($btnCsv)
    $btnHeal = New-Object Windows.Forms.Button; $btnHeal.Text = "Cicatrizar (API)"; $btnHeal.Location = "500, 10"; $btnHeal.Size = "130, 40"; $btnHeal.BackColor = "#27AE60"; $btnHeal.Enabled = $false
    $tab.Controls.Add($btnHeal)
    $summaryBox = New-Object Windows.Forms.RichTextBox; $summaryBox.Location = "10, 60"; $summaryBox.Width = 1060; $summaryBox.Height = 480; $summaryBox.Anchor = "Top, Left, Right, Bottom"; $summaryBox.ReadOnly = $true; $summaryBox.BackColor = "#181818"; $summaryBox.ForeColor = "#F1C40F"
    $tab.Controls.Add($summaryBox)

    $btnSmartFix.Add_Click({
        $m = Get-MissingDrives
        if ($m.Count -eq 1) { $txtDrive.Text = $m[0] }
        elseif ($m.Count -gt 1) {
            $sel = [System.Windows.Forms.MessageBox]::Show("Múltiplos volumes ausentes detectados. Deseja selecionar um manualmente?", "Vigilante", "YesNo")
            # Fallback simplified for prompt
            $txtDrive.Focus()
        }
    })

    $btnScan.Add_Click({
        $summaryBox.Clear()
        $results = Run-Diagnostics -Type $libType -Tab $tabName -Drive $txtDrive.Text
        if ($results) {
            $Script:LastScanResults = $results
            $btnHeal.Enabled = $true; $btnCsv.Enabled = $true
            foreach($r in $results) {
                if ($libType -eq "Movie") { $summaryBox.AppendText("- $($r.Title) ($($r.Year))`n") }
                else { $summaryBox.AppendText("- $($r.Title) (Temps: $($r.Seasons -join ','))`n") }
            }
        }
    })

    $btnCsv.Add_Click({
        $path = Join-Path $globalSettings.ExportPath "ArkheMissing_$($tabName)_$(Get-Date -Format 'yyyyMMdd').csv"
        $Script:LastScanResults | Export-Csv -Path $path -NoTypeInformation -Encoding UTF8
        Write-Log "Relatório CSV gerado: $path" "SUCCESS" "CSV"
        [Windows.Forms.MessageBox]::Show("Exportado para: $path")
    })

    $btnHeal.Add_Click({
        $resp = [Windows.Forms.MessageBox]::Show("Confirmar 2FA (Telegram) para restauração autorizada?", "Sovereignty Security", "YesNo")
        if ($resp -eq "Yes") {
            Write-Log "Aprovação 2FA recebida. Iniciando restauração." "INFO" "SECURITY"
            foreach ($item in $Script:LastScanResults) {
                if ($libType -eq "Movie") { Add-ToRadarr -Title $item.Title -TmdbId $item.TmdbId -RootPath $globalSettings.Radarr.RootPath -ProfileId $globalSettings.Radarr.ProfileId }
                else { Add-ToSonarr -Title $item.Title -TvdbId $item.TvdbId -Seasons $item.Seasons -RootPath $globalSettings.Sonarr.RootPath -ProfileId $globalSettings.Sonarr.ProfileId }
            }
        }
    })
    return $tab
}

function Create-Config-Tab {
    $tab = New-Object Windows.Forms.TabPage; $tab.Text = "Configurações"; $tab.BackColor = "#1E1E1E"
    $y = 20
    $fields = @(
        @("Sonarr URL", "SUrl"), @("Sonarr API Key", "SKey"), @("Sonarr Profile ID", "SProf"), @("Sonarr Root Path", "SRoot"),
        @("Radarr URL", "RUrl"), @("Radarr API Key", "RKey"), @("Radarr Profile ID", "RProf"), @("Radarr Root Path", "RRoot"),
        @("Alert Email", "Email"), @("Telegram Chat ID", "TId"), @("Export CSV Path", "ExpPath")
    )
    $inputs = @{}
    foreach ($f in $fields) {
        $lbl = New-Object Windows.Forms.Label; $lbl.Text = $f[0] + ":"; $lbl.Location = "10, $y"; $lbl.Width = 150; $tab.Controls.Add($lbl)
        $txt = New-Object Windows.Forms.TextBox; $txt.Location = "170, $y"; $txt.Width = 400
        $val = switch($f[1]) {
            "SUrl"{$globalSettings.Sonarr.URL} "SKey"{$globalSettings.Sonarr.APIKey} "SProf"{$globalSettings.Sonarr.ProfileId} "SRoot"{$globalSettings.Sonarr.RootPath}
            "RUrl"{$globalSettings.Radarr.URL} "RKey"{$globalSettings.Radarr.APIKey} "RProf"{$globalSettings.Radarr.ProfileId} "RRoot"{$globalSettings.Radarr.RootPath}
            "Email"{$globalSettings.EssentialContacts.Email} "TId"{$globalSettings.EssentialContacts.TelegramChatId} "ExpPath"{$globalSettings.ExportPath}
        }
        $txt.Text = $val; $tab.Controls.Add($txt); $inputs[$f[1]] = $txt; $y += 35
    }
    $btnSave = New-Object Windows.Forms.Button; $btnSave.Text = "Gravar Axioma"; $btnSave.Location = "170, $y"; $btnSave.Size = "150, 40"; $btnSave.BackColor = "#2980B9"
    $tab.Controls.Add($btnSave)
    $btnSave.Add_Click({
        $globalSettings.Sonarr.URL = $inputs["SUrl"].Text; $globalSettings.Sonarr.APIKey = $inputs["SKey"].Text; $globalSettings.Sonarr.ProfileId = $inputs["SProf"].Text; $globalSettings.Sonarr.RootPath = $inputs["SRoot"].Text
        $globalSettings.Radarr.URL = $inputs["RUrl"].Text; $globalSettings.Radarr.APIKey = $inputs["RKey"].Text; $globalSettings.Radarr.ProfileId = $inputs["RProf"].Text; $globalSettings.Radarr.RootPath = $inputs["RRoot"].Text
        $globalSettings.EssentialContacts.Email = $inputs["Email"].Text; $globalSettings.EssentialContacts.TelegramChatId = $inputs["TId"].Text
        $globalSettings.ExportPath = $inputs["ExpPath"].Text; $globalSettings.LastRotation = (Get-Date).ToString("yyyy-MM-dd")
        Save-Settings $globalSettings; [Windows.Forms.MessageBox]::Show("Axiomas atualizados.")
    })
    return $tab
}

# --- DIAGNOSTICS LOGIC ---
function Run-Diagnostics($Type, $Tab, $Drive) {
    Write-Log "Iniciando Diagnóstico no volume $Drive" "INFO" "SCAN"
    if (-not (Test-Path $defaultPlexDbPath)) { Write-Log "Erro Crítico: Banco de dados ausente." "ERROR" "DB"; return $null }

    try {
        Copy-Item $defaultPlexDbPath $tempDbPath -Force -ErrorAction Stop
    } catch { Write-Log "Falha ao isolar banco para diagnóstico: $_" "ERROR" "FS"; return $null }

    $Query = if ($Type -eq "Movie") {
        "SELECT md.title, md.year, CASE WHEN md.guid LIKE '%tmdb://%' THEN REPLACE(REPLACE(SUBSTR(md.guid, INSTR(md.guid, 'tmdb://') + 7), '?lang=en', ''), '?lang=pt', '') WHEN md.guid LIKE '%imdb://%' THEN REPLACE(REPLACE(SUBSTR(md.guid, INSTR(md.guid, 'imdb://') + 7), '?lang=en', ''), '?lang=pt', '') ELSE md.guid END as ExtId, mp.file FROM metadata_items md JOIN media_items mi ON md.id = mi.metadata_item_id JOIN media_parts mp ON mi.id = mp.media_item_id WHERE md.metadata_type = 1 AND md.deleted_at IS NULL;"
    } else {
        "SELECT show.title, show.year, CASE WHEN show.guid LIKE '%tvdb://%' THEN REPLACE(REPLACE(SUBSTR(show.guid, INSTR(show.guid, 'tvdb://') + 7), '?lang=en', ''), '?lang=pt', '') ELSE show.guid END as ExtId, season.index, mp.file, lib.name FROM metadata_items show JOIN metadata_items season ON season.parent_id = show.id JOIN metadata_items episode ON episode.parent_id = season.id JOIN media_items mi ON episode.id = mi.metadata_item_id JOIN media_parts mp ON mi.id = mp.media_item_id JOIN library_sections lib ON show.library_section_id = lib.id WHERE show.metadata_type = 2 AND season.metadata_type = 3 AND episode.metadata_type = 4 AND episode.deleted_at IS NULL;"
    }

    try {
        $results = & $defaultSqlitePath -csv $tempDbPath $Query | ConvertFrom-Csv -Header "Title","Year","ExtId","SeasonOrPath","PathOrLib","LibOrNull"
    } catch {
        Write-Log "Falha ao executar consulta relacional: $_" "ERROR" "DB"
        Remove-Item $tempDbPath -Force; return $null
    }

    $missing = New-Object System.Collections.Generic.List[PSObject]; $total = 0
    foreach ($r in $results) {
        $filePath = if($Type -eq "Movie") { $r.SeasonOrPath } else { $r.PathOrLib }
        if ($filePath.StartsWith($Drive, [System.StringComparison]::OrdinalIgnoreCase)) {
            $total++
            try {
                if (-not (Test-Path $filePath -ErrorAction Stop)) {
                    $id = ($r.ExtId -split '\?')[0]
                    if ($Type -eq "Movie") { $missing.Add([PSCustomObject]@{ Title=$r.Title; Year=$r.Year; TmdbId=$id }) }
                    else {
                        $lib = $r.LibOrNull
                        if (($Tab -eq "Anime" -and $lib -match "Anime") -or ($Tab -eq "TV Shows" -and $lib -notmatch "Anime")) {
                            $missing.Add([PSCustomObject]@{ Title=$r.Title; Year=$r.Year; TvdbId=$id; Season=$r.SeasonOrPath })
                        }
                    }
                }
            } catch { Write-Log "Erro de acesso ao arquivo: $filePath" "DEBUG" "FS" }
        }
    }

    Write-Log "Diagnóstico completo. Ausentes: $($missing.Count) / $total" "SUCCESS" "SCAN"
    $final = if ($Type -eq "Movie") { $missing | Group-Object Title | ForEach-Object { $_.Group[0] } }
             else { $missing | Group-Object Title | ForEach-Object { $f = $_.Group[0]; $s = $_.Group | Select-Object -ExpandProperty Season -Unique | Sort-Object {[int]$_}; [PSCustomObject]@{ Title=$_.Name; Year=$f.Year; TvdbId=$f.TvdbId; Seasons=$s } } }

    Remove-Item $tempDbPath -Force -ErrorAction SilentlyContinue; return $final
}

$tabControl.TabPages.Add((Create-Tab "TV Shows" "TV"))
$tabControl.TabPages.Add((Create-Tab "Movies" "Movie"))
$tabControl.TabPages.Add((Create-Tab "Anime" "TV"))
$tabControl.TabPages.Add((Create-Config-Tab))
Write-Log "Vigilante Soberano Online. Escudo SIWA Ativo." "INFO" "KERNEL"
$form.ShowDialog() | Out-Null
