Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# --- ARKHE(N) OS: MÓDULO DE PRESERVAÇÃO (v4.4) ---
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$configPath = Join-Path $scriptDir "arkhe_config.json"
$logPath = Join-Path $scriptDir "arkhe_scan.log"

$Script:LastScanResults = New-Object System.Collections.Generic.List[PSObject]

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
        $color = switch($Level) { "ERROR"{[Drawing.Color]::Red} "WARN"{[Drawing.Color]::Orange} "DEBUG"{[Drawing.Color]::Gray} Default{[Drawing.Color]::LimeGreen} }
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
        try { return Get-Content $configPath -Raw | ConvertFrom-Json } catch { Write-Log "Erro ao carregar configurações." "ERROR" }
    }
    $Default = @{
        PlexDbPath = "$env:LOCALAPPDATA\Plex Media Server\Plug-in Support\Databases\com.plexapp.plugins.library.db"
        Sonarr = @{ URL = "http://localhost:8989"; APIKey = ""; RootPath = "C:\Downloads" }
        Radarr = @{ URL = "http://localhost:7878"; APIKey = ""; RootPath = "C:\Downloads" }
        ExportPath = [System.IO.Path]::Combine($env:USERPROFILE, "Documents")
        DefaultDrive = "F:\"
    }
    $Default | ConvertTo-Json | Set-Content $configPath
    return $Default
}

function Save-Settings($settings) {
    $settings | ConvertTo-Json -Depth 5 | Set-Content $configPath
    Write-Log "Configurações salvas." "INFO" "CFG"
}

$globalSettings = Load-Settings

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

# --- DRIVE DETECTION ---
function Get-MissingDrives {
    Write-Log "Interrogando volumes no banco de dados..." "INFO" "DETECT"
    if (-not (Test-Path $defaultPlexDbPath)) { Write-Log "Banco não encontrado." "ERROR"; return @() }
    try {
        Copy-Item $defaultPlexDbPath $tempDbPath -Force
        $Query = "SELECT DISTINCT SUBSTR(file, 1, 3) FROM media_parts WHERE file IS NOT NULL;"
        $DbRoots = & $defaultSqlitePath $tempDbPath $Query | ForEach-Object { $_.Trim('"').ToUpper() }
        Remove-Item $tempDbPath -Force
        $Mounted = Get-PSDrive -PSProvider FileSystem | Select-Object -ExpandProperty Root | ForEach-Object { $_.ToUpper() }
        $Missing = $DbRoots | Where-Object { $Mounted -notcontains $_ -and $_ -match "^[A-Z]:\\" }
        return $Missing
    } catch { Write-Log "Falha na detecção: $_" "ERROR" }
    return @()
}

# --- RESTORATION APIs ---
function Add-To-Sonarr($title, $tvdbId, $seasons) {
    if (-not $globalSettings.Sonarr.APIKey) { Write-Log "API Key Sonarr ausente." "ERROR"; return }
    if (-not [int]::TryParse($tvdbId, [ref]0)) {
        Write-Log "ID '$tvdbId' não é numérico. Ignorando '$title'." "WARN"
        return
    }
    try {
        $headers = @{ "X-Api-Key" = $globalSettings.Sonarr.APIKey }
        $url = "$($globalSettings.Sonarr.URL)/api/v3/series"
        $seasonsPayload = $seasons | ForEach-Object { @{ seasonNumber = [int]$_; monitored = $true } }
        $body = @{
            title = $title; tvdbId = [int]$tvdbId; qualityProfileId = 1; languageProfileId = 1
            rootFolderPath = $globalSettings.Sonarr.RootPath; monitored = $true
            seasons = $seasonsPayload; addOptions = @{ searchForMissingEpisodes = $true }
        } | ConvertTo-Json -Depth 5
        Invoke-RestMethod -Uri $url -Method Post -Body $body -Headers $headers -ContentType "application/json"
        Write-Log "Série '$title' enviada para Sonarr." "INFO" "SONARR"
    } catch { Write-Log "Erro Sonarr ($title): $($_.Exception.Message)" "ERROR" }
}

function Add-To-Radarr($title, $tmdbId) {
    if (-not $globalSettings.Radarr.APIKey) { Write-Log "API Key Radarr ausente." "ERROR"; return }
    if (-not [int]::TryParse($tmdbId, [ref]0)) {
        Write-Log "ID '$tmdbId' não é numérico. Ignorando '$title'." "WARN"
        return
    }
    try {
        $headers = @{ "X-Api-Key" = $globalSettings.Radarr.APIKey }
        $url = "$($globalSettings.Radarr.URL)/api/v3/movie"
        $body = @{
            title = $title; tmdbId = [int]$tmdbId; qualityProfileId = 1
            rootFolderPath = $globalSettings.Radarr.RootPath; monitored = $true
            addOptions = @{ searchForMovie = $true }
        } | ConvertTo-Json
        Invoke-RestMethod -Uri $url -Method Post -Body $body -Headers $headers -ContentType "application/json"
        Write-Log "Filme '$title' enviado para Radarr." "INFO" "RADARR"
    } catch { Write-Log "Erro Radarr ($title): $($_.Exception.Message)" "ERROR" }
}

# --- GUI ---
$form = New-Object Windows.Forms.Form
$form.Text = "Arkhe(n) - Módulo de Preservação (v4.4)"
$form.Size = "1000, 850"; $form.BackColor = "#121212"; $form.ForeColor = "#E0E0E0"; $form.StartPosition = "CenterScreen"
$tabControl = New-Object Windows.Forms.TabControl; $tabControl.Dock = "Fill"; $form.Controls.Add($tabControl)
$logBox = New-Object Windows.Forms.RichTextBox; $logBox.Dock = "Bottom"; $logBox.Height = 250; $logBox.BackColor = "#000000"; $logBox.ForeColor = "#00FF00"; $logBox.ReadOnly = $true; $form.Controls.Add($logBox); $global:globalLogBox = $logBox
$infoPanel = New-Object Windows.Forms.Panel; $infoPanel.Dock = "Bottom"; $infoPanel.Height = 30; $form.Controls.Add($infoPanel)
$lblStatus = New-Object Windows.Forms.Label; $lblStatus.Text = "Linfócito de Integridade: $defaultPlexDbPath"; $lblStatus.AutoSize = $true; $lblStatus.Location = "5, 5"; $lblStatus.ForeColor = "Cyan"; $infoPanel.Controls.Add($lblStatus)

function Create-Tab($tabName, $libType) {
    $tab = New-Object Windows.Forms.TabPage; $tab.Text = $tabName; $tab.BackColor = "#1E1E1E"
    $btnSmartFix = New-Object Windows.Forms.Button; $btnSmartFix.Text = "Smart Fix"; $btnSmartFix.Location = "10, 10"; $btnSmartFix.Size = "100, 40"; $btnSmartFix.BackColor = "#2C3E50"; $tab.Controls.Add($btnSmartFix)
    $lblDrive = New-Object Windows.Forms.Label; $lblDrive.Text = "Volume:"; $lblDrive.Location = "120, 20"; $lblDrive.Width = 60; $tab.Controls.Add($lblDrive)
    $txtDrive = New-Object Windows.Forms.TextBox; $txtDrive.Text = $globalSettings.DefaultDrive; $txtDrive.Location = "185, 18"; $txtDrive.Width = 60; $tab.Controls.Add($txtDrive)
    $btnScan = New-Object Windows.Forms.Button; $btnScan.Text = "Diagnosticar"; $btnScan.Location = "260, 10"; $btnScan.Size = "120, 40"; $btnScan.BackColor = "#34495E"; $tab.Controls.Add($btnScan)
    $btnCsv = New-Object Windows.Forms.Button; $btnCsv.Text = "Exportar CSV"; $btnCsv.Location = "390, 10"; $btnCsv.Size = "120, 40"; $btnCsv.BackColor = "#7F8C8D"; $btnCsv.Enabled = $false; $tab.Controls.Add($btnCsv)
    $btnHeal = New-Object Windows.Forms.Button; $btnHeal.Text = "Cicatrizar (API)"; $btnHeal.Location = "520, 10"; $btnHeal.Size = "130, 40"; $btnHeal.BackColor = "#27AE60"; $btnHeal.Enabled = $false; $tab.Controls.Add($btnHeal)
    $summaryBox = New-Object Windows.Forms.RichTextBox; $summaryBox.Location = "10, 60"; $summaryBox.Width = 960; $summaryBox.Height = 450; $summaryBox.Anchor = "Top, Left, Right, Bottom"; $summaryBox.ReadOnly = $true; $summaryBox.BackColor = "#181818"; $summaryBox.ForeColor = "#F1C40F"; $tab.Controls.Add($summaryBox)
    $btnSmartFix.Add_Click({ $m = Get-MissingDrives; if ($m) { $txtDrive.Text = $m[0] } })
    $btnScan.Add_Click({
        $summaryBox.Clear()
        $results = Run-Diagnostics -Type $libType -Tab $tabName -Drive $txtDrive.Text
        if ($results) {
            $Script:LastScanResults = $results; $btnHeal.Enabled = $true; $btnCsv.Enabled = $true
            foreach($r in $results) {
                if ($libType -eq "Movie") { $summaryBox.AppendText("- $($r.Title) ($($r.Year))`n") }
                else { $summaryBox.AppendText("- $($r.Title) (Temps: $($r.Seasons -join ','))`n") }
            }
        }
    })
    $btnCsv.Add_Click({
        $path = Join-Path $globalSettings.ExportPath "PlexMissing_$($tabName)_$(Get-Date -Format 'yyyyMMdd').csv"
        $Script:LastScanResults | Export-Csv -Path $path -NoTypeInformation -Encoding UTF8; [Windows.Forms.MessageBox]::Show("Exportado para: $path")
    })
    $btnHeal.Add_Click({
        $resp = [Windows.Forms.MessageBox]::Show("Iniciar restauração automática via API?", "Arkhe(n) OS", "YesNo")
        if ($resp -eq "Yes") {
            foreach ($item in $Script:LastScanResults) {
                if ($libType -eq "Movie") { Add-To-Radarr -title $item.Title -tmdbId $item.TmdbId }
                else { Add-To-Sonarr -title $item.Title -tvdbId $item.TvdbId -seasons $item.Seasons }
            }
        }
    })
    return $tab
}

function Create-Config-Tab {
    $tab = New-Object Windows.Forms.TabPage; $tab.Text = "Configurações"; $tab.BackColor = "#1E1E1E"
    $y = 20
    $fields = @(@("Sonarr URL", "SonarrUrl"), @("Sonarr API Key", "SonarrApiKey"), @("Sonarr Root Path", "SRootPath"), @("Radarr URL", "RadarrUrl"), @("Radarr API Key", "RadarrApiKey"), @("Radarr Root Path", "RRootPath"), @("Export CSV Path", "ExportPath"))
    $inputs = @{}
    foreach ($f in $fields) {
        $lbl = New-Object Windows.Forms.Label; $lbl.Text = $f[0] + ":"; $lbl.Location = "10, $y"; $lbl.Width = 150; $tab.Controls.Add($lbl)
        $txt = New-Object Windows.Forms.TextBox; $txt.Location = "170, $y"; $txt.Width = 400
        $val = switch($f[1]) { "SonarrUrl"{$globalSettings.Sonarr.URL} "SonarrApiKey"{$globalSettings.Sonarr.APIKey} "SRootPath"{$globalSettings.Sonarr.RootPath} "RadarrUrl"{$globalSettings.Radarr.URL} "RadarrApiKey"{$globalSettings.Radarr.APIKey} "RRootPath"{$globalSettings.Radarr.RootPath} "ExportPath"{$globalSettings.ExportPath} }
        $txt.Text = $val; $tab.Controls.Add($txt); $inputs[$f[1]] = $txt; $y += 35
    }
    $btnSave = New-Object Windows.Forms.Button; $btnSave.Text = "Gravar Configs"; $btnSave.Location = "170, $y"; $btnSave.Size = "150, 40"; $btnSave.BackColor = "#2980B9"; $tab.Controls.Add($btnSave)
    $btnSave.Add_Click({
        $globalSettings.Sonarr.URL = $inputs["SonarrUrl"].Text; $globalSettings.Sonarr.APIKey = $inputs["SonarrApiKey"].Text; $globalSettings.Sonarr.RootPath = $inputs["SRootPath"].Text
        $globalSettings.Radarr.URL = $inputs["RadarrUrl"].Text; $globalSettings.Radarr.APIKey = $inputs["RadarrApiKey"].Text; $globalSettings.Radarr.RootPath = $inputs["RRootPath"].Text
        $globalSettings.ExportPath = $inputs["ExportPath"].Text; Save-Settings $globalSettings; [Windows.Forms.MessageBox]::Show("Salvo!")
    })
    return $tab
}

# --- DIAGNOSTICS LOGIC ---
function Run-Diagnostics($Type, $Tab, $Drive) {
    Write-Log "Iniciando Diagnóstico no volume $Drive" "INFO" "SCAN"
    if (-not (Test-Path $defaultPlexDbPath)) { Write-Log "Banco ausente." "ERROR"; return $null }
    Copy-Item $defaultPlexDbPath $tempDbPath -Force
    $Query = if ($Type -eq "Movie") {
        @"
SELECT md.title, md.year,
CASE
    WHEN md.guid LIKE '%tmdb://%' THEN REPLACE(REPLACE(SUBSTR(md.guid, INSTR(md.guid, 'tmdb://') + 7), '?lang=en', ''), '?lang=pt', '')
    WHEN md.guid LIKE '%imdb://%' THEN REPLACE(REPLACE(SUBSTR(md.guid, INSTR(md.guid, 'imdb://') + 7), '?lang=en', ''), '?lang=pt', '')
    ELSE md.guid
END as ExtId, mp.file
FROM metadata_items md JOIN media_items mi ON md.id = mi.metadata_item_id JOIN media_parts mp ON mi.id = mp.media_item_id
WHERE md.metadata_type = 1 AND md.deleted_at IS NULL;
"@
    } else {
        @"
SELECT show.title, show.year,
CASE
    WHEN show.guid LIKE '%tvdb://%' THEN REPLACE(REPLACE(SUBSTR(show.guid, INSTR(show.guid, 'tvdb://') + 7), '?lang=en', ''), '?lang=pt', '')
    ELSE show.guid
END as ExtId, season.index as Season, mp.file, lib.name
FROM metadata_items show
JOIN metadata_items season ON season.parent_id = show.id
JOIN metadata_items episode ON episode.parent_id = season.id
JOIN media_items mi ON episode.id = mi.metadata_item_id
JOIN media_parts mp ON mi.id = mp.media_item_id
JOIN library_sections lib ON show.library_section_id = lib.id
WHERE show.metadata_type = 2 AND season.metadata_type = 3 AND episode.metadata_type = 4 AND episode.deleted_at IS NULL;
"@
    }
    try { $results = & $defaultSqlitePath $tempDbPath $Query "-separator" "|" } catch { Write-Log "Erro Query." "ERROR"; Remove-Item $tempDbPath -Force; return $null }
    $missing = New-Object System.Collections.Generic.List[PSObject]; $total = 0
    foreach ($line in $results) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $p = $line -split '\|'; $filePath = if($Type -eq "Movie") { $p[3] } else { $p[4] }
        if ($filePath.StartsWith($Drive, [System.StringComparison]::OrdinalIgnoreCase)) {
            $total++
            try {
                if (-not (Test-Path $filePath -ErrorAction Stop)) {
                    $cleanId = ($p[2] -split '\?')[0]
                    if ($Type -eq "Movie") { $missing.Add([PSCustomObject]@{ Title=$p[0]; Year=$p[1]; TmdbId=$cleanId }) }
                    else {
                        if ($Tab -eq "Anime" -and $p[5] -notmatch "Anime") { continue }
                        if ($Tab -eq "TV Shows" -and $p[5] -match "Anime") { continue }
                        $missing.Add([PSCustomObject]@{ Title=$p[0]; Year=$p[1]; TvdbId=$cleanId; Season=$p[3] })
                    }
                }
            } catch { Write-Log "Acesso negado: $filePath" "DEBUG" "FS" }
        }
    }
    Write-Log "Diagnóstico completo. Ausentes: $($missing.Count) / $total" "INFO" "SCAN"
    $final = if ($Type -eq "Movie") { $missing | Group-Object Title | ForEach-Object { $_.Group[0] } }
             else { $missing | Group-Object Title | ForEach-Object { $f = $_.Group[0]; $s = $_.Group | Select-Object -ExpandProperty Season -Unique | Sort-Object {[int]$_}; [PSCustomObject]@{ Title=$_.Name; Year=$f.Year; TvdbId=$f.TvdbId; Seasons=$s } } }
    Remove-Item $tempDbPath -Force -ErrorAction SilentlyContinue; return $final
}

$tabControl.TabPages.Add((Create-Tab "TV Shows" "TV"))
$tabControl.TabPages.Add((Create-Tab "Movies" "Movie"))
$tabControl.TabPages.Add((Create-Tab "Anime" "TV"))
$tabControl.TabPages.Add((Create-Config-Tab))
Write-Log "Arkhe(n) OS Módulo de Preservação ONLINE. Φ = 1.000" "INFO" "KERNEL"
$form.ShowDialog() | Out-Null
