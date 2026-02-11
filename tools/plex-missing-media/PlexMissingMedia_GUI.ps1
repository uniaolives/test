Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# --- Configuration & Discovery ---
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$settingsPath = Join-Path $scriptDir "settings.json"

function Get-PlexDatabasePath {
    <#
    .SYNOPSIS
        Locates the Plex Media Server database path dynamically via Windows Registry.
    #>
    $RegistryPath = "HKCU:\Software\Plex, Inc.\Plex Media Server"
    $ValueName = "LocalAppDataPath"
    $DefaultDbSubPath = "Plex Media Server\Plug-in Support\Databases\com.plexapp.plugins.library.db"
    $DefaultPath = Join-Path $env:LOCALAPPDATA $DefaultDbSubPath

    if (Test-Path $RegistryPath) {
        $CustomPath = Get-ItemProperty -Path $RegistryPath -Name $ValueName -ErrorAction SilentlyContinue
        if ($CustomPath -and $CustomPath.$ValueName) {
            $FinalPath = Join-Path $CustomPath.$ValueName $DefaultDbSubPath
            if (Test-Path $FinalPath) { return $FinalPath }
        }
    }
    return $DefaultPath
}

# --- Settings Management ---
function Load-Settings {
    $settings = @{
        SonarrUrl = $env:SONARR_URL
        SonarrApiKey = $env:SONARR_API_KEY
        RadarrUrl = $env:RADARR_URL
        RadarrApiKey = $env:RADARR_API_KEY
        DefaultDrive = "F:\"
        ReacquisitionPath = "C:\Downloads"
    }

    if (Test-Path $settingsPath) {
        try {
            $json = Get-Content $settingsPath -Raw | ConvertFrom-Json
            foreach ($prop in $json.PSObject.Properties) {
                if ($prop.Value) { $settings[$prop.Name] = $prop.Value }
            }
        } catch {}
    }

    if (-not $settings.SonarrUrl) { $settings.SonarrUrl = "http://localhost:8989" }
    if (-not $settings.RadarrUrl) { $settings.RadarrUrl = "http://localhost:7878" }

    return $settings
}

function Save-Settings($settings) {
    $settings | ConvertTo-Json | Set-Content $settingsPath
}

$globalSettings = Load-Settings

$defaultPlexDbPath = Get-PlexDatabasePath
$defaultSqlitePath = Join-Path $scriptDir "sqlite3.exe"
if (-not (Test-Path $defaultSqlitePath)) {
    $defaultSqlitePath = "C:\tools\sqlite3.exe"
}
$tempDbPath = "$env:TEMP\plex_missing_media_temp.db"
$defaultOutDir = [System.IO.Path]::Combine($env:USERPROFILE, "Documents")

# --- Logging ---
function Update-Log($LogBox, $Message, $Color = [Drawing.Color]::Black, $IsError = $false) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $formattedMessage = "[$timestamp] $Message"

    if ($IsError) { $Color = [Drawing.Color]::Red }

    $LogBox.SelectionStart = $LogBox.TextLength
    $LogBox.SelectionLength = 0
    $LogBox.SelectionColor = $Color
    $LogBox.AppendText($formattedMessage)
    $LogBox.ScrollToCaret()
    [System.Windows.Forms.Application]::DoEvents()
}

# --- Drive Detection ---
function Get-PlexDrives($LogBox) {
    Update-Log $LogBox "Iniciando detecção automática de volumes...`n" [Drawing.Color]::Blue

    if (-not (Test-Path $defaultPlexDbPath)) {
        Update-Log $LogBox "ERRO: Banco de dados não localizado.`n" -IsError $true
        return @()
    }

    try {
        Copy-Item $defaultPlexDbPath $tempDbPath -Force -ErrorAction Stop
    } catch {
        Update-Log $LogBox "ERRO: Falha ao copiar banco: $($_.Exception.Message)`n" -IsError $true
        return @()
    }

    $query = "SELECT DISTINCT substr(file, 1, 3) FROM media_parts WHERE file IS NOT NULL;"
    try {
        $results = & $defaultSqlitePath $tempDbPath $query
    } catch {
        Update-Log $LogBox "ERRO DB: $($_.Exception.Message)`n" -IsError $true
        $results = @()
    }
    Remove-Item $tempDbPath -Force -ErrorAction SilentlyContinue

    $missingDrives = @()
    foreach ($drive in $results) {
        if ($drive -match "^[A-Z]:\\") {
            if (-not (Test-Path $drive)) {
                $missingDrives += $drive
                Update-Log $LogBox "Volume AUSENTE detectado: $drive`n" [Drawing.Color]::Orange
            }
        }
    }
    return $missingDrives
}

# --- API Integrations ---
function Add-To-Sonarr($title, $tvdbId, $seasons, $LogBox) {
    if (-not $globalSettings.SonarrApiKey) {
        Update-Log $LogBox "ERRO: API Key do Sonarr ausente.`n" -IsError $true
        return
    }

    Update-Log $LogBox "Adicionando '$title' ao Sonarr (TVDB: $tvdbId)...`n" [Drawing.Color]::Purple

    try {
        $headers = @{ "X-Api-Key" = $globalSettings.SonarrApiKey }
        $url = "$($globalSettings.SonarrUrl)/api/v3/series"

        # Format seasons for API
        $seasonsArray = @()
        foreach ($s in $seasons) {
            $seasonsArray += @{ seasonNumber = [int]$s; monitored = $true }
        }

        $body = @{
            title = $title
            tvdbId = [int]$tvdbId
            qualityProfileId = 1
            languageProfileId = 1
            rootFolderPath = $globalSettings.ReacquisitionPath
            monitored = $true
            seasons = $seasonsArray
            addOptions = @{ searchForMissingEpisodes = $true }
        } | ConvertTo-Json -Depth 5

        Invoke-RestMethod -Uri $url -Method Post -Body $body -Headers $headers -ContentType "application/json"
        Update-Log $LogBox "SUCESSO: '$title' enviado ao Sonarr.`n" [Drawing.Color]::DarkGreen
    } catch {
        Update-Log $LogBox "ERRO API Sonarr: $($_.Exception.Message)`n" -IsError $true
    }
}

function Add-To-Radarr($title, $tmdbId, $LogBox) {
    if (-not $globalSettings.RadarrApiKey) {
        Update-Log $LogBox "ERRO: API Key do Radarr ausente.`n" -IsError $true
        return
    }

    Update-Log $LogBox "Adicionando '$title' ao Radarr (TMDB: $tmdbId)...`n" [Drawing.Color]::Purple

    try {
        $headers = @{ "X-Api-Key" = $globalSettings.RadarrApiKey }
        $url = "$($globalSettings.RadarrUrl)/api/v3/movie"

        $body = @{
            title = $title
            tmdbId = [int]$tmdbId
            qualityProfileId = 1
            rootFolderPath = $globalSettings.ReacquisitionPath
            monitored = $true
            addOptions = @{ searchForMovie = $true }
        } | ConvertTo-Json

        Invoke-RestMethod -Uri $url -Method Post -Body $body -Headers $headers -ContentType "application/json"
        Update-Log $LogBox "SUCESSO: '$title' enviado ao Radarr.`n" [Drawing.Color]::DarkGreen
    } catch {
        Update-Log $LogBox "ERRO API Radarr: $($_.Exception.Message)`n" -IsError $true
    }
}

# --- Main Form ---
$form = New-Object Windows.Forms.Form
$form.Text = "Plex Missing Media Scanner (v4.1 Holy Restoration)"
$form.Size = New-Object Drawing.Size(1000, 800)
$form.StartPosition = "CenterScreen"

$tabControl = New-Object Windows.Forms.TabControl
$tabControl.Dock = "Fill"
$form.Controls.Add($tabControl)

$infoPanel = New-Object Windows.Forms.Panel
$infoPanel.Dock = "Bottom"
$infoPanel.Height = 30
$form.Controls.Add($infoPanel)

$lblStatus = New-Object Windows.Forms.Label
$lblStatus.Text = "Plex DB: $defaultPlexDbPath"
$lblStatus.AutoSize = $true
$lblStatus.Location = New-Object Drawing.Point(5, 5)
$lblStatus.ForeColor = [Drawing.Color]::DarkCyan
$infoPanel.Controls.Add($lblStatus)

# --- Tab Creation ---
function Create-Tab($tabName, $libraryType) {
    $tab = New-Object Windows.Forms.TabPage
    $tab.Text = $tabName

    $panel = New-Object Windows.Forms.Panel
    $panel.Dock = "Top"
    $panel.Height = 150
    $tab.Controls.Add($panel)

    $lblDrive = New-Object Windows.Forms.Label
    $lblDrive.Text = "Volume Perdido:"
    $lblDrive.Location = New-Object Drawing.Point(10, 10)
    $lblDrive.AutoSize = $true
    $panel.Controls.Add($lblDrive)

    $txtDrive = New-Object Windows.Forms.TextBox
    $txtDrive.Text = $globalSettings.DefaultDrive
    $txtDrive.Location = New-Object Drawing.Point(120, 10)
    $txtDrive.Width = 100
    $panel.Controls.Add($txtDrive)

    $btnDetect = New-Object Windows.Forms.Button
    $btnDetect.Text = "Detectar"
    $btnDetect.Location = New-Object Drawing.Point(230, 8)
    $btnDetect.Width = 80
    $panel.Controls.Add($btnDetect)

    $lblCsv = New-Object Windows.Forms.Label
    $lblCsv.Text = "Export CSV:"
    $lblCsv.Location = New-Object Drawing.Point(10, 40)
    $lblCsv.AutoSize = $true
    $panel.Controls.Add($lblCsv)

    $txtCsv = New-Object Windows.Forms.TextBox
    $txtCsv.Text = Join-Path $defaultOutDir "PlexMissing_$($tabName).csv"
    $txtCsv.Location = New-Object Drawing.Point(120, 40)
    $txtCsv.Width = 400
    $panel.Controls.Add($txtCsv)

    $btnScan = New-Object Windows.Forms.Button
    $btnScan.Text = "Iniciar Scan"
    $btnScan.Location = New-Object Drawing.Point(10, 75)
    $btnScan.Width = 120
    $btnScan.Height = 40
    $btnScan.BackColor = [Drawing.Color]::AliceBlue
    $panel.Controls.Add($btnScan)

    $btnAddAll = New-Object Windows.Forms.Button
    $btnAddAll.Text = "Reintegrar via API"
    $btnAddAll.Location = New-Object Drawing.Point(140, 75)
    $btnAddAll.Width = 140
    $btnAddAll.Height = 40
    $btnAddAll.Enabled = $false
    $panel.Controls.Add($btnAddAll)

    $logBox = New-Object Windows.Forms.RichTextBox
    $logBox.Location = New-Object Drawing.Point(5, 160)
    $logBox.Width = 480
    $logBox.Height = 500
    $logBox.Anchor = "Top, Left, Bottom"
    $logBox.ReadOnly = $true
    $tab.Controls.Add($logBox)

    $summaryBox = New-Object Windows.Forms.RichTextBox
    $summaryBox.Location = New-Object Drawing.Point(495, 160)
    $summaryBox.Width = 480
    $summaryBox.Height = 500
    $summaryBox.Anchor = "Top, Left, Right, Bottom"
    $summaryBox.ReadOnly = $true
    $tab.Controls.Add($summaryBox)

    $lastScanResults = New-Object System.Collections.Generic.List[PSObject]

    $btnScan.Add_Click({
        $lastScanResults.Clear()
        $results = Run-Scan -TabName $tabName -LibraryType $libraryType -Drive $txtDrive.Text -CsvPath $txtCsv.Text -LogBox $logBox -SummaryBox $summaryBox
        if ($results) {
            foreach($r in $results) { $lastScanResults.Add($r) }
            $btnAddAll.Enabled = $true
        }
    })

    $btnDetect.Add_Click({
        $drives = Get-PlexDrives -LogBox $logBox
        if ($drives.Count -gt 0) { $txtDrive.Text = $drives[0] }
    })

    $btnAddAll.Add_Click({
        foreach ($item in $lastScanResults) {
            if ($libraryType -eq "Movie") {
                Add-To-Radarr -title $item.Title -tmdbId $item.TmdbId -LogBox $logBox
            } else {
                Add-To-Sonarr -title $item.Title -tvdbId $item.TvdbId -seasons $item.Seasons -LogBox $logBox
            }
        }
    })

    return $tab
}

# --- Settings Tab ---
function Create-Settings-Tab {
    $tab = New-Object Windows.Forms.TabPage
    $tab.Text = "Configurações"

    $y = 20
    $controls = @(
        @("Sonarr URL", "SonarrUrl"),
        @("Sonarr API Key", "SonarrApiKey"),
        @("Radarr URL", "RadarrUrl"),
        @("Radarr API Key", "RadarrApiKey"),
        @("Default Drive", "DefaultDrive"),
        @("Reacquisition Path", "ReacquisitionPath")
    )

    $textboxes = @{}

    foreach ($c in $controls) {
        $lbl = New-Object Windows.Forms.Label
        $lbl.Text = $c[0] + ":"
        $lbl.Location = New-Object Drawing.Point(10, $y)
        $lbl.Width = 150
        $tab.Controls.Add($lbl)

        $txt = New-Object Windows.Forms.TextBox
        $txt.Text = $globalSettings[$c[1]]
        $txt.Location = New-Object Drawing.Point(170, $y)
        $txt.Width = 400
        $tab.Controls.Add($txt)
        $textboxes[$c[1]] = $txt

        $y += 30
    }

    $btnSave = New-Object Windows.Forms.Button
    $btnSave.Text = "Salvar Configurações"
    $btnSave.Location = New-Object Drawing.Point(170, $y + 10)
    $btnSave.Width = 150
    $tab.Controls.Add($btnSave)

    $btnSave.Add_Click({
        foreach ($key in $textboxes.Keys) {
            $globalSettings[$key] = $textboxes[$key].Text
        }
        Save-Settings $globalSettings
        [Windows.Forms.MessageBox]::Show("Salvo!")
    })

    return $tab
}

# --- Scan Logic (Holy Query) ---
function Run-Scan($TabName, $LibraryType, $Drive, $CsvPath, $LogBox, $SummaryBox) {
    $LogBox.Clear(); $SummaryBox.Clear()
    Update-Log $LogBox "Iniciando Diagnóstico...`n" [Drawing.Color]::DarkBlue

    if (-not (Test-Path $defaultSqlitePath)) { Update-Log $LogBox "sqlite3.exe ausente.`n" -IsError $true; return $null }

    Update-Log $LogBox "Câmara de Isolamento...`n"
    try { Copy-Item $defaultPlexDbPath $tempDbPath -Force -ErrorAction Stop }
    catch { Update-Log $LogBox "Falha ao isolar banco.`n" -IsError $true; return $null }

    # Advanced SQL with GUID parsing
    $query = ""
    if ($LibraryType -eq "Movie") {
        $query = @"
        SELECT md.title, md.year,
        CASE
            WHEN md.guid LIKE '%tmdb://%' THEN REPLACE(REPLACE(SUBSTR(md.guid, INSTR(md.guid, 'tmdb://') + 7), '?lang=en', ''), '?lang=pt', '')
            WHEN md.guid LIKE '%imdb://%' THEN REPLACE(REPLACE(SUBSTR(md.guid, INSTR(md.guid, 'imdb://') + 7), '?lang=en', ''), '?lang=pt', '')
            ELSE md.guid
        END as ExtId,
        mp.file FROM metadata_items md JOIN media_items mi ON md.id = mi.metadata_item_id JOIN media_parts mp ON mi.id = mp.media_item_id WHERE md.metadata_type = 1;
"@
    } else {
        $query = @"
        SELECT show.title, show.year,
        CASE
            WHEN show.guid LIKE '%tvdb://%' THEN REPLACE(REPLACE(SUBSTR(show.guid, INSTR(show.guid, 'tvdb://') + 7), '?lang=en', ''), '?lang=pt', '')
            ELSE show.guid
        END as ExtId,
        ep.parent_index as Season, mp.file, lib.name FROM metadata_items show JOIN metadata_items ep ON ep.parent_id = show.id JOIN media_items mi ON ep.id = mi.metadata_item_id JOIN media_parts mp ON mi.id = mp.media_item_id JOIN library_sections lib ON show.library_section_id = lib.id WHERE show.metadata_type = 2;
"@
    }

    try {
        $results = & $defaultSqlitePath $tempDbPath $query "-separator" "|"
    } catch {
        Update-Log $LogBox "Erro DB.`n" -IsError $true
        Remove-Item $tempDbPath -Force; return $null
    }

    $missingList = New-Object System.Collections.Generic.List[PSObject]
    $totalOnDrive = 0

    foreach ($line in $results) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = $line -split '\|'

        if ($LibraryType -eq "Movie") {
            $filePath = $parts[3]
            if ($filePath.StartsWith($Drive, [System.StringComparison]::OrdinalIgnoreCase)) {
                $totalOnDrive++
                if (-not (Test-Path $filePath)) {
                    $missingList.Add([PSCustomObject]@{ Title = $parts[0]; Year = $parts[1]; TmdbId = $parts[2]; File = $filePath })
                }
            }
        } else {
            $filePath = $parts[4]; $libName = $parts[5]
            $isAnimeLib = $libName -match "Anime"
            if ($TabName -eq "Anime" -and -not $isAnimeLib) { continue }
            if ($TabName -eq "TV Shows" -and $isAnimeLib) { continue }

            if ($filePath.StartsWith($Drive, [System.StringComparison]::OrdinalIgnoreCase)) {
                $totalOnDrive++
                if (-not (Test-Path $filePath)) {
                    $missingList.Add([PSCustomObject]@{ Title = $parts[0]; Year = $parts[1]; TvdbId = $parts[2]; Season = $parts[3]; File = $filePath })
                }
            }
        }
    }

    Update-Log $LogBox "Diagnóstico Finalizado. Perdidos: $($missingList.Count)/$totalOnDrive`n"

    $finalResults = @()
    if ($missingList.Count -gt 0) {
        if ($LibraryType -eq "Movie") {
            $finalResults = $missingList | Group-Object Title | ForEach-Object {
                $first = $_.Group[0]
                # Strip any remaining query params from ExtId
                $cleanId = $first.TmdbId -split '\?' | Select-Object -First 1
                [PSCustomObject]@{ Title = $_.Name; Year = $first.Year; TmdbId = $cleanId }
            }
        } else {
            $finalResults = $missingList | Group-Object Title | ForEach-Object {
                $first = $_.Group[0]
                $cleanId = $first.TvdbId -split '\?' | Select-Object -First 1
                $seasons = $_.Group | Select-Object -ExpandProperty Season -Unique | Sort-Object {[int]$_}
                [PSCustomObject]@{ Title = $_.Name; Year = $first.Year; TvdbId = $cleanId; Seasons = $seasons }
            }
        }
        $finalResults | Export-Csv -Path $CsvPath -NoTypeInformation
        foreach($item in $finalResults) { $SummaryBox.AppendText("- $($item.Title) ($($item.Year))`n") }
    }

    Remove-Item $tempDbPath -Force -ErrorAction SilentlyContinue
    return $finalResults
}

# --- Build GUI ---
$tabControl.TabPages.Add((Create-Tab "TV Shows" "TV"))
$tabControl.TabPages.Add((Create-Tab "Movies" "Movie"))
$tabControl.TabPages.Add((Create-Tab "Anime" "TV"))
$tabControl.TabPages.Add((Create-Settings-Tab))

# --- Run ---
$form.ShowDialog() | Out-Null
