Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# --- Configuration & Discovery ---
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
            if (Test-Path $FinalPath) {
                return $FinalPath
            }
        }
    }
    return $DefaultPath
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$defaultPlexDbPath = Get-PlexDatabasePath
$defaultSqlitePath = Join-Path $scriptDir "sqlite3.exe"
if (-not (Test-Path $defaultSqlitePath)) {
    $defaultSqlitePath = "C:\tools\sqlite3.exe"
}
$tempDbPath = "$env:TEMP\plex_missing_media_temp.db"
$defaultOutDir = [System.IO.Path]::Combine($env:USERPROFILE, "Documents")

# --- Helper: GUID Parsing ---
function Get-CleanID($RawGuid) {
    if ($null -eq $RawGuid -or $RawGuid -eq "") { return "" }
    # Handles: com.plexapp.agents.thetvdb://75978?lang=en
    # Handles: plex://show/5d77682676839a001f6ec9a6
    # Handles: tvdb://12345
    if ($RawGuid -like "*://*") {
        $parts = $RawGuid -split '://'
        if ($parts.Count -gt 1) {
            $idPart = $parts[1].Split('?')[0]
            return $idPart
        }
    }
    return $RawGuid
}

# --- Main Form ---
$form = New-Object Windows.Forms.Form
$form.Text = "Plex Missing Media Scanner (v2.0 Arr-Ready)"
$form.Size = New-Object Drawing.Size(850, 700)
$form.StartPosition = "CenterScreen"

$tabControl = New-Object Windows.Forms.TabControl
$tabControl.Dock = "Fill"
$form.Controls.Add($tabControl)

$infoPanel = New-Object Windows.Forms.Panel
$infoPanel.Dock = "Bottom"
$infoPanel.Height = 30
$form.Controls.Add($infoPanel)

$lblStatus = New-Object Windows.Forms.Label
$lblStatus.Text = "Source of Truth: $defaultPlexDbPath"
$lblStatus.AutoSize = $true
$lblStatus.Location = New-Object Drawing.Point(5, 5)
$lblStatus.ForeColor = [Drawing.Color]::Gray
$infoPanel.Controls.Add($lblStatus)

# --- Tab Creation ---
function Create-Tab($tabName, $libraryType) {
    $tab = New-Object Windows.Forms.TabPage
    $tab.Text = $tabName

    $panel = New-Object Windows.Forms.Panel
    $panel.Dock = "Top"
    $panel.Height = 110
    $tab.Controls.Add($panel)

    $lblDrive = New-Object Windows.Forms.Label
    $lblDrive.Text = "Lost Drive/Path (e.g. F:\):"
    $lblDrive.Location = New-Object Drawing.Point(10, 10)
    $lblDrive.AutoSize = $true
    $panel.Controls.Add($lblDrive)

    $txtDrive = New-Object Windows.Forms.TextBox
    $txtDrive.Text = "F:\"
    $txtDrive.Location = New-Object Drawing.Point(150, 10)
    $txtDrive.Width = 200
    $panel.Controls.Add($txtDrive)

    $lblCsv = New-Object Windows.Forms.Label
    $lblCsv.Text = "Output CSV Path:"
    $lblCsv.Location = New-Object Drawing.Point(10, 40)
    $lblCsv.AutoSize = $true
    $panel.Controls.Add($lblCsv)

    $txtCsv = New-Object Windows.Forms.TextBox
    $txtCsv.Text = Join-Path $defaultOutDir "PlexMissing_$($tabName)_ArrReady.csv"
    $txtCsv.Location = New-Object Drawing.Point(150, 40)
    $txtCsv.Width = 450
    $panel.Controls.Add($txtCsv)

    $btnScan = New-Object Windows.Forms.Button
    $btnScan.Text = "Scan for Missing"
    $btnScan.Location = New-Object Drawing.Point(10, 75)
    $btnScan.Width = 140
    $panel.Controls.Add($btnScan)

    $logBox = New-Object Windows.Forms.RichTextBox
    $logBox.Location = New-Object Drawing.Point(5, 135)
    $logBox.Width = 380
    $logBox.Height = 450
    $logBox.Anchor = "Top, Left, Bottom"
    $logBox.ReadOnly = $true
    $tab.Controls.Add($logBox)

    $summaryBox = New-Object Windows.Forms.RichTextBox
    $summaryBox.Location = New-Object Drawing.Point(395, 135)
    $summaryBox.Width = 430
    $summaryBox.Height = 450
    $summaryBox.Anchor = "Top, Left, Right, Bottom"
    $summaryBox.ReadOnly = $true
    $tab.Controls.Add($summaryBox)

    $btnScan.Add_Click({
        Run-Scan -TabName $tabName -LibraryType $libraryType -Drive $txtDrive.Text -CsvPath $txtCsv.Text -LogBox $logBox -SummaryBox $summaryBox
    })

    return $tab
}

function Update-Log($LogBox, $Message, $Color = [Drawing.Color]::Black) {
    $LogBox.SelectionStart = $LogBox.TextLength
    $LogBox.SelectionLength = 0
    $LogBox.SelectionColor = $Color
    $LogBox.AppendText($Message)
    $LogBox.ScrollToCaret()
    [System.Windows.Forms.Application]::DoEvents()
}

# --- Scan Logic ---
function Run-Scan($TabName, $LibraryType, $Drive, $CsvPath, $LogBox, $SummaryBox) {
    $LogBox.Clear()
    $SummaryBox.Clear()

    Update-Log $LogBox "Linfócito de Integridade apontado para: $defaultPlexDbPath`n" [Drawing.Color]::Cyan

    if (-not (Test-Path $defaultSqlitePath)) {
        Update-Log $LogBox "ERROR: sqlite3.exe not found at $defaultSqlitePath`n" [Drawing.Color]::Red
        return
    }

    if (-not (Test-Path $defaultPlexDbPath)) {
        Update-Log $LogBox "ERROR: Plex Database not found at $defaultPlexDbPath`n" [Drawing.Color]::Red
        return
    }

    Update-Log $LogBox "Câmara de Isolamento: Criando cache temporário...`n"
    try {
        Copy-Item $defaultPlexDbPath $tempDbPath -Force -ErrorAction Stop
    } catch {
        Update-Log $LogBox "ERROR: Failed to copy database: $($_.Exception.Message)`n" [Drawing.Color]::Red
        return
    }

    Update-Log $LogBox "Querying metadata (Arr-Ready)...`n"

    $query = ""
    if ($LibraryType -eq "Movie") {
        $query = "SELECT mi.title, mi.year, mi.guid, mp.file FROM metadata_items mi JOIN media_items m_item ON mi.id = m_item.metadata_item_id JOIN media_parts mp ON m_item.id = mp.media_item_id WHERE mi.metadata_type = 1;"
    } else {
        $query = "SELECT show.title, show.year, show.guid, season.[index], mp.file, lib.name FROM metadata_items show JOIN metadata_items season ON season.parent_id = show.id JOIN metadata_items episode ON episode.parent_id = season.id JOIN media_items m_item ON episode.id = m_item.metadata_item_id JOIN media_parts mp ON m_item.id = mp.media_item_id JOIN library_sections lib ON show.library_section_id = lib.id WHERE show.metadata_type = 2 AND season.metadata_type = 3 AND episode.metadata_type = 4;"
    }

    $results = & $defaultSqlitePath $tempDbPath $query "-separator" "|"

    if ($null -eq $results -or $results.Count -eq 0) {
        Update-Log $LogBox "No results found.`n"
        Remove-Item $tempDbPath -ErrorAction SilentlyContinue
        return
    }

    $missingList = New-Object System.Collections.Generic.List[PSObject]
    $totalOnDrive = 0

    Update-Log $LogBox "Mapeando feridas informacionais...`n"

    $counter = 0
    foreach ($line in $results) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = $line -split '\|'

        $file = ""
        if ($LibraryType -eq "Movie") {
            if ($parts.Count -lt 4) { continue }
            $file = $parts[3]
        } else {
            if ($parts.Count -lt 6) { continue }
            $file = $parts[4]
            $libName = $parts[5]
            $isAnimeLib = $libName -match "Anime"
            if ($TabName -eq "Anime" -and -not $isAnimeLib) { continue }
            if ($TabName -eq "TV Shows" -and $isAnimeLib) { continue }
        }

        if ($file.StartsWith($Drive, [System.StringComparison]::OrdinalIgnoreCase)) {
            $totalOnDrive++
            $counter++
            if ($counter % 100 -eq 0) { Update-Log $LogBox "Checking item $counter...`r" }

            if (-not (Test-Path $file)) {
                if ($LibraryType -eq "Movie") {
                    $missingList.Add([PSCustomObject]@{ Title = $parts[0]; Year = $parts[1]; Guid = $parts[2]; File = $file })
                } else {
                    $missingList.Add([PSCustomObject]@{ Title = $parts[0]; Year = $parts[1]; Guid = $parts[2]; Season = $parts[3]; File = $file })
                }
            }
        }
    }

    $lossSeverity = 0
    if ($totalOnDrive -gt 0) { $lossSeverity = $missingList.Count / $totalOnDrive }

    Update-Log $LogBox "`nScan finished. Severidade de Perda (Φ): {0:P2}`n" -f $lossSeverity

    if ($lossSeverity -eq 1.0) {
        Update-Log $LogBox "⚠️ ALERTA: MORTE DE UNIDADE (Φ=100%). O volume $Drive pode estar offline.`n" [Drawing.Color]::OrangeRed
    } elseif ($lossSeverity -gt 0.3) {
        Update-Log $LogBox "⚠️ ALERTA: CORRUPÇÃO DE SETOR (Φ > 30%). Grande volume de dados ausentes.`n" [Drawing.Color]::Orange
    }

    if ($missingList.Count -gt 0) {
        Update-Log $LogBox "Gerando receita de restauração em $CsvPath...`n"

        try {
            if ($LibraryType -eq "Movie") {
                $exportData = $missingList | Group-Object Title | Sort-Object Name | ForEach-Object {
                    $first = $_.Group[0]
                    [PSCustomObject]@{
                        Title    = $_.Name
                        Year     = $first.Year
                        TmdbId   = Get-CleanID $first.Guid
                        Severity = "{0:P2}" -f $lossSeverity
                        LostRoot = $Drive
                    }
                }
                $exportData | Export-Csv -Path $CsvPath -NoTypeInformation -Delimiter "," -Encoding UTF8

                $SummaryBox.AppendText("MISSING MOVIES:`n")
                foreach ($item in $exportData) { $SummaryBox.AppendText("- $($item.Title) ($($item.Year))`n") }
            } else {
                $grouped = $missingList | Group-Object Title | Sort-Object Name
                $csvData = New-Object System.Collections.Generic.List[PSObject]

                $SummaryBox.AppendText("MISSING SERIES:`n")
                foreach ($group in $grouped) {
                    $first = $group.Group[0]
                    $seasons = $group.Group | Select-Object -ExpandProperty Season -Unique | Sort-Object {[int]$_}
                    $seasonStr = $seasons -join ", "
                    $SummaryBox.AppendText("- $($group.Name) Seasons $seasonStr missing`n")

                    $csvData.Add([PSCustomObject]@{
                        Title    = $group.Name
                        Year     = $first.Year
                        TvdbId   = Get-CleanID $first.Guid
                        Seasons  = $seasonStr
                        Severity = "{0:P2}" -f $lossSeverity
                        LostRoot = $Drive
                    })
                }
                $csvData | Export-Csv -Path $CsvPath -NoTypeInformation -Delimiter "," -Encoding UTF8
            }
            Update-Log $LogBox "Protocolo de Higiene: Limpando rastros...`n"
        } catch {
            Update-Log $LogBox "ERROR: Failed to write CSV: $($_.Exception.Message)`n" [Drawing.Color]::Red
        }
    } else {
        Update-Log $LogBox "Nenhuma ferida detectada neste drive. Paz de Fase mantida.`n" [Drawing.Color]::Green
    }

    Remove-Item $tempDbPath -Force -ErrorAction SilentlyContinue
    Update-Log $LogBox "Higiene Completa. Φ = 1.000`n" [Drawing.Color]::DarkGray
}

# --- Tabs ---
$tabControl.TabPages.Add((Create-Tab "TV Shows" "TV"))
$tabControl.TabPages.Add((Create-Tab "Movies" "Movie"))
$tabControl.TabPages.Add((Create-Tab "Anime" "TV"))

# --- Run ---
Write-Host "Arkhe(n) Restoration Module: Online"
$form.ShowDialog() | Out-Null
