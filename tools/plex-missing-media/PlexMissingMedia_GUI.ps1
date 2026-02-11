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
            if (Test-Path $FinalPath) { return $FinalPath }
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

# --- Main Form ---
$form = New-Object Windows.Forms.Form
$form.Text = "Plex Missing Media Scanner (v3.0 Holy Query Edition)"
$form.Size = New-Object Drawing.Size(900, 750)
$form.StartPosition = "CenterScreen"

$tabControl = New-Object Windows.Forms.TabControl
$tabControl.Dock = "Fill"
$form.Controls.Add($tabControl)

$infoPanel = New-Object Windows.Forms.Panel
$infoPanel.Dock = "Bottom"
$infoPanel.Height = 30
$form.Controls.Add($infoPanel)

$lblStatus = New-Object Windows.Forms.Label
$lblStatus.Text = "Linfócito de Integridade apontado para: $defaultPlexDbPath"
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
    $panel.Height = 110
    $tab.Controls.Add($panel)

    $lblDrive = New-Object Windows.Forms.Label
    $lblDrive.Text = "Volume Perdido (ex: F:\):"
    $lblDrive.Location = New-Object Drawing.Point(10, 10)
    $lblDrive.AutoSize = $true
    $panel.Controls.Add($lblDrive)

    $txtDrive = New-Object Windows.Forms.TextBox
    $txtDrive.Text = "F:\"
    $txtDrive.Location = New-Object Drawing.Point(150, 10)
    $txtDrive.Width = 200
    $panel.Controls.Add($txtDrive)

    $lblCsv = New-Object Windows.Forms.Label
    $lblCsv.Text = "Receita de Restauração (CSV):"
    $lblCsv.Location = New-Object Drawing.Point(10, 40)
    $lblCsv.AutoSize = $true
    $panel.Controls.Add($lblCsv)

    $txtCsv = New-Object Windows.Forms.TextBox
    $txtCsv.Text = Join-Path $defaultOutDir "PlexMissing_$($tabName)_HolyScan.csv"
    $txtCsv.Location = New-Object Drawing.Point(150, 40)
    $txtCsv.Width = 450
    $panel.Controls.Add($txtCsv)

    $btnScan = New-Object Windows.Forms.Button
    $btnScan.Text = "Iniciar Diagnóstico"
    $btnScan.Location = New-Object Drawing.Point(10, 75)
    $btnScan.Width = 150
    $btnScan.BackColor = [Drawing.Color]::AliceBlue
    $panel.Controls.Add($btnScan)

    $logBox = New-Object Windows.Forms.RichTextBox
    $logBox.Location = New-Object Drawing.Point(5, 135)
    $logBox.Width = 420
    $logBox.Height = 500
    $logBox.Anchor = "Top, Left, Bottom"
    $logBox.ReadOnly = $true
    $tab.Controls.Add($logBox)

    $summaryBox = New-Object Windows.Forms.RichTextBox
    $summaryBox.Location = New-Object Drawing.Point(430, 135)
    $summaryBox.Width = 445
    $summaryBox.Height = 500
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

    Update-Log $LogBox "Iniciando Diagnóstico Arkhe(n)...`n" [Drawing.Color]::DarkBlue

    if (-not (Test-Path $defaultSqlitePath)) {
        Update-Log $LogBox "ERRO: sqlite3.exe não localizado em $defaultSqlitePath`n" [Drawing.Color]::Red
        return
    }

    if (-not (Test-Path $defaultPlexDbPath)) {
        Update-Log $LogBox "ERRO: Banco de dados não localizado.`n" [Drawing.Color]::Red
        return
    }

    Update-Log $LogBox "Câmara de Isolamento: Criando cache de transação...`n" [Drawing.Color]::DarkCyan
    try {
        Copy-Item $defaultPlexDbPath $tempDbPath -Force -ErrorAction Stop
    } catch {
        Update-Log $LogBox "ERRO: Falha ao isolar o banco: $($_.Exception.Message)`n" [Drawing.Color]::Red
        return
    }

    Update-Log $LogBox "Executando Query Sagrada ($LibraryType)...`n"

    $query = ""
    if ($LibraryType -eq "Movie") {
        $query = @"
        SELECT
            md.title AS MovieTitle,
            md.year AS Year,
            CASE
                WHEN md.guid LIKE '%themoviedb://%' THEN REPLACE(SUBSTR(md.guid, INSTR(md.guid, 'themoviedb://') + 13), '?lang=pt', '')
                WHEN md.guid LIKE '%tmdb://%' THEN REPLACE(SUBSTR(md.guid, INSTR(md.guid, 'tmdb://') + 7), '?lang=pt', '')
                ELSE md.guid
            END AS TmdbId,
            mp.file AS FilePath
        FROM metadata_items AS md
        JOIN media_items AS mi ON md.id = mi.metadata_item_id
        JOIN media_parts AS mp ON mi.id = mp.media_item_id
        WHERE md.metadata_type = 1 AND md.deleted_at IS NULL AND mp.file IS NOT NULL;
"@
    } else {
        $query = @"
        WITH series_guids AS (
            SELECT
                id, title, year, library_section_id,
                CASE
                    WHEN guid LIKE '%thetvdb://%' THEN REPLACE(SUBSTR(guid, INSTR(guid, 'thetvdb://') + 10), '?lang=pt', '')
                    WHEN guid LIKE '%tvdb://%' THEN REPLACE(SUBSTR(guid, INSTR(guid, 'tvdb://') + 7), '?lang=pt', '')
                    ELSE guid
                END AS tvdb_id
            FROM metadata_items
            WHERE metadata_type = 2 AND deleted_at IS NULL
        )
        SELECT
            sg.title AS SeriesTitle,
            sg.year AS Year,
            sg.tvdb_id AS TvdbId,
            ep.parent_index AS SeasonNumber,
            ep."index" AS EpisodeNumber,
            mp.file AS FilePath,
            lib.name AS LibName
        FROM metadata_items AS ep
        JOIN series_guids AS sg ON ep.parent_id = sg.id
        JOIN media_items AS mi ON ep.id = mi.metadata_item_id
        JOIN media_parts AS mp ON mi.id = mp.media_item_id
        JOIN library_sections AS lib ON sg.library_section_id = lib.id
        WHERE ep.metadata_type = 4 AND ep.deleted_at IS NULL AND mp.file IS NOT NULL;
"@
    }

    $results = & $defaultSqlitePath $tempDbPath $query "-separator" "|" "-header"

    if ($null -eq $results -or $results.Count -le 1) {
        Update-Log $LogBox "Nenhum dado retornado pela query.`n"
        Remove-Item $tempDbPath -ErrorAction SilentlyContinue
        return
    }

    # Header is in $results[0]
    $data = $results | Select-Object -Skip 1

    $missingList = New-Object System.Collections.Generic.List[PSObject]
    $totalOnDrive = 0

    Update-Log $LogBox "Mapeando feridas informacionais no volume $Drive...`n"

    $counter = 0
    foreach ($line in $data) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = $line -split '\|'

        if ($LibraryType -eq "Movie") {
            if ($parts.Count -lt 4) { continue }
            $filePath = $parts[3]
            if ($filePath.StartsWith($Drive, [System.StringComparison]::OrdinalIgnoreCase)) {
                $totalOnDrive++
                if (-not (Test-Path $filePath)) {
                    $missingList.Add([PSCustomObject]@{ Title = $parts[0]; Year = $parts[1]; TmdbId = $parts[2]; File = $filePath })
                }
            }
        } else {
            if ($parts.Count -lt 7) { continue }
            $filePath = $parts[5]
            $libName = $parts[6]
            $isAnimeLib = $libName -match "Anime"
            if ($TabName -eq "Anime" -and -not $isAnimeLib) { continue }
            if ($TabName -eq "TV Shows" -and $isAnimeLib) { continue }

            if ($filePath.StartsWith($Drive, [System.StringComparison]::OrdinalIgnoreCase)) {
                $totalOnDrive++
                if (-not (Test-Path $filePath)) {
                    $missingList.Add([PSCustomObject]@{ Title = $parts[0]; Year = $parts[1]; TvdbId = $parts[2]; Season = $parts[3]; Episode = $parts[4]; File = $filePath })
                }
            }
        }

        $counter++
        if ($counter % 100 -eq 0) { Update-Log $LogBox "Processados $counter itens...`r" }
    }

    $lossSeverity = 0
    if ($totalOnDrive -gt 0) { $lossSeverity = $missingList.Count / $totalOnDrive }

    Update-Log $LogBox "`nDiagnóstico Finalizado. Severidade (Φ): {0:P2}`n" -f $lossSeverity

    # Severity Thresholds (Índice de Colapso de Volume)
    if ($lossSeverity -ge 0.3) {
        Update-Log $LogBox "⚠️ ALERTA: MORTE DE UNIDADE (Severidade Φ ≥ 30%). Braço mecânico ou conexão física comprometida.`n" [Drawing.Color]::Red
    } elseif ($lossSeverity -ge 0.1) {
        Update-Log $LogBox "⚠️ ALERTA: CORRUPÇÃO DE SETOR (10% ≤ Φ < 30%). Investigue a saúde do drive.`n" [Drawing.Color]::Orange
    } elseif ($lossSeverity -gt 0) {
        Update-Log $LogBox "ℹ️ AVISO: DESGASTE NATURAL (Φ < 10%). Arquivos movidos ou deletados manualmente.`n" [Drawing.Color]::Goldenrod
    }

    if ($missingList.Count -gt 0) {
        Update-Log $LogBox "Gerando Receita de Restauração em $CsvPath...`n" [Drawing.Color]::DarkGreen

        try {
            if ($LibraryType -eq "Movie") {
                $exportData = $missingList | Group-Object Title | Sort-Object Name | ForEach-Object {
                    $first = $_.Group[0]
                    [PSCustomObject]@{
                        Title    = $_.Name
                        Year     = $first.Year
                        TmdbId   = $first.TmdbId
                        Severity = "{0:P2}" -f $lossSeverity
                        LostRoot = $Drive
                    }
                }
                $exportData | Export-Csv -Path $CsvPath -NoTypeInformation -Delimiter "," -Encoding UTF8

                $SummaryBox.AppendText("FILMES AUSENTES:`n")
                foreach ($item in $exportData) { $SummaryBox.AppendText("- $($item.Title) ($($item.Year))`n") }
            } else {
                $grouped = $missingList | Group-Object Title | Sort-Object Name
                $csvData = New-Object System.Collections.Generic.List[PSObject]

                $SummaryBox.AppendText("SÉRIES AUSENTES:`n")
                foreach ($group in $grouped) {
                    $first = $group.Group[0]
                    $seasons = $group.Group | Select-Object -ExpandProperty Season -Unique | Sort-Object {[int]$_}
                    $seasonStr = $seasons -join ","
                    $SummaryBox.AppendText("- $($group.Name) (Temp: $seasonStr)`n")

                    $csvData.Add([PSCustomObject]@{
                        Title    = $group.Name
                        Year     = $first.Year
                        TvdbId   = $first.TvdbId
                        Seasons  = $seasonStr
                        Severity = "{0:P2}" -f $lossSeverity
                        LostRoot = $Drive
                    })
                }
                $csvData | Export-Csv -Path $CsvPath -NoTypeInformation -Delimiter "," -Encoding UTF8
            }
            Update-Log $LogBox "Protocolo de Higiene: Limpando rastro de metadados...`n" [Drawing.Color]::Gray
        } catch {
            Update-Log $LogBox "ERRO ao exportar CSV: $($_.Exception.Message)`n" [Drawing.Color]::Red
        }
    } else {
        Update-Log $LogBox "Nenhuma ferida informacional detectada. Paz de Fase mantida.`n" [Drawing.Color]::DarkGreen
    }

    Remove-Item $tempDbPath -Force -ErrorAction SilentlyContinue
    Update-Log $LogBox "Higiene Completa. Φ = 1.000`n" [Drawing.Color]::DarkSlateGray
}

# --- Tabs ---
$tabControl.TabPages.Add((Create-Tab "TV Shows" "TV"))
$tabControl.TabPages.Add((Create-Tab "Movies" "Movie"))
$tabControl.TabPages.Add((Create-Tab "Anime" "TV"))

# --- Initialization ---
Write-Host "Módulo de Restauração Arkhe(n) v3.0: Online"
$form.ShowDialog() | Out-Null
