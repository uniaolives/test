Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# --- Configuration ---
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$defaultPlexDbPath = "$env:LOCALAPPDATA\Plex Media Server\Plug-in Support\Databases\com.plexapp.plugins.library.db"
$defaultSqlitePath = Join-Path $scriptDir "sqlite3.exe"
if (-not (Test-Path $defaultSqlitePath)) {
    $defaultSqlitePath = "C:\tools\sqlite3.exe"
}
$tempDbPath = "$env:TEMP\plex_missing_media_temp.db"

# Default output folder: Documents
$defaultOutDir = [System.IO.Path]::Combine($env:USERPROFILE, "Documents")

# --- Main Form ---
$form = New-Object Windows.Forms.Form
$form.Text = "Plex Missing Media Scanner"
$form.Size = New-Object Drawing.Size(800, 600)
$form.StartPosition = "CenterScreen"

$tabControl = New-Object Windows.Forms.TabControl
$tabControl.Dock = "Fill"
$form.Controls.Add($tabControl)

# --- Tab Creation Function ---
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
    $txtCsv.Text = Join-Path $defaultOutDir "PlexMissing_$($tabName)_F_Drive.csv"
    $txtCsv.Location = New-Object Drawing.Point(150, 40)
    $txtCsv.Width = 400
    $panel.Controls.Add($txtCsv)

    $btnScan = New-Object Windows.Forms.Button
    $btnScan.Text = "Scan for Missing"
    $btnScan.Location = New-Object Drawing.Point(10, 75)
    $btnScan.Width = 120
    $panel.Controls.Add($btnScan)

    $logLabel = New-Object Windows.Forms.Label
    $logLabel.Text = "Log Panel"
    $logLabel.Location = New-Object Drawing.Point(10, 115)
    $tab.Controls.Add($logLabel)

    $summaryLabel = New-Object Windows.Forms.Label
    $summaryLabel.Text = "Human-Readable Summary"
    $summaryLabel.Location = New-Object Drawing.Point(355, 115)
    $tab.Controls.Add($summaryLabel)

    $logBox = New-Object Windows.Forms.RichTextBox
    $logBox.Location = New-Object Drawing.Point(5, 135)
    $logBox.Width = 345
    $logBox.Height = 380
    $logBox.Anchor = "Top, Left, Bottom"
    $logBox.ReadOnly = $true
    $tab.Controls.Add($logBox)

    $summaryBox = New-Object Windows.Forms.RichTextBox
    $summaryBox.Location = New-Object Drawing.Point(355, 135)
    $summaryBox.Width = 420
    $summaryBox.Height = 380
    $summaryBox.Anchor = "Top, Left, Right, Bottom"
    $summaryBox.ReadOnly = $true
    $tab.Controls.Add($summaryBox)

    $btnScan.Add_Click({
        Run-Scan -TabName $tabName -LibraryType $libraryType -Drive $txtDrive.Text -CsvPath $txtCsv.Text -LogBox $logBox -SummaryBox $summaryBox
    })

    return $tab
}

# --- Helper Function for Log Updates ---
function Update-Log($LogBox, $Message, $Color = [Drawing.Color]::Black) {
    $LogBox.SelectionStart = $LogBox.TextLength
    $LogBox.SelectionLength = 0
    $LogBox.SelectionColor = $Color
    $LogBox.AppendText($Message)
    $LogBox.ScrollToCaret()
    # Keep UI responsive during long operations
    [System.Windows.Forms.Application]::DoEvents()
}

# --- Scan Logic ---
function Run-Scan($TabName, $LibraryType, $Drive, $CsvPath, $LogBox, $SummaryBox) {
    $LogBox.Clear()
    $SummaryBox.Clear()

    Update-Log $LogBox "Starting scan for $TabName...`n"

    if (-not (Test-Path $defaultSqlitePath)) {
        Update-Log $LogBox "ERROR: sqlite3.exe not found at $defaultSqlitePath`n" [Drawing.Color]::Red
        return
    }

    if (-not (Test-Path $defaultPlexDbPath)) {
        Update-Log $LogBox "ERROR: Plex Database not found at $defaultPlexDbPath`n" [Drawing.Color]::Red
        return
    }

    Update-Log $LogBox "Copying DB to temp...`n"
    try {
        Copy-Item $defaultPlexDbPath $tempDbPath -Force -ErrorAction Stop
    } catch {
        Update-Log $LogBox "ERROR: Failed to copy database: $($_.Exception.Message)`n" [Drawing.Color]::Red
        return
    }

    Update-Log $LogBox "Querying Plex DB...`n"

    $query = ""
    if ($LibraryType -eq "Movie") {
        $query = "SELECT mi.title, mp.file FROM metadata_items mi JOIN media_items m_item ON mi.id = m_item.metadata_item_id JOIN media_parts mp ON m_item.id = mp.media_item_id WHERE mi.metadata_type = 1;"
    } else {
        # TV or Anime
        $query = "SELECT show.title, season.[index], mp.file, lib.name FROM metadata_items show JOIN metadata_items season ON season.parent_id = show.id JOIN metadata_items episode ON episode.parent_id = season.id JOIN media_items m_item ON episode.id = m_item.metadata_item_id JOIN media_parts mp ON m_item.id = mp.media_item_id JOIN library_sections lib ON show.library_section_id = lib.id WHERE show.metadata_type = 2 AND season.metadata_type = 3 AND episode.metadata_type = 4;"
    }

    # Run sqlite3.exe
    $results = & $defaultSqlitePath $tempDbPath $query "-separator" "|"

    if ($null -eq $results -or $results.Count -eq 0) {
        Update-Log $LogBox "No results found in DB query.`n"
        Remove-Item $tempDbPath -ErrorAction SilentlyContinue
        return
    }

    $missingList = New-Object System.Collections.Generic.List[PSObject]

    Update-Log $LogBox "Checking $($results.Count) items against disk...`n"

    $counter = 0
    foreach ($line in $results) {
        $counter++
        if ($counter % 100 -eq 0) {
            Update-Log $LogBox "Checked $counter items...`r"
        }

        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = $line -split '\|'
        if ($LibraryType -eq "Movie") {
            if ($parts.Count -lt 2) { continue }
            $title = $parts[0]
            $file = $parts[1]

            if ($file.StartsWith($Drive, [System.StringComparison]::OrdinalIgnoreCase)) {
                if (-not (Test-Path $file)) {
                    $missingList.Add([PSCustomObject]@{ Title = $title; File = $file })
                }
            }
        } else {
            if ($parts.Count -lt 4) { continue }
            $title = $parts[0]
            $season = $parts[1]
            $file = $parts[2]
            $libName = $parts[3]

            # --- CUSTOMIZE ANIME DETECTION HERE ---
            # By default, it looks for "Anime" in the library section name.
            $isAnimeLib = $libName -match "Anime"

            if ($TabName -eq "Anime" -and -not $isAnimeLib) { continue }
            if ($TabName -eq "TV Shows" -and $isAnimeLib) { continue }

            if ($file.StartsWith($Drive, [System.StringComparison]::OrdinalIgnoreCase)) {
                if (-not (Test-Path $file)) {
                    $missingList.Add([PSCustomObject]@{ Title = $title; Season = $season; File = $file })
                }
            }
        }
    }

    Update-Log $LogBox "`nFound $($missingList.Count) missing files.`n"

    if ($missingList.Count -gt 0) {
        Update-Log $LogBox "Writing CSV to $CsvPath...`n"

        try {
            if ($LibraryType -eq "Movie") {
                $exportData = $missingList | Group-Object Title | Sort-Object Name | ForEach-Object {
                    [PSCustomObject]@{
                        MovieTitle = $_.Name
                        LostRoot   = $Drive
                    }
                }
                $exportData | Export-Csv -Path $CsvPath -NoTypeInformation -Delimiter "," -Encoding UTF8

                $SummaryBox.AppendText("MISSING MOVIES:`n")
                foreach ($item in $exportData) {
                    $SummaryBox.AppendText("- $($item.MovieTitle)`n")
                }
            } else {
                # Group by Title and collect unique seasons
                $grouped = $missingList | Group-Object Title | Sort-Object Name
                $csvData = New-Object System.Collections.Generic.List[PSObject]

                $SummaryBox.AppendText("MISSING SERIES:`n")
                foreach ($group in $grouped) {
                    $seasons = $group.Group | Select-Object -ExpandProperty Season -Unique | Sort-Object {[int]$_}
                    $seasonStr = $seasons -join ", "
                    $SummaryBox.AppendText("- $($group.Name) Seasons $seasonStr missing`n")

                    $csvData.Add([PSCustomObject]@{
                        SeriesTitle    = $group.Name
                        SeasonsMissing = $seasonStr
                        LostRoot       = $Drive
                    })
                }
                $csvData | Export-Csv -Path $CsvPath -NoTypeInformation -Delimiter "," -Encoding UTF8
            }
            Update-Log $LogBox "Scan Complete.`n"
        } catch {
            Update-Log $LogBox "ERROR: Failed to write CSV: $($_.Exception.Message)`n" [Drawing.Color]::Red
        }
    } else {
        Update-Log $LogBox "No missing files found for this drive.`n"
    }

    # Cleanup temp DB
    Remove-Item $tempDbPath -ErrorAction SilentlyContinue
}

# --- Add Tabs ---
$tabControl.TabPages.Add((Create-Tab "TV Shows" "TV"))
$tabControl.TabPages.Add((Create-Tab "Movies" "Movie"))
$tabControl.TabPages.Add((Create-Tab "Anime" "TV"))

# --- Run Form ---
Write-Host "Plex Missing Media Scanner GUI is running..."
$form.ShowDialog() | Out-Null
