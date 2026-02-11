# Plex Missing Media Scanner GUI

A Windows PowerShell GUI tool that scans your Plex database and tells you which TV shows, movies, and anime are missing from a given drive (e.g., after a disk failure, format, or path change).

Instead of clicking through every season and episode in Plex, this tool generates a clean summary and CSV report you can use as a reacquisition checklist.

## ✨ Features
- 🖥 **Tabbed GUI** – Separate tabs for TV Shows, Movies, and Anime.
- 📂 **Per-drive scanning** – Point it at a missing drive (e.g., `F:\`) and it finds what used to live there.
- 📚 **Plex-aware** – Reads directly from the Plex SQLite database.
- 📋 **Series-level summary** – For TV/Anime it shows: `(TVSHOWNAME) Seasons 1, 2, 3 missing` instead of a huge list of every individual episode.
- 🎬 **Movies tab** – Lists movie titles that are missing from that drive.
- 📄 **CSV export** – Easy-to-filter report for rebuilding your library.
- 🔒 **Read-only** – The script only reads Plex’s DB and your file system. It does not modify anything.
- ⚡ **Responsive UI** – Updates the log panel in real-time during long scans.

## 🛠 Requirements
- Windows 10/11
- PowerShell 5+ (built-in on modern Windows)
- Plex Media Server installed on the same machine or its database available locally.
- `sqlite3.exe` (bundled with the tool or placed in `C:\tools\`).

### Default Paths
The script expects:
- **Plex database at:** `%LOCALAPPDATA%\Plex Media Server\Plug-in Support\Databases\com.plexapp.plugins.library.db`
- **SQLite at:** `C:\tools\sqlite3.exe` (or in the same directory as the script).
- **Default Output:** Your `Documents` folder.

You can change these paths in the script if your setup is different.

## 🚀 Getting Started
1. Download `PlexMissingMedia_GUI.ps1`.
2. Ensure you have `sqlite3.exe`. You can download it from the [SQLite website](https://www.sqlite.org/download.html).
3. Place them in a folder, for example: `C:\tools\`.
4. Run the script:
   - Right-click `PlexMissingMedia_GUI.ps1` and select **Run with PowerShell**.
   - Or run from a terminal: `.\PlexMissingMedia_GUI.ps1`

## 🧭 How to Use the GUI
1. **Open the tool** – you’ll see three tabs: TV Shows, Movies, and Anime.
2. **On each tab:**
   - Set the drive/root you lost, e.g.: `F:\` or `F:\Anime\`.
   - Choose an output CSV path (defaults to your Documents folder).
   - Click **"Scan for Missing"**.
3. **Check the panels:**
   - The **left-hand log panel** shows progress (Loading DB, Querying, Checking files, Writing CSV).
   - The **right-hand summary panel** shows a human-readable list of missing items.
4. **Open the CSV** in Excel or any spreadsheet tool – that’s your reacquire list.

## 📂 Example Output
### TV / Anime CSV
Columns: `SeriesTitle`, `SeasonsMissing`, `LostRoot`
Example row: `Family Guy, "1, 2, 3, 4, 5", F:\`

### Movies CSV
Columns: `MovieTitle`, `LostRoot`
Example row: `Inception, F:\`

## 🔧 Customization
You can tweak the script to:
- Change default drive (e.g., `G:\` instead of `F:\`).
- Adjust the **Anime library detection** logic (edit the `$isAnimeLib` line in the script).
- Change default output paths or Plex DB locations.

## ⚠️ Safety
The script operates in **read-only** mode. It copies the Plex DB to a temp file before querying.
It does **not** modify:
- Your Plex configuration.
- Your Plex library.
- Any media files.
