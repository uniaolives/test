# Plex Missing Media Scanner GUI (v2.0 Arr-Ready)

A Windows PowerShell GUI tool that scans your Plex database and identifies which TV shows, movies, and anime are missing from a given drive (e.g., after a disk failure, format, or path change).

Version 2.0 is **Arr-Ready**, meaning it extracts external IDs (TVDB/TMDB) and years to allow direct import into Sonarr and Radarr.

## ✨ Features
- 🖥 **Tabbed GUI** – Separate tabs for TV Shows, Movies, and Anime.
- 📂 **Per-drive scanning** – Point it at a missing drive (e.g., `F:\`) and it finds what used to live there.
- 🔍 **Dynamic DB Discovery** – Automatically locates the Plex SQLite database via Windows Registry.
- 🧬 **Arr-Ready Metadata** – Extracts **TVDB/TMDB IDs** and **Year** for every missing item.
- 📊 **Loss Severity Metric (Φ)** – Calculates the percentage of lost media on the drive:
  - **Φ < 10%**: Natural attrition (files moved/deleted).
  - **Φ > 30%**: Sector corruption alert.
  - **Φ = 100%**: **Morte de Unidade** (Volume offline or total failure).
- 📄 **Arr-Compatible CSV** – Exported lists can be used as Custom Lists in Sonarr/Radarr.
- 🔒 **Read-only & Hygienic** – Operates on a temporary DB cache and cleans up all metadata traces after execution.
- ⚡ **Responsive UI** – Updates the log panel in real-time during long scans.

## 🛠 Requirements
- Windows 10/11
- PowerShell 5+
- Plex Media Server installed locally.
- `sqlite3.exe` (bundled or in `C:\tools\`).

### Default Paths
- **Plex database:** Discovered via Registry.
- **SQLite:** `sqlite3.exe` in the script directory or `C:\tools\sqlite3.exe`.
- **Default Output:** Your `Documents` folder.

## 🚀 Getting Started
1. Place `PlexMissingMedia_GUI.ps1` and `sqlite3.exe` in the same folder.
2. Run the script: Right-click -> **Run with PowerShell**.

## 🧭 How to Use the GUI
1. **Lost Drive/Path**: Enter the root of the missing drive (e.g., `F:\`).
2. **Scan for Missing**: The left panel shows the mapping progress and the **Loss Severity (Φ)**.
3. **Summary**: The right panel shows a human-readable list.
4. **CSV Export**: The CSV in your Documents folder is ready for Sonarr/Radarr import.

## 📂 Example Output (Arr-Ready)
### TV CSV
`Title, Year, TvdbId, Seasons, Severity, LostRoot`
`Family Guy, 1999, 75978, "1, 2, 3", 100.00%, F:\`

### Movies CSV
`Title, Year, TmdbId, Severity, LostRoot`
`Inception, 2010, 27205, 100.00%, F:\`

## ⚠️ Safety & Hygiene
The script uses a **Isolation Chamber** (temporary copy) for the database to avoid locks. Once the scan is complete, it triggers a **Hygienic Protocol** to remove the temporary file, ensuring no metadata remains outside the Plex directory.
