# Plex Missing Media Scanner GUI (v4.0 Arkhe-Integrated)

A Windows PowerShell GUI tool that scans your Plex database and identifies which TV shows, movies, and anime are missing from a given drive (e.g., after a disk failure, format, or path change).

Version 4.0 is **Arkhe-Integrated**, featuring automatic drive detection, granular logging with timestamps, and direct integration with Sonarr and Radarr APIs.

## ✨ Features
- 🖥 **Tabbed GUI** – Separate tabs for TV Shows, Movies, Anime, and Settings.
- 📂 **Auto-Drive Detection** – One-click detection of missing volumes by analyzing your Plex database records.
- 🔍 **Dynamic DB Discovery** – Automatically locates the Plex SQLite database via Windows Registry.
- 🧬 **Sonarr/Radarr Integration** – Directly add missing items back to your automation services via their APIs.
- 📊 **Loss Severity Metric (Φ)** – Diagnostic metric for drive health.
- 📜 **Granular Logging** – Detailed logs with timestamps and color-coded errors for filesystem and database access.
- ⚙️ **Persistent Settings** – Store your API keys and URLs in `settings.json` or use environment variables.
- 🔒 **Read-only & Hygienic** – Operates on a temporary DB cache and cleans up all metadata traces after execution.

## 🛠 Requirements
- Windows 10/11
- PowerShell 5+
- Plex Media Server installed locally.
- `sqlite3.exe` (placed in the script folder or `C:\tools\`).

## 🚀 Getting Started
1. Place `PlexMissingMedia_GUI.ps1` and `sqlite3.exe` in the same folder.
2. Run the script: Right-click -> **Run with PowerShell**.
3. Go to the **Configurações** tab to set your Sonarr/Radarr URLs and API keys.

## 🧭 How to Use the GUI
1. **Detectar Volumes**: Use this button to automatically find missing drive letters used in your Plex library.
2. **Iniciar Diagnóstico**: Start the scan for missing files on the selected drive.
3. **Reintegrar via API**: After a scan, use the reintegration button to send missing items directly to Sonarr or Radarr.
4. **Log Panel**: Check the timestamped logs for detailed progress and any errors encountered.

## ⚙️ Configuration
Settings are saved to `settings.json` in the script directory. You can also use environment variables:
- `SONARR_URL`, `SONARR_API_KEY`
- `RADARR_URL`, `RADARR_API_KEY`

## ⚠️ Safety & Persona
This tool is built on the principles of the **Arkhe(n) OS**. It treats your data as a biological entity, using isolation for diagnosis and hygiene for cleanup. No modifications are made to your Plex library or configuration.
