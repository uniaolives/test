# Plex Missing Media Scanner GUI (v4.3 Restoration)

A Windows PowerShell GUI tool that scans your Plex database and identifies which TV shows, movies, and anime are missing from a given drive (e.g., after a disk failure, format, or path change).

Version 4.3 is the **"Holy Restoration"** edition, featuring robust ID parsing, CSV exports, and automated reacquisition via Sonarr/Radarr.

## ✨ Features
- 🖥 **Tabbed GUI** – Separate tabs for TV Shows, Movies, Anime, and Configurações.
- 📄 **CSV Export** – Generate per-category restoration lists with external IDs (TVDB/TMDB).
- ⚡ **Smart Fix (Auto-Detect)** – Instantly identifies missing volumes by interrogating the Plex database.
- 📜 **Granular Logging** – Timestamped logs with severity levels saved to `arkhe_scan.log`.
- 🧬 **Arr-Ready Restauração** – One-click "Cicatrizar (API)" button to send missing items directly to Sonarr or Radarr with duplicate checks.
- ⚙️ **Persistent Axioms** – Settings stored in `arkhe_config.json` for persistent URLs, API Keys, and Export Paths.
- 🔒 **Read-only & Hygienic** – Operates on a temporary DB cache and cleans up all metadata traces after execution.
- 🛠 **Type-Safety** – Handles modern Plex agents (plex:// GUIDs) gracefully by validating numeric IDs for API calls.

## 📁 Repository Structure
- `PlexMissingMedia_GUI.ps1`: The integrated source code.
- `arkhe_config.json`: Persistent user preferences.
- `arkhe_scan.log`: Detailed operation history.
- `Compile_Arkhe.bat`: Batch script for executable generation.
- `Axioma_Governanca.md`: Ethical contract and preservation principles.
- `README.md`: This documentation.

## 🛠 Requirements
- Windows 10/11
- PowerShell 5+
- Plex Media Server installed locally.
- `sqlite3.exe` (placed in the script folder or `C:\tools\`).

## 🚀 Getting Started
1. Place `PlexMissingMedia_GUI.ps1` and `sqlite3.exe` in the same folder.
2. Run the script: Right-click -> **Run with PowerShell**.
3. Go to **Configurações** to set your API keys, URLs, and **Export CSV Path**.

## 🧭 How to Use the GUI
1. **Smart Fix**: Click to automatically select the missing drive.
2. **Diagnosticar**: Map the missing items in your library.
3. **Exportar CSV**: Save a filtered report for manual rebuilding.
4. **Cicatrizar (API)**: Authorize automated restoration via Sonarr/Radarr.

## ⚠️ Safety & Persona
This tool is part of the **Arkhe(n) OS**. It treats data as a biological entity: using isolation for diagnosis and hygiene for cleanup. No modifications are made to your Plex library or configuration.
