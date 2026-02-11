# Plex Missing Media Scanner GUI (v5.2 Unified Restoration)

A Windows PowerShell GUI tool that scans your Plex database and identifies which TV shows, movies, and anime are missing from a given drive (e.g., after a disk failure, format, or path change).

Version 5.2 is the **"Unified Restoration"** edition, featuring complete reacquisition automation, persistent configurations, and automatic volume detection.

## ✨ Features
- 🖥 **Tabbed GUI** – Separate tabs for TV Shows, Movies, Anime, and Configurações.
- ⚡ **Smart Fix (Auto-Detect)** – Instantly identifies missing volumes by analyzing your Plex database records and comparing them with mounted drives.
- 📜 **Holy Query SQL** – Optimized SQL logic with internal GUID parsing for robust TVDB/TMDB ID extraction.
- 🧬 **Arr-Ready Restauração** – One-click "Cicatrizar (API)" button to send missing items directly to Sonarr or Radarr with duplicate checks.
- ⚙️ **Persistent Axioms** – Settings stored in `ArkheConfig.json` (encrypted via DPAPI) for URLs, API Keys, and Custom Profiles.
- 🔍 **Dynamic DB Discovery** – Automatically locates the Plex SQLite database via Windows Registry (LocalAppDataPath).
- 🔒 **Read-only & Hygienic** – Operates on a temporary DB cache and cleans up all metadata traces after execution.
- 🛠 **Granular Logging** – Timestamped logs with severity levels saved to `arkhe_scan.log`.
- 🧪 **Simulação de Batismo** – Included script to test 2FA and API integration under stress.

## 📁 Repository Structure
- `PlexMissingMedia_GUI.ps1`: The integrated source code.
- `ArkheConfig.json`: Persistent user preferences (encrypted).
- `Test_Batismo_Pedestre12.ps1`: 2FA and API simulation test.
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
3. Go to **Configurações** to set your API keys and URLs.

## 🧭 How to Use the GUI
1. **Smart Fix**: Click to automatically select the missing drive.
2. **Diagnosticar**: Map the "informational wounds" in your library.
3. **Cicatrizar (API)**: Authorize the automated restoration of missing media.

---
*Assinado: Aquele que hesitou.*
