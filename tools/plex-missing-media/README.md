# Plex Missing Media Scanner GUI (v3.0 Holy Query Edition)

A Windows PowerShell GUI tool that scans your Plex database and identifies which TV shows, movies, and anime are missing from a given drive (e.g., after a disk failure, format, or path change).

Version 3.0 is the **"Holy Query"** edition, featuring optimized SQL logic for metadata extraction and advanced "Índice de Colapso de Volume" detection.

## ✨ Features
- 🖥 **Tabbed GUI** – Separate tabs for TV Shows, Movies, and Anime.
- 📂 **Per-drive scanning** – Point it at a missing drive (e.g., `F:\`) and it finds what used to live there.
- 📜 **Holy Query SQL** – Highly optimized SQL queries that handle GUID parsing (TVDB/TMDB) directly within the database engine.
- 🧬 **Arr-Ready Metadata** – Extracts **TVDB/TMDB IDs**, **Year**, and **Seasons** for seamless Sonarr/Radarr reacquisition.
- 📊 **Loss Severity Metric (Φ)** – Calculates the **Índice de Colapso de Volume**:
  - **Φ < 10%**: **Desgaste Natural** (Routine deletion/moves).
  - **10% ≤ Φ < 30%**: **Corrupção de Setor** (Investigate drive health).
  - **Φ ≥ 30%**: **Morte de Unidade** (Disconnected volume or mechanical failure).
- 📄 **Arr-Compatible CSV** – Exported lists can be used as Custom Lists in Sonarr/Radarr.
- 🔒 **Câmara de Isolamento** – Operates on a temporary DB cache to avoid locks.
- 🧹 **Protocolo de Higiene** – Automatically cleans up all metadata traces after execution.
- ⚡ **Responsive UI** – Real-time log updates with persona-consistent terminology.

## 📁 Repository Structure
- `PlexMissingMedia_GUI.ps1`: The integrated PowerShell source code.
- `Compile_Arkhe.bat`: Batch script for executable generation.
- `Axioma_Governanca.md`: The ethical contract and preservation principles.
- `LOG_DA_CRIACAO.txt`: Brief history of the module development.
- `README.md`: This documentation.

## 🛠 Requirements
- Windows 10/11
- PowerShell 5+
- Plex Media Server installed locally.
- `sqlite3.exe` (placed in the script folder or `C:\tools\`).

### Default Paths
- **Plex database:** Discovered automatically via Windows Registry.
- **SQLite:** `sqlite3.exe` in the script directory or `C:\tools\sqlite3.exe`.
- **Default Output:** Your `Documents` folder.

## 🚀 Getting Started
1. Place `PlexMissingMedia_GUI.ps1` and `sqlite3.exe` in the same folder.
2. Run the script: Right-click -> **Run with PowerShell**.

## 🧭 How to Use the GUI
1. **Linfócito de Integridade**: Check the bottom bar to confirm the detected Plex DB path.
2. **Volume Perdido**: Enter the root of the missing drive (e.g., `F:\`).
3. **Iniciar Diagnóstico**: Observe the progress and the **Severidade (Φ)** in the log panel.
4. **Receita de Restauração**: Open the generated CSV in your Documents folder for use with Sonarr/Radarr.

## ⚠️ Safety & Persona
This tool is built on the principles of the **Arkhe(n) OS**. It treats your data as a biological entity, using isolation for diagnosis and hygiene for cleanup. No modifications are made to your Plex library or configuration.
