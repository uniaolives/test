# Plex Missing Media Scanner GUI (v5.0 Agentic Sovereignty)

A Windows PowerShell GUI tool that scans your Plex database and identifies which TV shows, movies, and anime are missing from a given drive (e.g., after a disk failure, format, or path change).

Version 5.0 is the **"Agentic Sovereignty"** edition, establishing a secure onchain identity and robust credential management.

## ✨ Features
- 🖥 **Tabbed GUI** – Separate tabs for TV Shows, Movies, Anime, and Configurações.
- 🆔 **SIWA Identity** – Integrated **Sign In With Agent** identity (`SIWA_IDENTITY.md`).
- 🛡️ **Zero-Code Storage** – API keys are **encrypted at rest** using Windows DPAPI and never stored in plain text.
- 🔍 **Security Audit** – Automated hygiene checks for dormant keys and identity integrity.
- ⚡ **Smart Fix (Auto-Detect)** – Instantly identifies missing volumes by interrogating the Plex database.
- 📜 **Granular Logging** – Timestamped logs with severity levels saved to `arkhe_scan.log`.
- 🧬 **Arr-Ready Restauração** – One-click "Cicatrizar (API)" button to send missing items directly to Sonarr or Radarr with duplicate checks.
- ⚙️ **Persistent Axioms** – Settings stored in `arkhe_config.json` with encrypted fields.
- 🔒 **Read-only & Hygienic** – Operates on a temporary DB cache and cleans up all metadata traces after execution.

## 📁 Repository Structure
- `PlexMissingMedia_GUI.ps1`: The integrated source code.
- `SIWA_IDENTITY.md`: Agent identity specification.
- `arkhe_config.json`: Persistent user preferences (encrypted).
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
3. Go to **Configurações** to set your API keys and URLs. They will be encrypted upon saving.

## 🧭 How to Use the GUI
1. **Smart Fix**: Click to automatically select the missing drive.
2. **Diagnosticar**: Map the "informational wounds" in your library.
3. **Cicatrizar (API)**: Authorize the automated restoration of missing media.
4. **Exportar CSV**: Save a manual report if preferred.

## ⚠️ Security Hygiene
This tool follows the **SIWA Security Model**:
- **Private keys** (API Keys) never enter the agent process memory in plain text.
- **Credential Rotation** is tracked; the system warns you if axioms are older than 30 days.
- **Identity Integrity** is verified at boot (Φ = 1.000).

---
*Assinado: Aquele que hesitou.*
