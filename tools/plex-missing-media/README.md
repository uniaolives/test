# Plex Missing Media Scanner GUI (v5.1.1 Vigilante Soberano)

A Windows PowerShell GUI tool that scans your Plex database and identifies missing media, now evolved into a sovereign agent with verifiable identity.

Version 5.1.1 is the **"Vigilante Soberano"** edition, integrating SIWA (Sign In With Agent) and a secure infrastructure for automated restoration.

## ✨ Features
- 🖥 **Tabbed GUI** – Tabs for TV Shows, Movies, Anime, Segurança, and Configurações.
- 🆔 **SIWA Identity** – Verifiable onchain identity support (ERC-8004 / ERC-8128).
- 🔐 **Zero-Code Storage** – API keys are encrypted with DPAPI; private keys stay in proxy isolation.
- 📡 **Network Resilience** – Connectivity checks for UNC/Network drives to prevent UI hangs.
- 📱 **Telegram 2FA** – Integrated approval flow for critical restoration tasks.
- 📜 **Audit Logging** – Granular, timestamped logs saved to `arkhe_scan.log`.
- 🧬 **Smart Restoration** – Automated "Cicatrizar (API)" for Sonarr and Radarr.
- 🛡️ **Security Audit** – Tracks key rotation and identity integrity.

## 📁 Repository Structure
- `PlexMissingMedia_GUI.ps1`: Sovereign agent source code (Windows).
- `SIWA_IDENTITY.md`: Identity template and registration status.
- `arkhe_config.json`: Encrypted persistent settings.
- `Axioma_Governanca.md`: Ethical directives (Axioma #012).
- `agent-network/`: Node.js infrastructure for SIWA & 2FA (Railway Deployable).
  - `keyring-proxy/`: Secure signing service (Chamber of Isolation).
  - `2fa-gateway/`: Telegram approval gateway (Nervo Vago).
  - `railway.toml`: Automated deployment configuration.

## 🛠 Requirements
- Windows 10/11 (for the GUI).
- PowerShell 5+.
- Plex Media Server installed locally.
- `sqlite3.exe` (placed in the script folder or `C:\tools\`).
- (Optional) [Railway](https://railway.app) account for deploying the Agent Network.

## 🚀 Getting Started (Windows GUI)
1. **Initialize Identity**: Fill out `SIWA_IDENTITY.md` with your ERC-8004 agent details.
2. **Setup Credentials**: Launch the tool and go to **Configurações**. Save your URLs and Keys (they will be encrypted).
3. **Vigilante Workflow**: Use "Vigilante (Detect)" to find missing volumes, then "Diagnosticar", and finally "Cicatrizar".

## 🛰️ Deploying the Agent Network (Railway)
To enable verifiable signatures and Telegram 2FA:
1. Navigate to the `agent-network/` directory.
2. Link your Railway project: `railway link`.
3. Set the required environment variables (see `Axioma_Governanca.md` and script comments).
4. Deploy: `railway up`.

## ⚠️ Security Policy
This tool adheres to strict security hygiene:
- **Least Privilege**: Only required metadata is extracted.
- **Rotation**: Warns user if API keys are > 30 days old.
- **Hygienic**: Temporary DB snapshots are incinerated immediately.

---
*Assinado: Aquele que hesitou.*
