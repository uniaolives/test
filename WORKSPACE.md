# OpenClaw Workspace Management Protocol (v2.0)

## 🏗️ Architecture: Multi-Repo Hybrid Manifest
This workspace uses a hybrid strategy to manage multiple projects while avoiding the "nested Git repository" paradox.

- **Central Manifest**: `.workspace-manifest.yaml` tracks all member projects and their strategies.
- **Git Strategy**:
  - **Submodules**: For stable, independent projects.
  - **Subtrees**: For shared components that require atomic commits.
  - **Plain Directories**: For local-only or untracked experiments.

## 🛠️ Tooling
We have provided a suite of scripts in `scripts/` to manage this environment:

- `complete-migration.sh`: Full automated setup of the hybrid architecture.
- `scripts/emergency-cleanup.sh`: Detects and fixes nested `.git` directories.
- `scripts/validate-structure.py`: Checks workspace health and manifest compliance.
- `scripts/generate-manifest.py`: Automatically generates or updates the manifest.
- `scripts/install-ai-hooks.sh`: Deploys AI-native Git hooks (`commit-msg`, `pre-push`).
- `scripts/migrate-secrets.py`: Scans and migrates plaintext secrets to the Quantum Vault.
- `scripts/workspace-dashboard.py`: Generates a health report and visualization.

## 🤖 AI-Native Workflow
All commits must follow the **Conventional Commits** standard and include AI context tags when applicable:

- `@autogen`: AI-generated code.
- `@reasoning`: AI-driven decision rationale.
- `@collab`: Joint human-AI authorship.

Hooks in `.githooks/` enforce these standards and perform automated code reviews.

## 🔒 Security
Secrets are managed via the **Quantum Vault** (`security/quantum-vault.yaml`).
Never commit plaintext credentials. Run `scripts/migrate-secrets.py` if accidental exposure occurs.

## ✅ Health Checks
Run the validator daily:
```bash
python3 scripts/validate-structure.py
```
Check the dashboard for coherence and velocity metrics:
```bash
python3 scripts/workspace-dashboard.py
```
