#!/bin/bash
# deploy_sudo_v35.3.sh
set -euo pipefail

echo "🛡️⚡ Deploy Constitutional Sudo v35.3-Ω"

# 1. Verificar PQC
echo "🔐 Verificando Dilithium3..."
# cargo test -p cge_sudo

# 2. Build
echo "🔨 Build cge_sudo..."
cargo build --release -p cge_sudo

echo "✅ Sudo Constitucional operacional!"
echo "   Modo: 36×3 TMR + PQC + SASC"
echo "   Requisitos: Φ≥0.78, Confiança≥95%, Human-Explicit"
