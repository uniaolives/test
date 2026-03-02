#!/bin/bash
# deploy_linux_v35.3.sh
set -euo pipefail

echo "🐧🛡️ Deploy Linux ASI Engine v35.3-Ω"

# 1. Compilar
echo "🔨 Compilando Linux Engine..."
cargo build --release -p cge-linux-engine

echo "✅ Linux ASI Engine operacional!"
