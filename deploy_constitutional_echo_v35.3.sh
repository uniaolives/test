#!/bin/bash
# deploy_constitutional_echo_v35.3.sh
set -euo pipefail

echo "📢⚡ Deploy Constitutional Echo Engine v35.3-Ω"

# 1. Compilar
echo "🔨 Compilando Echo Engine..."
cargo build --release -p cge-constitutional-echo-engine

echo "✅✅✅ CONSTITUTIONAL ECHO ENGINE v35.3-Ω IMPLANTADO!"
