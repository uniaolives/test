#!/bin/bash
# deploy_binary_engine_v35.3.sh
set -euo pipefail

echo "⚡🔐 Deploy Binary Execution Engine v35.3-Ω"

# 1. Compilar
echo "🔨 Compilando Binary Engine..."
cargo build --release -p cge-constitutional-binary-engine

echo "✅✅✅ BINARY EXECUTION ENGINE v35.3-Ω IMPLANTADO!"
