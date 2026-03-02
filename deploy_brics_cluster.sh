#!/bin/bash
# deploy_brics_cluster.sh
set -euo pipefail

echo "🌍🌎🌏 EXPANSÃO CGE ALPHA v35.3-Ω CLUSTER BRICS+"

# 1. Compilar
echo "🔨 Compilando BRICS Cluster..."
cargo build --release -p cge-brics-cluster

echo "✅ CLUSTER BRICS+ OPERACIONAL!"
