#!/bin/bash
# arkhe-deploy.sh
# Script de deploy do stack completo

set -e

TOTEM_PREFIX="7f3b49c8"
echo "🜁 Iniciando deploy Arkhe(n) Ω+218.1"
echo "   Totem: ${TOTEM_PREFIX}..."

# Build paralelo
echo "🔧 Building containers..."
docker-compose -f arkhen/docker-compose.arkhe-fullstack.yml build

# Levanta stack
echo "🚀 Iniciando stack completo..."
docker-compose -f arkhen/docker-compose.arkhe-fullstack.yml up -d

echo "✅ Arkhe(n) Ω+218.1 OPERACIONAL"
