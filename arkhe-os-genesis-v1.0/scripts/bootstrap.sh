#!/bin/bash
# Executado na primeira inicialização do nó

set -e

echo "🌀 Bootstrapping Arkhe node..."

# Aguarda serviços subirem
sleep 10

# Testa handover básico
curl -X POST http://localhost:8080/handover \
  -H "Content-Type: application/json" \
  -d '{"to":"genesis","payload":"hello"}'

echo "✅ Bootstrap concluído"
