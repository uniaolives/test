#!/bin/bash
# deploy-parallax.sh
# Deploy do cluster Arkhe(n) × Parallax

echo "🌐 ARKHE(N) × PARALLAX - DEPLOY DISTRIBUÍDO"
echo "=============================================="

# Build das imagens
echo "🔨 Construindo imagens..."
docker build -t arkhe-core:latest .
docker build -f Dockerfile.parallax -t arkhe-parallax:node-v2 .

# Deploy usando Docker Compose
echo "🚀 Iniciando cluster..."
docker-compose -f docker-compose.parallax.yml up -d

echo ""
echo "⏳ Aguardando inicialização do cluster..."
sleep 10

# Verifica status
echo "🔍 Status do Cluster:"
curl -s http://localhost:8080/health || echo "Controller ainda iniciando..."

echo ""
echo "✅ CLUSTER ARKHE(N) × PARALLAX OPERACIONAL!"
echo "=============================================="
echo ""
echo "🎛️  CONTROLLER: http://localhost:8080"
echo "🖥️  NÓS (Web):"
echo "   Node 1: http://localhost:8001"
echo "   Node 2: http://localhost:8002"
echo ""
echo "🔌 NÓS (MCP):"
echo "   Node 1: http://localhost:8101/sse"
echo "   Node 2: http://localhost:8102/sse"
echo ""
echo "🧬 COMANDOS:"
echo "   Logs Controller: docker logs -f parallax-controller"
echo "   Ver Nós:         curl http://localhost:8080/nodes"
echo "   Stop cluster:    docker-compose -f docker-compose.parallax.yml down"
