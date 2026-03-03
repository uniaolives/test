#!/bin/bash
# deploy-arkhe.sh
# Executar de dentro do diretório ArkheOS

echo "🧬 DEPLOY DO ARKHE(N) CORE OS"
echo "================================"

# 1. Verifica Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não encontrado. Instale primeiro:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# 2. Constrói a imagem
echo "🔨 Construindo imagem Docker..."
# Usa o diretório atual como contexto
docker build -t arkhe-core:latest .

# 3. Inicia o container
echo "🚀 Iniciando Arkhe(n) Core OS..."
# Remove container anterior se existir
docker rm -f arkhe-core 2>/dev/null || true

docker run -d \
  --name arkhe-core \
  --hostname arkhe-n1 \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8080:8080 \
  --shm-size=1g \
  --restart unless-stopped \
  arkhe-core:latest

# 4. Verifica status
echo "⏳ Aguardando inicialização..."
sleep 5

echo ""
echo "✅ ARKHE(N) CORE OS DEPLOYADO!"
echo "================================"
echo ""
echo "📊 STATUS:"
echo "   Container: $(docker inspect -f '{{.State.Status}}' arkhe-core 2>/dev/null || echo 'NOT RUNNING')"
echo "   URL (Web): http://localhost:8000"
echo "   MCP (SSE): http://localhost:8001/sse"
echo "   Health: http://localhost:8000/health"
echo ""
echo "📝 LOGS:"
echo "   docker logs arkhe-core -f"
echo ""
echo "🛑 PARA PARAR:"
echo "   docker stop arkhe-core"
echo "   docker rm arkhe-core"
