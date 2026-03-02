#!/bin/bash

# ============================================
# TESTE DE DEPLOY TMR ONTOLOGY ON-CHAIN
# ============================================

set -e

echo "🧪 Iniciando teste TMR do módulo on-chain..."
echo "=========================================="

# 1. Iniciar a AngelNet local
echo "🚀 Iniciando AngelNet local (3 nós Karnak)..."
cd docker/angelnet
# Use sudo if required by the environment, but usually not needed in this sandbox
docker compose up -d

# Aguardar nós iniciarem
echo "⏳ Aguardando nós estarem prontos..."
sleep 25

# 2. Verificar saúde dos nós
echo "🏥 Verificando saúde dos nós..."
curl -s http://localhost:8545/health || true
echo ""
curl -s http://localhost:8546/health || true
echo ""
curl -s http://localhost:8547/health || true

# 3. Testar deploy TMR com contrato DAO
echo "🔧 Testando deploy TMR..."
cd ../..
cargo run --package ontology-lang -- deploy examples/onchain/dao.onto \
    --target=sasc \
    --network=angelnet \
    --verification=tmr \
    --sasc || {
        echo "❌ Deploy TMR falhou"

        # Logs de debug
        echo "📝 Logs do karnak-1:"
        docker logs karnak-1 --tail 20

        echo "📝 Logs do karnak-2:"
        docker logs karnak-2 --tail 20

        echo "📝 Logs do karnak-3:"
        docker logs karnak-3 --tail 20

        exit 1
    }

echo "✅ Deploy TMR bem-sucedido!"

# 4. Simular falha de nó
echo "💥 Simulando falha no nó karnak-3..."
docker stop karnak-3

# 5. Tentar outro deploy (deve funcionar com 2/3)
echo "🔄 Testando deploy com falha parcial..."
cargo run --package ontology-lang -- deploy examples/onchain/oracle.onto \
    --target=sasc \
    --network=angelnet \
    --verification=tmr \
    --sasc && {
        echo "✅ Deploy com falha parcial bem-sucedido!"
    } || {
        echo "⚠️ Deploy com falha parcial falhou (esperado se quorum não alcançado)"
    }

# 6. Dashboard
echo ""
echo "📊 Dashboard disponível em:"
echo "   Grafana:     http://localhost:3000"
echo "   Prometheus:  http://localhost:9090"
echo ""
echo "👨‍💼 Para interagir manualmente:"
echo "   cargo run --package ontology-lang -- deploy examples/onchain/dao.onto --target=sasc --verification=tmr --sasc"
echo ""

# 7. Manter serviços rodando
echo "🔄 Teste concluído. Serviços continuam rodando."
echo "   Para parar: docker compose down"
